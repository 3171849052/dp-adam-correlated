"""Offline FP32 worker, independent Poisson sampling, exact state recovery."""
import argparse
import json
import os
import random
import time
import traceback
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from opacus.accountants import RDPAccountant
from exp17b.common import RESULTS, PROTOCOL, save, read, sha, output, code_hashes, assets, success
from exp17b.cv_model import pretrained_vit, initialization_digest
from exp17b.data import cv_datasets
from exp17b.clipping import StandardGhostModule, clipped_microbatch
from exp17b.optimizer import PrivateAdam
from exp17b.mechanism import poisson_indices, privatize


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)


@torch.no_grad()
def evaluate(model, dataset):
    model.eval();loss=correct=n=0
    for x,y in DataLoader(dataset,batch_size=250,shuffle=False,num_workers=0):
        x,y=x.cuda(),y.cuda();logits=model(x)
        loss+=torch.nn.functional.cross_entropy(logits,y,reduction='sum').item()
        correct+=(logits.argmax(1)==y).sum().item();n+=len(y)
    if not np.isfinite(loss): raise FloatingPointError('Nonfinite evaluation')
    return dict(accuracy=correct/n,validation_loss=loss/n,examples=n)


def atomic_checkpoint(path, state):
    temp=path.with_suffix('.tmp');torch.save(state,temp);os.replace(temp,path)


def run(cfg):
    folder=output(cfg);folder.mkdir(parents=True,exist_ok=True)
    frozen=read(RESULTS/'protocol_freeze.json')
    if frozen['code']!=code_hashes() or frozen['assets']!=assets(): raise RuntimeError('Source/input freeze mismatch')
    identity=sha(RESULTS/'protocol_freeze.json')
    if success(cfg,identity): return
    if (folder/'summary.json').exists(): raise RuntimeError('Failed trial retained; inspect before manual recovery')
    if cfg['stage']!='smoke':
        authorized=read(RESULTS/'authorized.json')
        if cfg not in authorized: raise RuntimeError('Trial not authorized by frozen search/final plan')
    formal=cfg['stage']=='D'
    if formal and not (RESULTS/'final_freeze.json').exists(): raise RuntimeError('Official test is locked until final freeze')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2','3') or torch.cuda.device_count()!=1:
        raise RuntimeError('Exactly one physical GPU in 2,3 required')
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    start=time.monotonic();model=None
    try:
        seed_all(cfg['seed'])
        base,metadata=pretrained_vit();init_hash=initialization_digest(base)
        stage='final' if formal else 'search'
        if cfg['stage']=='smoke': stage=cfg.get('horizon','search')
        privacy=frozen['calibration'][stage]
        data_cfg=SimpleNamespace(formal=stage=='final',stage='final' if formal else 'search',seed=cfg['seed'])
        training,validation,order=cv_datasets(data_cfg,metadata)
        assert len(training)==privacy['population']
        assert len(validation)==(10000 if formal else 5000)
        assert len(set(order.tolist()))==len(order)
        if cfg['stage']=='smoke': validation=Subset(validation,range(64))
        model=StandardGhostModule(base.to(device),30.)
        params=list(model.parameters())
        if not all(p.dtype==torch.float32 and p.requires_grad for p in params): raise ValueError('Full FP32 required')
        opt=PrivateAdam(params,cfg['method'],cfg['lr'],privacy['nu_squared'],cfg.get('r'),cfg['alpha'])
        accountant=RDPAccountant()
        sampler=torch.Generator().manual_seed(cfg['seed']+1000000)
        noise=torch.Generator(device=device).manual_seed(cfg['seed']+2000000)
        total_steps=2 if cfg['stage']=='smoke' else cfg['steps']
        rows=[];evaluations=[];selections=[];completed=0;prior_seconds=0
        checkpoint=folder/'checkpoint.pt'
        resolved=dict(config=cfg,identity=identity,initialization_sha256=init_hash,metadata=metadata,
            physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],privacy=privacy,
            official_assets=assets(True) if formal else None)
        if (folder/'config.json').exists():
            previous=read(folder/'config.json')
            if any(previous[k]!=resolved[k] for k in ('config','identity','metadata','official_assets','initialization_sha256')):
                raise RuntimeError('Checkpoint config/input mismatch')
        else: save(folder/'config.json',resolved)
        if checkpoint.exists():
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if state['identity']!=identity or state['config']!=cfg: raise RuntimeError('Checkpoint identity mismatch')
            model._module.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
            accountant.load_state_dict(state['accountant']);sampler.set_state(state['sampler']);noise.set_state(state['noise'])
            torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'])
            random.setstate(state['random']);np.random.set_state(state['numpy'])
            completed=state['step'];rows=state['rows'];evaluations=state['evaluations'];selections=state['selections']
            prior_seconds=state['seconds']
            del state
        np.save(folder/'train_order.npy',order)
        (folder/'diagnostics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        torch.cuda.reset_peak_memory_stats()
        def checkpoint_now(step):
            atomic_checkpoint(checkpoint,dict(identity=identity,config=cfg,step=step,
                model=model._module.state_dict(),optimizer=opt.state_dict(),accountant=accountant.state_dict(),
                sampler=sampler.get_state(),noise=noise.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),
                random=random.getstate(),numpy=np.random.get_state(),rows=rows,evaluations=evaluations,
                selections=selections,seconds=prior_seconds+time.monotonic()-start))
        if not checkpoint.exists(): checkpoint_now(0)
        for step in range(completed+1,total_steps+1):
            torch.cuda.synchronize();step_start=time.monotonic();model.train()
            indices=poisson_indices(len(training),privacy['sample_rate'],sampler)
            selections.append(indices.numpy())
            sums=[torch.zeros_like(p) for p in params];loss=clips=0;norm_sum=0.;norm_max=0.
            # num_workers=0 makes augmentation RNG exactly recoverable at logical-step boundaries.
            loader=DataLoader(Subset(training,indices.tolist()),batch_size=PROTOCOL['physical_batch'],
                              shuffle=False,num_workers=0,pin_memory=True)
            for x,y in loader:
                model.zero_grad(set_to_none=True)
                part,nclip=clipped_microbatch(model,x.to(device),y.to(device),30.)
                loss+=part;clips+=nclip;norm_sum+=model.last_norms.sum().item()
                norm_max=max(norm_max,model.last_norms.max().item())
                with torch.no_grad():
                    for acc,p in zip(sums,params): acc.add_(p.grad)
            privatize(params,sums,privacy['noise_multiplier'],noise)
            del sums
            opt.step();accountant.step(noise_multiplier=privacy['noise_multiplier'],sample_rate=privacy['sample_rate'])
            epsilon=accountant.get_epsilon(delta=1e-5)
            if epsilon>8.000001: raise RuntimeError('Privacy budget exceeded')
            torch.cuda.synchronize()
            row=dict(step=step,batch_size=len(indices),train_loss=loss/len(indices) if len(indices) else None,
                clip_fraction=clips/len(indices) if len(indices) else None,
                gradient_norm_mean=norm_sum/len(indices) if len(indices) else None,
                gradient_norm_max=norm_max,epsilon=epsilon,noise_draws=step,optimizer_steps=step,
                seconds=time.monotonic()-step_start,**opt.diagnostics)
            rows.append(row)
            with (folder/'diagnostics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            if step % (privacy['steps']//5)==0 or step==total_steps:
                evaluation=dict(step=step,**evaluate(model,validation));evaluations.append(evaluation)
                save(folder/'evaluations.json',evaluations);print(json.dumps(evaluation),flush=True)
            save(folder/'progress.json',dict(step=step,total=total_steps,pid=os.getpid(),gpu=os.environ['CUDA_VISIBLE_DEVICES'],last=row))
            if step%10==0 or step==total_steps: checkpoint_now(step)
        if total_steps==privacy['steps'] and abs(epsilon-privacy['actual_epsilon'])>1e-8: raise RuntimeError('RDP final mismatch')
        np.savez_compressed(folder/'poisson_samples.npz',**{f'step_{i+1}':a for i,a in enumerate(selections)})
        files=('config.json','checkpoint.pt','train_order.npy','poisson_samples.npz','evaluations.json','diagnostics.jsonl')
        save(folder/'summary.json',dict(status='completed',config=cfg,identity=identity,steps=total_steps,
            **evaluations[-1],actual_epsilon=epsilon,calibration=privacy,
            seconds=prior_seconds+time.monotonic()-start,train_seconds=sum(r['seconds'] for r in rows),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],device=torch.cuda.get_device_name(),
            evaluation_scope='official_test' if formal else 'internal_64_smoke' if cfg['stage']=='smoke' else 'internal_validation',
            hashes={name:sha(folder/name) for name in files}))
    except Exception as error:
        save(folder/'summary.json',dict(status='failed',config=cfg,identity=identity,error=repr(error),traceback=traceback.format_exc()))
        raise
    finally:
        if model is not None: model.remove_hooks()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    run(json.loads(parser.parse_args().config))
