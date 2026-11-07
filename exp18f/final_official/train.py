"""BERT-Tiny ghost clipping plus the exact Exp18 optimizer/noise implementations."""
import os
import sys
import time
from pathlib import Path

from exp18f.final_official.common import (BASE, ROOT, BATCH, K, T, configs, outdir, now, read, save, sha,
                           load_matrix, WORKLOADS, verify_seal, calibration, verify_freeze)
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from exp18f.final_official.model import make_model, seed_all, digest
from exp18f.final_official.clipping import StandardGhostModule, ScaledGhostModule, clipped_microbatch
from exp18f.final_official.optimizer import Update
from exp18f.final_official.noise import OnlineNoise


@torch.no_grad()
def evaluate(model, loader):
    model.eval(); model.disable_hooks()
    loss = correct = count = 0
    for batch in loader:
        batch = tuple(t.cuda() for t in batch)
        logits = model(batch[:3])
        loss += float(torch.nn.functional.cross_entropy(logits,batch[3],reduction='sum'))
        correct += int((logits.argmax(1)==batch[3]).sum()); count += len(batch[3])
    model.enable_hooks(); model.train()
    assert count and np.isfinite(loss)
    return dict(accuracy=correct/count,loss=loss/count,examples=count)


def clipped_sum(model, inputs, targets, opt, clip):
    # d is frozen throughout both backwards; hooks measure ||g_i / d||.
    if opt.group == 'stpex2':
        offset = 0
        for p in opt.ps:
            p._logical_scale = opt.d[offset:offset+p.numel()].view_as(p).reciprocal()
            offset += p.numel()
    model.zero_grad(set_to_none=True)
    loss, clips = clipped_microbatch(model,inputs,targets,clip)
    q = torch.cat([p.grad.flatten() for p in opt.ps])
    if opt.group == 'stpex2': q.div_(opt.d)
    return q, loss, clips


def run(path):
    assert '/envs/curve/' in sys.executable
    started = time.monotonic(); cfg = read(path); out = path.parent
    assert cfg in configs(cfg['stage'])
    verify_seal()
    if cfg['stage'] == 'formal': verify_freeze()
    assert torch.cuda.device_count()==1
    torch.cuda.set_device(0); seed_all(cfg['seed'])
    base = make_model(); initial_hash = digest(base)
    wrapper = ScaledGhostModule if cfg['group']=='stpex2' else StandardGhostModule
    model = wrapper(base.cuda(),cfg['clip']); opt = Update(model,cfg['group'],cfg['lr'],cfg['eps'])
    assert all(p.requires_grad and p.dtype==torch.float32 for p in opt.ps)
    iid = cfg['method'].endswith('-iid')
    matrix = {'metadata':{}} if iid else load_matrix(WORKLOADS[cfg['group']])
    privacy = calibration(cfg)
    noise = OnlineNoise(opt.ps,matrix,'iid' if iid else 'b2lt',privacy['sigma_sum'],T,cfg['seed']+1)
    tensors = torch.load(ROOT / 'exp18e/assets/nlp_tokens_128.pt',weights_only=True)
    with np.load(ROOT / 'exp18e/assets/nlp_split.npz') as split:
        train = split['train']; valid = split['search_validation']
    order = train[torch.randperm(len(train),generator=torch.Generator().manual_seed(cfg['seed'])).numpy()]
    keys = ('input_ids','attention_mask','token_type_ids','labels')
    def subset(indices): return TensorDataset(*(tensors[k][indices] for k in keys))
    training = subset(order)
    validation = subset(valid[:64] if cfg['stage']=='smoke' else valid)
    loader = DataLoader(validation,batch_size=100,shuffle=False)
    np.save(out / 'train_order.npy',order)
    pairing = dict(initial_model_sha256=initial_hash,order_sha256=sha(out / 'train_order.npy'),
                   noise_seed=cfg['seed']+1,dropout_seed='seed+100000+logical_step')
    save(out / 'pairing.json',pairing); save(out / 'privacy.json',privacy)
    save(out / 'process.json',dict(pid=os.getpid(),time=now(),physical_gpu=int(os.environ['EXP18F_FINAL_GPU']),
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],visible_device_count=torch.cuda.device_count(),
        device=str(torch.cuda.get_device_properties(0)),python=sys.executable))
    if cfg['stage']=='smoke':
        save(out / 'ready.json',dict(pid=os.getpid(),gpu=int(os.environ['EXP18F_FINAL_GPU']),time=now()))
        deadline = time.monotonic()+120
        while len(list((BASE / 'smoke').glob('**/ready.json'))) < 6:
            assert time.monotonic()<deadline,'Six-worker readiness timeout'
            time.sleep(.2)
    curve = []; train_seconds = 0
    torch.cuda.reset_peak_memory_stats()
    with (out / 'steps.jsonl').open('w',buffering=1) as log:
        for step in range(cfg['steps']):
            torch.cuda.synchronize(); tick = time.monotonic()
            torch.manual_seed(cfg['seed']+100000+step); torch.cuda.manual_seed(cfg['seed']+100000+step)
            start = (step%K)*BATCH
            batch = tuple(t[start:start+BATCH].cuda() for t in training.tensors)
            assert len(batch[3])==BATCH
            q,loss,clips = clipped_sum(model,batch[:3],batch[3],opt,cfg['clip'])
            q.add_(torch.cat([z.flatten() for z in noise.next()])).div_(BATCH)
            assert torch.isfinite(q).all()
            alpha = opt.step(q)
            assert noise.step_count==opt.t==step+1
            if not all(torch.isfinite(p).all() for p in opt.ps):
                raise FloatingPointError('Nonfinite model parameter')
            model.zero_grad(set_to_none=True)
            del batch,q
            torch.cuda.synchronize(); train_seconds += time.monotonic()-tick
            import json
            log.write(json.dumps(dict(step=step+1,epoch=step//K+1,train_loss=loss/BATCH,
                clipped_fraction=clips/BATCH,alpha=alpha,seconds=time.monotonic()-tick,
                gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],allocated_bytes=torch.cuda.memory_allocated(),
                reserved_bytes=torch.cuda.memory_reserved(),peak_reserved_bytes=torch.cuda.max_memory_reserved()),allow_nan=False)+'\n')
            if (step+1)%K==0 or step+1==cfg['steps']:
                metrics = evaluate(model,loader)
                curve.append(dict(step=step+1,internal=metrics)); save(out / 'curve.json',curve)
                print(f"{cfg['method']} seed={cfg['seed']} step={step+1} internal={metrics}",flush=True)
    if cfg['stage']=='formal':
        official = torch.load(ROOT / 'exp18e/assets/nlp_official_tokens_128.pt',weights_only=True)
        metrics = evaluate(model,DataLoader(TensorDataset(*(official[k] for k in keys)),batch_size=100))
        assert metrics['examples']==872 and opt.t==310
    checkpoint = out / 'checkpoint.pt'
    torch.save(dict(model=base.state_dict(),optimizer=opt.state_dict(),config=cfg,privacy=privacy,
                    pairing=pairing,noise_generator_state=noise.generator.get_state()),checkpoint)
    summary = dict(status='complete',formal=cfg['stage']=='formal',config=cfg,steps=opt.t,
        evaluation='official_validation' if cfg['stage']=='formal' else ('internal_smoke64' if cfg['stage']=='smoke' else 'internal_validation'),
        metrics=metrics,privacy=privacy,pairing=pairing,seconds=time.monotonic()-started,
        train_seconds=train_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],physical_gpu=int(os.environ['EXP18F_FINAL_GPU']),ended=now())
    save(out / 'summary.json',summary)
    save(out / 'complete.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()
        and p.name not in ('complete.json','worker.log')})
    print(f"COMPLETE {cfg['method']} seed={cfg['seed']}: {metrics}",flush=True)

if __name__ == '__main__': run(Path(sys.argv[1]))
