"""Offline worker: two-step smoke prefixes and complete five-epoch frozen trials."""
import argparse
import hashlib
import json
import os
import time
import traceback
import resource
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from exp16e import ROOT,RESULTS
from exp16e.config import Trial, file_hash, array_hash, save_json,json_hash
from exp16e.clipping import StandardGhostModule, ScaledGhostModule, clipped_microbatch
from exp16e.optimizer import make_optimizer, LogicalBatch, unclipped_microbatch
from exp16e.noise import OnlineNoise
from exp16e.privacy import calibrate, fixed_epoch_sensitivity, epsilon_from_mu
from exp16e.nlp_model import seed_all, make_model, digest
from exp16e.data import cv_datasets, nlp_datasets
from exp16e.support import code_hashes, version, write_csv,verify_assets,official_assets
from exp16e.matrices import load


@torch.no_grad()
def evaluate(model, loader, device, task):
    model.eval(); total_loss=correct=count=0
    for batch in loader:
        batch=tuple(t.to(device) for t in batch)
        inputs,targets=(batch[0],batch[1]) if task=='cv' else (batch[:3],batch[3])
        logits=model(inputs)
        total_loss+=float(torch.nn.functional.cross_entropy(logits,targets,reduction='sum'))
        correct+=int((logits.argmax(1)==targets).sum());count+=len(targets)
    if not np.isfinite(total_loss) or count==0: raise FloatingPointError('Invalid evaluation')
    return dict(accuracy=correct/count,validation_loss=total_loss/count,validation_examples=count)


def run(cfg):
    if cfg.stage in ('search','review','final'):
        from exp16e.search import authorize
        authorize(cfg)
    started=time.monotonic();output=cfg.output
    if (output/'config.json').exists(): raise FileExistsError(f'Run already exists or incomplete: {output}')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('1','2','3') or torch.cuda.device_count()!=1:
        raise RuntimeError('Isolate one physical GPU with CUDA_VISIBLE_DEVICES=1/2/3')
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    output.mkdir(parents=True,exist_ok=True);seed_all(cfg.seed);model=None
    resolved=dict(cfg.protocol(),git_commit=version(),code_sha256=code_hashes(),
                  physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),
                  assets_manifest_sha256=file_hash(ROOT/'exp9/results/assets_manifest.json'),
                  input_assets_sha256=verify_assets())
    if cfg.stage=='final':
        resolved['final_freeze_sha256']=file_hash(RESULTS/'final_freeze.json')
        resolved['official_assets_sha256']=official_assets()
    resolved['config_sha256']=json_hash(resolved)
    save_json(output/'config.json',resolved)
    seed_all(cfg.seed)
    try:
        if cfg.task=='cv':
            from exp16e.cv_model import pretrained_vit, initialization_digest
            base,metadata=pretrained_vit()
            init_hash,head_hash=initialization_digest(base),initialization_digest(base.head)
            training,validation,order=cv_datasets(cfg,metadata)
        else:
            base=make_model();init_hash,head_hash=digest(base),digest(base.classifier)
            training,validation,order=nlp_datasets(cfg)
        model=(ScaledGhostModule if cfg.scaled else StandardGhostModule)(base.to(device),cfg.clip) if cfg.private else base.to(device)
        if not all(p.requires_grad and p.dtype==torch.float32 for p in model.parameters()): raise ValueError('Full FP32 finetuning required')
        optimizer=make_optimizer(model,cfg)
        noise=None;privacy=dict(private=False);provenance=dict(family='none',clipped=False)
        rng_start=None
        if cfg.private:
            matrix=load(cfg.workload,cfg.mechanism,cfg.total_steps)
            d,S,W=matrix['d'],matrix['C'],matrix['A']
            provenance=dict(source=str(matrix['path'].relative_to(ROOT)),matrix_sha256=file_hash(matrix['path']))
            privacy=calibrate(S,cfg.protocol())
            full_epsilon=epsilon_from_mu(cfg.clip*privacy['sensitivity']/privacy['innovation_std_sum'],1e-5)
            if abs(full_epsilon-8.)>1e-7: raise ValueError('Incorrect privacy calibration')
            privacy.update(private=True,verified_full_epsilon=full_epsilon,delta=1e-5,target_epsilon=8.,clip_space='scaled' if cfg.scaled else 'original')
            np.savez(output/'matrices.npz',coefficients=d,strategy=S,workload=W,inverse=matrix['D'])
            noise=OnlineNoise(model.parameters(),matrix,cfg.mechanism,privacy['innovation_std_sum'],cfg.total_steps,cfg.seed+1)
            rng_start=array_hash(noise.generator.get_state().numpy())
        np.save(output/'train_order.npy',order)
        logical=LogicalBatch(model,optimizer,1000//cfg.physical_batch,1000,noise,cfg.scaled,cfg.eps_scale)
        prefix_steps=2 if cfg.stage=='smoke' else None
        if prefix_steps: training=Subset(training,range(prefix_steps*1000))
        if cfg.stage=='smoke': validation=Subset(validation,range(64))
        kwargs=dict(batch_size=cfg.physical_batch,shuffle=False,pin_memory=True,
                    num_workers=2 if cfg.task=='cv' else 0,generator=torch.Generator().manual_seed(cfg.seed))
        if cfg.task=='cv': kwargs['multiprocessing_context']='spawn'
        loader=DataLoader(training,**kwargs)
        eval_loader=DataLoader(validation,batch_size=64 if cfg.stage=='smoke' else cfg.physical_batch if cfg.task=='cv' else 100,shuffle=False)
        torch.cuda.reset_peak_memory_stats(device)
        rows=[];epochs=[];rng_trace=[];train_seconds=0.;micro_count=0
        for epoch in range(1 if prefix_steps else 5):
            model.train();loss_sum=count=clip_count=0
            loss_part=clips_part=norm_part=0.;norm_max=0.;augmentation=hashlib.sha256()
            for batch in loader:
                if logical.micro_steps==0:
                    torch.cuda.synchronize();step_start=time.monotonic()
                if cfg.task=='nlp':
                    dropout_seed=cfg.seed+100000+logical.optimizer_steps
                    torch.manual_seed(dropout_seed);torch.cuda.manual_seed(dropout_seed)
                    batch=tuple(t.to(device) for t in batch);inputs,targets=batch[:3],batch[3]
                else:
                    inputs,targets=batch
                    if logical.micro_steps==0: augmentation.update(inputs[:4].contiguous().numpy().tobytes())
                    inputs,targets=inputs.to(device),targets.to(device)
                if len(targets)!=cfg.physical_batch: raise ValueError('Incomplete physical batch')
                logical.begin_microbatch()
                loss,clips=clipped_microbatch(model,inputs,targets,cfg.clip) if cfg.private else unclipped_microbatch(model,inputs,targets)
                loss_part+=loss;clips_part+=clips
                if cfg.private:
                    norm_part+=float(model.last_norms.sum());norm_max=max(norm_max,float(model.last_norms.max()))
                count+=len(targets);loss_sum+=loss;clip_count+=clips;micro_count+=1
                if logical.finish_microbatch():
                    torch.cuda.synchronize();elapsed=time.monotonic()-step_start;train_seconds+=elapsed
                    step=logical.optimizer_steps
                    mu=cfg.clip*fixed_epoch_sensitivity(S,5,cfg.spacing,steps=step)/privacy['innovation_std_sum'] if cfg.private else None
                    row=dict(step=step,epoch=epoch+1,train_loss=loss_part/1000,clip_fraction=clips_part/1000 if cfg.private else None,
                        gradient_norm_mean=norm_part/1000 if cfg.private else None,gradient_norm_max=norm_max if cfg.private else None,
                        step_seconds=elapsed,innovation_std_sum=privacy.get('innovation_std_sum'),
                        noise_marginal_std_sum=noise.marginal_std(step-1) if noise else None,epsilon=epsilon_from_mu(mu,1e-5) if cfg.private else None,gradient_norm=logical.last_gradient_norm)
                    if cfg.private and row['epsilon']>8.+1e-7: raise ValueError('Privacy budget exceeded')
                    if noise and noise.step_count!=step: raise ValueError('Noise draw/step mismatch')
                    rows.append(row)
                    with (output/'diagnostics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                    loss_part=clips_part=norm_part=0.;norm_max=0.
            expected=prefix_steps or (epoch+1)*cfg.spacing
            if logical.optimizer_steps!=expected or logical.micro_steps: raise ValueError('Incorrect step count')
            evaluation=evaluate(model,eval_loader,device,cfg.task)
            record=dict(epoch=epoch+1,step=logical.optimizer_steps,train_loss=loss_sum/count,
                        clip_fraction=clip_count/count if cfg.private else None,epsilon=rows[-1]['epsilon'],**evaluation)
            epochs.append(record);write_csv(output/'metrics.csv',epochs)
            rng_trace.append(augmentation.hexdigest() if cfg.task=='cv' else [cfg.seed+100000+epoch*cfg.spacing,cfg.seed+100000+expected-1])
            print(json.dumps(record),flush=True)
        if cfg.private and not prefix_steps and abs(epochs[-1]['epsilon']-8.)>1e-7: raise ValueError('Final epsilon mismatch')
        torch.save(dict(model=(model._module if cfg.private else model).state_dict(),optimizer=optimizer.state_dict(),logical_steps=logical.optimizer_steps),output/'checkpoint.pt')
        hashes={name:file_hash(output/name) for name in ('config.json','checkpoint.pt','train_order.npy','metrics.csv','diagnostics.jsonl')+ (('matrices.npz',) if cfg.private else ())}
        result=dict(**cfg.values(),trial_id=cfg.id,status='completed',finite=True,
            optimizer_steps=logical.optimizer_steps,noise_draws=noise.step_count if noise else 0,physical_batches=micro_count,
            completed_epochs=0 if prefix_steps else 5,planned_total_steps=cfg.total_steps,
            accuracy=epochs[-1]['accuracy'],validation_loss=epochs[-1]['validation_loss'],
            evaluation_scope='64 internal validation examples, smoke only' if cfg.stage=='smoke' else 'official evaluation' if cfg.stage=='final' else 'internal validation',
            official_evaluation_used=cfg.stage=='final',epochs=epochs,calibration=privacy,mechanism=provenance,
            config_sha256=resolved['config_sha256'],code_sha256=resolved['code_sha256'],git_commit=resolved['git_commit'],
            file_sha256=hashes,initialization_sha256=init_hash,classifier_initialization_sha256=head_hash,
            train_order_sha256=array_hash(order),rng_trace=rng_trace,noise_rng_start=rng_start,
            noise_rng_end=array_hash(noise.generator.get_state().numpy()) if noise else None,actual_epsilon=epochs[-1]['epsilon'],
            seconds=time.monotonic()-started,train_seconds=train_seconds,
            cpu_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            device=torch.cuda.get_device_name(device),physical_gpu=resolved['physical_gpu'])
        save_json(output/'summary.json',result);return result
    except Exception as error:
        status='oom' if isinstance(error,torch.cuda.OutOfMemoryError) else 'non_finite' if isinstance(error,FloatingPointError) else 'failed'
        save_json(output/'summary.json',dict(**cfg.values(),trial_id=cfg.id,status=status,error=f'{type(error).__name__}: {error}',
            traceback=traceback.format_exc(),seconds=time.monotonic()-started))
        raise
    finally:
        if model is not None and cfg.private:model.remove_hooks()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-json',required=True)
    cfg=Trial(**json.loads(parser.parse_args().trial_json));run(cfg)


if __name__=='__main__':main()
