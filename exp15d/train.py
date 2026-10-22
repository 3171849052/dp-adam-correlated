"""Offline worker: smoke/pilot prefixes and complete five-epoch frozen trials."""
import argparse
import hashlib
import json
import os
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from exp15d.config import Trial, file_hash, array_hash, save_json
from exp15d.clipping import StandardGhostModule, ScaledGhostModule, LogicalBatch, clipped_microbatch
from exp15d.optimizer import Optimizer
from exp15d.noise import BandInvMFNoise
from exp15d.privacy import calibrate, fixed_epoch_sensitivity, epsilon_from_mu
from exp15d.nlp_model import seed_all, make_model, digest
from exp15d.data import cv_datasets, nlp_datasets
from exp15d.support import matrices, code_hashes, version, write_csv


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
        from exp15d.search import authorize
        authorize(cfg)
    started=time.monotonic();output=cfg.output
    if (output/'summary.json').exists(): raise FileExistsError(f'Run already exists: {output}')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0','1','2','3') or torch.cuda.device_count()!=1:
        raise RuntimeError('Isolate one physical GPU with CUDA_VISIBLE_DEVICES=0/1/2/3')
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    output.mkdir(parents=True,exist_ok=True);seed_all(cfg.seed);model=None
    resolved=dict(cfg.protocol(),git_commit=version(),code_sha256=code_hashes(),
                  physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),
                  assets_manifest_sha256=file_hash(__import__('exp15d').ROOT/'exp9/results/assets_manifest.json'))
    resolved['config_sha256']=__import__('exp15d.config',fromlist=['json_hash']).json_hash(resolved)
    save_json(output/'config.json',resolved)
    try:
        if cfg.task=='cv':
            from exp15d.cv_model import pretrained_vit, initialization_digest
            base,metadata=pretrained_vit()
            init_hash,head_hash=initialization_digest(base),initialization_digest(base.head)
            training,validation,order=cv_datasets(cfg,metadata)
        else:
            base=make_model();init_hash,head_hash=digest(base),digest(base.classifier)
            training,validation,order=nlp_datasets(cfg)
        model=(ScaledGhostModule if cfg.scaled else StandardGhostModule)(base.to(device),cfg.clip)
        if not all(p.requires_grad and p.dtype==torch.float32 for p in model.parameters()): raise ValueError('Full FP32 finetuning required')
        if {id(p) for p in model.parameters()}!={id(p) for p in model.trainable_parameters}: raise ValueError('Missing trainable tensors')
        optimizer=Optimizer(model.parameters(),cfg.method,cfg.lr,eps_scale=cfg.eps_scale)
        d,S,W,actual_workload,provenance=matrices(cfg)
        privacy=calibrate(S,cfg.protocol())
        full_epsilon=epsilon_from_mu(cfg.clip*privacy['sensitivity']/privacy['innovation_std_sum'],1e-5)
        if abs(full_epsilon-cfg.epsilon)>1e-7: raise ValueError('Incorrect privacy calibration')
        privacy.update(verified_full_epsilon=full_epsilon,delta=1e-5,target_epsilon=cfg.epsilon,
                       clip_space='preconditioned' if cfg.scaled else 'original')
        np.savez(output/'matrices.npz',coefficients=d,strategy=S,workload=W,actual_workload=actual_workload)
        np.save(output/'train_order.npy',order)
        noise=BandInvMFNoise(model.parameters(),d,privacy['innovation_std_sum'],cfg.total_steps,cfg.seed+1)
        rng_start=array_hash(noise.generator.get_state().numpy())
        accumulation=1000//cfg.physical_batch
        logical=LogicalBatch(model,optimizer,accumulation,1000,noise,scaled=cfg.scaled)
        prefix_steps=2 if cfg.stage=='smoke' else 20 if cfg.stage=='pilot' else None
        if prefix_steps: training=Subset(training,range(prefix_steps*1000))
        if cfg.stage=='smoke': validation=Subset(validation,range(64))
        kwargs=dict(batch_size=cfg.physical_batch,shuffle=False,pin_memory=True,
                    num_workers=2 if cfg.task=='cv' else 0,generator=torch.Generator().manual_seed(cfg.seed))
        if cfg.task=='cv': kwargs['multiprocessing_context']='spawn'
        loader=DataLoader(training,**kwargs)
        eval_loader=DataLoader(validation,batch_size=64 if cfg.stage=='smoke' else 100,shuffle=False)
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
                loss,clips=clipped_microbatch(model,inputs,targets,cfg.clip)
                loss_part+=loss;clips_part+=clips;norm_part+=float(model.last_norms.sum());norm_max=max(norm_max,float(model.last_norms.max()))
                count+=len(targets);loss_sum+=loss;clip_count+=clips;micro_count+=1
                if logical.finish_microbatch():
                    torch.cuda.synchronize();elapsed=time.monotonic()-step_start;train_seconds+=elapsed
                    step=logical.optimizer_steps
                    mu=cfg.clip*fixed_epoch_sensitivity(S,5,cfg.spacing,steps=step)/privacy['innovation_std_sum']
                    row=dict(step=step,epoch=epoch+1,train_loss=loss_part/1000,clip_fraction=clips_part/1000,
                        gradient_norm_mean=norm_part/1000,gradient_norm_max=norm_max,
                        step_seconds=elapsed,innovation_std_sum=privacy['innovation_std_sum'],
                        noise_marginal_std_sum=noise.marginal_std(step-1),epsilon=epsilon_from_mu(mu,1e-5),
                        **{k:v for k,v in optimizer.last_diagnostics.items() if k!='step'})
                    if row['epsilon']>cfg.epsilon+1e-7: raise ValueError('Privacy budget exceeded')
                    if noise.step_count!=step: raise ValueError('Noise draw/step mismatch')
                    rows.append(row)
                    with (output/'diagnostics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                    loss_part=clips_part=norm_part=0.;norm_max=0.
            expected=prefix_steps or (epoch+1)*cfg.spacing
            if logical.optimizer_steps!=expected or logical.micro_steps: raise ValueError('Incorrect step count')
            evaluation=evaluate(model,eval_loader,device,cfg.task)
            record=dict(epoch=epoch+1,step=logical.optimizer_steps,train_loss=loss_sum/count,
                        clip_fraction=clip_count/count,epsilon=rows[-1]['epsilon'],**evaluation)
            epochs.append(record);write_csv(output/'metrics.csv',epochs)
            rng_trace.append(augmentation.hexdigest() if cfg.task=='cv' else [cfg.seed+100000+epoch*cfg.spacing,cfg.seed+100000+expected-1])
            print(json.dumps(record),flush=True)
        if not prefix_steps and abs(epochs[-1]['epsilon']-cfg.epsilon)>1e-7: raise ValueError('Final epsilon mismatch')
        torch.save(dict(model=model._module.state_dict(),optimizer=optimizer.state_dict(),logical_steps=logical.optimizer_steps),output/'checkpoint.pt')
        hashes={name:file_hash(output/name) for name in ('config.json','checkpoint.pt','matrices.npz','train_order.npy','metrics.csv','diagnostics.jsonl')}
        result=dict(**cfg.values(),trial_id=cfg.id,status='completed',finite=True,
            optimizer_steps=logical.optimizer_steps,noise_draws=noise.step_count,physical_batches=micro_count,
            completed_epochs=0 if prefix_steps else 5,planned_total_steps=cfg.total_steps,
            accuracy=epochs[-1]['accuracy'],validation_loss=epochs[-1]['validation_loss'],
            evaluation_scope='64 internal validation examples, smoke only' if cfg.stage=='smoke' else 'official evaluation' if cfg.stage=='final' else 'internal validation',
            official_evaluation_used=cfg.stage=='final',epochs=epochs,calibration=privacy,mechanism=provenance,
            config_sha256=resolved['config_sha256'],code_sha256=resolved['code_sha256'],git_commit=resolved['git_commit'],
            file_sha256=hashes,initialization_sha256=init_hash,classifier_initialization_sha256=head_hash,
            train_order_sha256=array_hash(order),rng_trace=rng_trace,noise_rng_start=rng_start,
            noise_rng_end=array_hash(noise.generator.get_state().numpy()),actual_epsilon=epochs[-1]['epsilon'],
            seconds=time.monotonic()-started,train_seconds=train_seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            device=torch.cuda.get_device_name(device),physical_gpu=resolved['physical_gpu'])
        save_json(output/'summary.json',result);return result
    except Exception as error:
        status='oom' if isinstance(error,torch.cuda.OutOfMemoryError) else 'non_finite' if isinstance(error,FloatingPointError) else 'failed'
        save_json(output/'summary.json',dict(**cfg.values(),trial_id=cfg.id,status=status,error=f'{type(error).__name__}: {error}',
            traceback=traceback.format_exc(),code_sha256=resolved['code_sha256'],
            config_sha256=resolved['config_sha256'],nonfinite_origin='model_loss_norms_or_optimizer_arithmetic' if status=='non_finite' else None,seconds=time.monotonic()-started))
        raise
    finally:
        if model is not None:model.remove_hooks()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-json',required=True)
    cfg=Trial(**json.loads(parser.parse_args().trial_json));run(cfg)


if __name__=='__main__':main()
