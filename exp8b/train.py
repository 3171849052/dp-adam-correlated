"""Full BERT fine-tuning with one physical=logical 1000-example two-backward step."""
import argparse
import csv
import json
import time
import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from exp8b import RESULTS
from exp8b.config import *
from exp8b.data import datasets,official_validation
from exp8b.model import seed_all,make_model,digest
from exp8b.clipping import StandardGhostModule,ScaledGhostModule,LogicalBatch,clipped_microbatch
from exp8b.bandinvmf import build_matrices,bias_matrices,materialize
from exp8b.noise import BandInvMFNoise
from exp8b.privacy import calibrate,fixed_epoch_sensitivity,epsilon_from_mu


def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a,dtype=np.float64).tobytes()).hexdigest()


def mechanism(cfg):
    d,S,W=build_matrices(METHODS[cfg.method]['noise'],310,4,.9)
    meta=bias_matrices()[3] if METHODS[cfg.method]['noise']=='momentum_bias_bandinvmf' else dict(construction='Exp7b/Exp2 banded inverse square-root coefficients, rebuilt at T=310')
    return d,S,W,meta,calibrate(S,cfg.protocol())


def setup(cfg,gpu):
    seed_all(cfg.seed)
    device=torch.device(f'cuda:{gpu}'); torch.cuda.set_device(device)
    base=make_model(cfg.dropout)
    hashes=dict(initialization_sha256=digest(base),classifier_initialization_sha256=digest(base.classifier),
                pretrained_sha256=file_hash(MODEL_PATH/'pytorch_model.bin'),split_sha256=file_hash(RESULTS/'split.npz'),
                tokens_sha256=file_hash(RESULTS/'tokens_128.pt'),tokenizer_sha256=file_hash(MODEL_PATH/'vocab.txt'),
                model_config_sha256=file_hash(MODEL_PATH/'config.json'))
    wrapper=ScaledGhostModule if cfg.geometry=='scale' else StandardGhostModule
    model=wrapper(base.to(device),cfg.C)
    assert all(p.requires_grad and p.dtype==torch.float32 for p in model.parameters())
    assert {id(p) for p in model.parameters()}=={id(p) for p in model.trainable_parameters}
    opt=torch.optim.Adam(model.parameters(),lr=cfg.lr,betas=(.9,.999),eps=1e-8,weight_decay=0.)
    d,S,W,meta,privacy=mechanism(cfg)
    noise=BandInvMFNoise(model.parameters(),d,privacy['innovation_std_sum'],310,cfg.seed+1)
    logical=LogicalBatch(model,opt,1,1000,noise,scaled=cfg.geometry=='scale',eps_scale=cfg.eps_scale or .1)
    return model,opt,logical,device,hashes


def logical_step(model,logical,train,start,cfg,device):
    model.train()
    # Stateless paired dropout schedule, independent of evaluation and DP noise draws.
    torch.manual_seed(cfg.seed+100000+logical.optimizer_steps)
    torch.cuda.manual_seed(cfg.seed+100000+logical.optimizer_steps)
    batch=tuple(t[start:start+1000].to(device) for t in train.tensors)
    assert len(batch[0])==1000 and batch[0].shape[1]==128
    logical.begin_microbatch()
    loss,clips=clipped_microbatch(model,batch[:3],batch[3],cfg.C)
    norms=model.last_norms
    diagnostics=dict(train_loss=loss/1000,clip_fraction=clips/1000,
                     gradient_norm_mean=float(norms.mean()),gradient_norm_p50=float(norms.median()),
                     gradient_norm_p90=float(torch.quantile(norms,.9)),gradient_norm_max=float(norms.max()))
    assert logical.finish_microbatch()
    tensors=list(model.parameters())+[opt_state[k] for opt_state in logical.optimizer.state.values() for k in ('exp_avg','exp_avg_sq')]
    if not all(torch.isfinite(t).all().item() for t in tensors): raise FloatingPointError('non-finite model/Adam state')
    assert logical.noise.step_count==logical.optimizer_steps
    return diagnostics


@torch.no_grad()
def evaluate(model,dataset,device,batch_size=100):
    model.eval(); loss=correct=0
    for batch in DataLoader(dataset,batch_size=batch_size,shuffle=False):
        batch=tuple(t.to(device) for t in batch)
        logits=model(batch[:3])
        loss+=float(torch.nn.functional.cross_entropy(logits,batch[3],reduction='sum'))
        correct+=int((logits.argmax(1)==batch[3]).sum())
    return dict(validation_loss=loss/len(dataset),accuracy=correct/len(dataset),validation_examples=len(dataset))


def write_csv(path,records):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def run(cfg,gpu=0,category='search',output=None):
    assert category in ('smoke','search','final')
    assert cfg.seed==(SEED) if category!='final' else cfg.seed in FINAL_SEEDS
    values=trial(cfg.method,cfg.lr,cfg.C,cfg.eps_scale,cfg.seed)
    output=Path(output) if output else trial_dir(values,category)
    assert output.resolve().is_relative_to(BASE)
    output.mkdir(parents=True,exist_ok=True)
    assert not (output/'summary.json').exists(), 'Trial already exists; use launcher to audit/reuse'
    model=None
    started=time.monotonic()
    frozen_hash=None
    if category=='final':
        from exp8b.frozen import load_frozen
        frozen=load_frozen()
        assert all(values[k]==frozen[cfg.method][k] for k in ('lr','C','eps_scale'))
        frozen_hash=file_hash(RESULTS/'frozen_configs.json')
    save_json(output/'config.json',dict(cfg.protocol(),category=category,frozen_configs_sha256=frozen_hash))
    try:
        model,opt,logical,device,hashes=setup(cfg,gpu)
        train,valid=datasets()
        if category=='final': valid=official_validation(category)
        d,S,W,meta,privacy=mechanism(cfg)
        np.savez(output/'matrices.npz',coefficients=d,strategy=S,workload=W)
        with np.load(RESULTS/'split.npz') as split: np.save(output/'train_order.npy',split['train'])
        records=[];diagnostics=[];dropout_trace=[]
        steps=1 if category=='smoke' else 310
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
        train_seconds=0.
        for step in range(steps):
            torch.cuda.synchronize(device); t=time.monotonic()
            diag=logical_step(model,logical,train,(step%62)*1000,cfg,device)
            torch.cuda.synchronize(device); elapsed=time.monotonic()-t; train_seconds+=elapsed
            dropout_trace.append(cfg.seed+100000+step)
            diag.update(step=step+1,step_seconds=elapsed,noise_marginal_std_sum=logical.noise.marginal_std(step),
                        innovation_std_sum=privacy['innovation_std_sum'])
            diagnostics.append(diag)
            if (step+1)%62==0 or step+1==steps:
                # Smoke never evaluates either validation set.
                evaluation=dict(validation_loss=0.,accuracy=0.,validation_examples=0) if category=='smoke' else evaluate(model,valid,device)
                mu=cfg.C*fixed_epoch_sensitivity(S,5,62,steps=step+1)/privacy['innovation_std_sum']
                row=dict(step=step+1,epoch=(step+1)/62,train_loss=float(np.mean([r['train_loss'] for r in diagnostics[-62:]])),
                         clip_fraction=float(np.mean([r['clip_fraction'] for r in diagnostics[-62:]])),
                         **evaluation,epsilon=epsilon_from_mu(mu,1e-5))
                records.append(row); print(json.dumps(row),flush=True)
        torch.save(dict(model=model._module.state_dict(),optimizer=opt.state_dict(),logical_steps=steps),output/'checkpoint.pt')
        write_csv(output/'metrics.csv',records);write_csv(output/'mechanism_metrics.csv',diagnostics)
        result=dict(values,status='completed',category=category,geometry=cfg.geometry,**hashes,
                    accuracy=records[-1]['accuracy'],epochs=records,completed_epochs=0 if category=='smoke' else 5,
                    optimizer_steps=steps,noise_draws=logical.noise.step_count,physical_batches=steps,
                    physical_batch_size=1000,max_length=128,planned_total_steps=310,finite=True,
                    official_validation_used=category=='final',validation_examples=records[-1]['validation_examples'],
                    seconds=time.monotonic()-started,logical_step_seconds=train_seconds/steps,samples_per_second=1000*steps/train_seconds,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    device=torch.cuda.get_device_name(device),calibration=privacy,matrix_optimization=meta,
                    coefficients=d.tolist(),strategy_sha256=array_hash(S),workload_sha256=array_hash(W),
                    matrix_sha256=file_hash(output/'matrices.npz'),checkpoint_sha256=file_hash(output/'checkpoint.pt'),
                    train_order_sha256=file_hash(output/'train_order.npy'),config_sha256=file_hash(output/'config.json'),
                    metrics_sha256=file_hash(output/'metrics.csv'),mechanism_metrics_sha256=file_hash(output/'mechanism_metrics.csv'),
                    dropout_trace_sha256=hashlib.sha256(json.dumps(dropout_trace).encode()).hexdigest(),
                    frozen_configs_sha256=frozen_hash,result_dir=str(output.relative_to(ROOT)),
                    first_step_diagnostics=diagnostics[0])
        save_json(output/'summary.json',result)
        return result
    except Exception as error:
        oom=isinstance(error,torch.cuda.OutOfMemoryError)
        save_json(output/'summary.json',dict(values,status='oom' if oom else 'non_finite' if isinstance(error,FloatingPointError) else 'failed',
                  category=category,error=str(error),oom=oom,physical_batch_size=1000,max_length=128,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(gpu) if torch.cuda.is_initialized() else 0,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(gpu) if torch.cuda.is_initialized() else 0,
                  seconds=time.monotonic()-started,result_dir=str(output.relative_to(ROOT))))
        raise
    finally:
        if model is not None: model.remove_hooks()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=METHODS,required=True);p.add_argument('--lr',type=float,required=True)
    p.add_argument('--C',type=float,required=True);p.add_argument('--eps-scale',type=float)
    p.add_argument('--seed',type=int,default=SEED);p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--category',choices=('smoke','search','final'),default='search')
    a=p.parse_args();run(Config(method=a.method,seed=a.seed,lr=a.lr,C=a.C,eps_scale=a.eps_scale),a.gpu,a.category)

if __name__=='__main__': main()
