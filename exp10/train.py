"""One isolated GPU worker. CV and NLP use identical logical DP/Adam semantics."""
import argparse
import csv
import json
import os
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from exp10 import RESULTS
from exp10.config import Trial, METHODS, file_hash, array_hash, save_json
from exp10.clipping import StandardGhostModule, LogicalBatch, clipped_microbatch
from exp10.noise import BandInvMFNoise, materialize
from exp10.matrices import get_matrix
from exp10.optim import BlockAdam, PreconditionerDiagnostics
from exp10.privacy import calibrate, fixed_epoch_sensitivity, epsilon_from_mu
from exp10.nlp_model import seed_all, make_model, digest
from exp10.data import cv_datasets, nlp_datasets

@torch.no_grad()
def evaluate(model, loader, device, task):
    model.eval(); loss=correct=count=0
    for batch in loader:
        if task == 'cv':
            inputs, targets = (t.to(device) for t in batch)
        else:
            batch = tuple(t.to(device) for t in batch)
            inputs, targets = batch[:3], batch[3]
        logits = model(inputs)
        loss += float(torch.nn.functional.cross_entropy(logits,targets,reduction='sum'))
        correct += int((logits.argmax(1)==targets).sum()); count += len(targets)
    if not np.isfinite(loss): raise FloatingPointError('non-finite evaluation loss')
    return dict(accuracy=correct/count, validation_loss=loss/count, validation_examples=count)

def run(cfg):
    started = time.monotonic(); output = cfg.output
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('1','2','3')
    assert torch.cuda.device_count() == 1
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    seed_all(cfg.seed)
    from exp10.audit import code_hashes
    resolved = dict(cfg.protocol(), physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']), local_device='cuda:0',
                    code_sha256=code_hashes(), assets_manifest_sha256=file_hash(RESULTS/'assets_manifest.json'))
    if cfg.stage == 'final':
        from exp10.frozen import load_frozen
        winner = load_frozen()[cfg.task][cfg.method]
        assert all(getattr(cfg,k) == winner[k] for k in ('lr','C','module'))
        resolved['frozen_manifest_sha256'] = file_hash(RESULTS/'frozen_manifest.json')
        resolved['official_assets_sha256'] = file_hash(RESULTS/'official_assets.json')
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'config.json',resolved)
    model = None
    try:
        if cfg.task == 'cv':
            from exp10.cv_model import pretrained_vit, initialization_digest
            base, metadata = pretrained_vit()
            init_hash, head_hash = initialization_digest(base), initialization_digest(base.head)
            training, validation, order = cv_datasets(cfg,metadata)
        else:
            base = make_model()
            init_hash, head_hash = digest(base), digest(base.classifier)
            training, validation, order = nlp_datasets(cfg)
        model = StandardGhostModule(base.to(device),cfg.C)
        assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())
        assert {id(p) for p in model.parameters()} == {id(p) for p in model.trainable_parameters}
        optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr,betas=(.9,.999),eps=1e-8,weight_decay=0.)
        if cfg.method == METHODS[1]: optimizer = BlockAdam(model.parameters(),cfg.lr,int(cfg.module))
        diagnostics = PreconditionerDiagnostics()
        d,S,W,matrix_meta = get_matrix(cfg.total_steps,cfg.module if cfg.method == METHODS[0] else None)
        privacy = calibrate(S,cfg.protocol())
        full_epsilon = epsilon_from_mu(cfg.C*privacy['sensitivity']/privacy['innovation_std_sum'],1e-5)
        assert abs(full_epsilon-cfg.epsilon) < 1e-7
        privacy.update(verified_full_epsilon=full_epsilon, delta=1e-5, target_epsilon=cfg.epsilon)
        np.savez(output/'matrices.npz',coefficients=d,strategy=S,workload=W)
        np.save(output/'train_order.npy',order)
        noise = BandInvMFNoise(model.parameters(),d,privacy['innovation_std_sum'],cfg.total_steps,cfg.seed+1)
        accumulation = 1000//cfg.physical_batch
        logical = LogicalBatch(model,optimizer,accumulation,1000,noise)
        # Three smoke steps cross b=2 boundary.
        if cfg.stage == 'smoke': training = Subset(training,range(3000))
        kwargs = dict(batch_size=cfg.physical_batch,shuffle=False,pin_memory=True,
                      num_workers=2 if cfg.task=='cv' else 0,
                      generator=torch.Generator().manual_seed(cfg.seed))
        if cfg.task=='cv': kwargs['multiprocessing_context']='spawn'
        loader = DataLoader(training,**kwargs)
        eval_loader = DataLoader(validation,batch_size=cfg.physical_batch if cfg.task=='cv' else 100,shuffle=False)
        torch.cuda.reset_peak_memory_stats(device)
        records=[]; mechanisms=[]; rng_trace=[]
        train_seconds=0.; micro_count=0
        epochs = 1 if cfg.stage=='smoke' else 5
        pure_variance = (.1*W @ materialize(d,cfg.total_steps))**2
        pure_variance = pure_variance.sum(axis=1)*(privacy['innovation_std_sum']/1000)**2
        fields = ('step','epoch','step_seconds','innovation_std_sum','noise_marginal_std_sum',
                  'pure_mf_cumulative_momentum_noise_variance','preconditioner_mean',
                  'preconditioner_change_relative','denominator_lag_relative','block_age')
        with (output/'mechanism_metrics.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for epoch in range(epochs):
                model.train(); count=0
                step_start=None
                augmentation=__import__('hashlib').sha256()
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
                    assert len(targets)==cfg.physical_batch
                    logical.begin_microbatch()
                    clipped_microbatch(model,inputs,targets,cfg.C)
                    micro_count+=1;count+=len(targets)
                    if logical.finish_microbatch():
                        tensors=list(model.parameters())+[state[k] for state in optimizer.state.values() for k in ('exp_avg','exp_avg_sq')]
                        if not all(torch.isfinite(t).all().item() for t in tensors): raise FloatingPointError('non-finite model/Adam state')
                        torch.cuda.synchronize();elapsed=time.monotonic()-step_start;train_seconds+=elapsed
                        row=dict(step=logical.optimizer_steps,epoch=epoch+1,step_seconds=elapsed,
                                 innovation_std_sum=privacy['innovation_std_sum'],
                                 noise_marginal_std_sum=noise.marginal_std(logical.optimizer_steps-1),
                                 pure_mf_cumulative_momentum_noise_variance=float(pure_variance[logical.optimizer_steps-1]),
                                 **diagnostics.record(optimizer))
                        writer.writerow(row);f.flush();mechanisms.append(row)
                        assert noise.step_count==logical.optimizer_steps
                        if logical.optimizer_steps%10==0: print(json.dumps(row),flush=True)
                expected=(epoch+1)*(3 if cfg.stage=='smoke' else cfg.spacing)
                assert logical.optimizer_steps==expected and logical.micro_steps==0
                # Smoke touches no held-out evaluation data.
                evaluation = dict(accuracy=0.,validation_loss=0.,validation_examples=0) if cfg.stage=='smoke' else evaluate(model,eval_loader,device,cfg.task)
                mu=cfg.C*fixed_epoch_sensitivity(S,5,cfg.spacing,steps=logical.optimizer_steps)/privacy['innovation_std_sum']
                record=dict(epoch=epoch+1,step=logical.optimizer_steps,epsilon=epsilon_from_mu(mu,1e-5),**evaluation)
                assert record['epsilon'] <= cfg.epsilon+1e-7
                records.append(record);print(json.dumps(record),flush=True)
                rng_trace.append(augmentation.hexdigest() if cfg.task=='cv' else f'{cfg.seed+100000+epoch*cfg.spacing}:{cfg.seed+100000+expected-1}')
        steps = 3 if cfg.stage=='smoke' else cfg.total_steps
        assert logical.optimizer_steps==noise.step_count==steps and micro_count==steps*accumulation
        if cfg.stage!='smoke': assert abs(records[-1]['epsilon']-cfg.epsilon)<1e-7
        torch.save(dict(model=model._module.state_dict(),optimizer=optimizer.state_dict(),logical_steps=steps),output/'checkpoint.pt')
        from exp10.report import write_csv
        write_csv(output/'metrics.csv',records)
        hashes={name:file_hash(output/name) for name in ('config.json','checkpoint.pt','matrices.npz','train_order.npy','metrics.csv','mechanism_metrics.csv')}
        result=dict(cfg.values(),trial_id=cfg.id,status='completed',accuracy=records[-1]['accuracy'],
                    validation_loss=records[-1]['validation_loss'],completed_epochs=0 if cfg.stage=='smoke' else 5,optimizer_steps=steps,noise_draws=noise.step_count,
                    innovation_tensor_draws=steps*len(list(model.parameters())),physical_batches=micro_count,
                    physical_batch_size=cfg.physical_batch,planned_total_steps=cfg.total_steps,finite=True,
                    initialization_sha256=init_hash,classifier_initialization_sha256=head_hash,
                    train_order_sha256=hashes['train_order.npy'],rng_trace=rng_trace,
                    strategy_sha256=array_hash(S),workload_sha256=array_hash(W),coefficients=d.tolist(),
                    file_sha256=hashes,code_sha256=resolved['code_sha256'],assets_manifest_sha256=resolved['assets_manifest_sha256'],
                    calibration=privacy,actual_epsilon=records[-1]['epsilon'],epochs=records,
                    official_evaluation_used=cfg.stage == 'final',
                    seconds=time.monotonic()-started,train_seconds=train_seconds,logical_step_seconds=train_seconds/steps,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    physical_gpu=resolved['physical_gpu'],device=torch.cuda.get_device_name(device),
                    result_dir=str(output.relative_to(RESULTS)),first_step_diagnostics=mechanisms[0],last_step_diagnostics=mechanisms[-1])
        result['matrix_optimization']=matrix_meta
        result['noise_covariance']=dict(average_diagonal=matrix_meta['covariance_diagonal_mean']*cfg.C**2,
            average_lag1=matrix_meta['covariance_lag1_mean']*cfg.C**2,
            J1=matrix_meta['J1'],J2=matrix_meta['J2']*cfg.C**4)
        save_json(output/'summary.json',result)
        return result
    except Exception as error:
        status = 'oom' if isinstance(error,torch.cuda.OutOfMemoryError) else 'non_finite' if isinstance(error,FloatingPointError) else 'failed'
        save_json(output/'summary.json',dict(cfg.values(),trial_id=cfg.id,status=status,error=f'{type(error).__name__}: {error}',
                  traceback=traceback.format_exc(),seconds=time.monotonic()-started,physical_gpu=resolved['physical_gpu'],
                  physical_batch_size=cfg.physical_batch,peak_allocated_bytes=torch.cuda.max_memory_allocated(device)))
        raise
    finally:
        if model is not None: model.remove_hooks()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-json',required=True,help='Canonical Trial JSON supplied by FIFO scheduler')
    cfg=Trial(**json.loads(parser.parse_args().trial_json));run(cfg)

if __name__=='__main__': main()
