"""One noise innovation and Adam update per 1000 examples, local inputs only."""
import argparse
import csv
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from exp8a import RESULTS
from exp8a.config import Config, MODEL_PATH, save_json, file_hash
from exp8a.data import datasets
from exp8a.model import seed_all, make_model, digest
from exp8a.clipping import StandardGhostModule, ScaledGhostModule, LogicalBatch, clipped_microbatch
from exp8a.bandinvmf import bias_matrices, BandInvMFNoise, materialize
from exp8a.privacy import calibrate, fixed_epoch_sensitivity, epsilon_from_mu


def mechanism(cfg):
    d,S,W,meta = bias_matrices()
    privacy = calibrate(S,cfg.protocol())
    return d,S,W,meta,privacy


def setup(cfg, geometry, gpu):
    assert geometry in ('standard','scale')
    seed_all(cfg.seed)
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(device)
    assert '3080 Ti' in torch.cuda.get_device_name(device)
    base = make_model()
    hashes = dict(initialization_sha256=digest(base),classifier_initialization_sha256=digest(base.classifier),
                  checkpoint_sha256=file_hash(MODEL_PATH/'pytorch_model.bin'),
                  split_sha256=file_hash(RESULTS/'split.npz'))
    base = base.to(device)
    model = (ScaledGhostModule if geometry=='scale' else StandardGhostModule)(base,cfg.C)
    assert all(p.requires_grad and p.dtype==torch.float32 for p in model.parameters())
    assert {id(p) for p in model.parameters()}=={id(p) for p in model.trainable_parameters}
    opt = torch.optim.Adam(model.parameters(),lr=cfg.lr,betas=(cfg.beta1,cfg.beta2),eps=cfg.adam_eps,weight_decay=0.)
    d,S,W,meta,privacy = mechanism(cfg)
    noise = BandInvMFNoise(model.parameters(),d,privacy['innovation_std_sum'],310,cfg.seed+1)
    logical = LogicalBatch(model,opt,1000//cfg.physical_batch_size,1000,noise,
                           scaled=geometry=='scale',eps_scale=cfg.eps_scale)
    return model,opt,logical,device,hashes


def logical_step(model, logical, dataset, start, cfg, device):
    model.train()
    losses, clips = 0.,0
    for offset in range(start,start+1000,cfg.physical_batch_size):
        batch = tuple(t[offset:offset+cfg.physical_batch_size].to(device) for t in dataset.tensors)
        logical.begin_microbatch()
        loss, clipped = clipped_microbatch(model,batch[:3],batch[3],cfg.C)
        losses += loss
        clips += clipped
        logical.finish_microbatch()
    assert logical.micro_steps == 0 and logical.noise.step_count == logical.optimizer_steps
    tensors = list(model.parameters()) + [logical.optimizer.state[p][k] for p in model.parameters() for k in ('exp_avg','exp_avg_sq')]
    if not all(torch.isfinite(t).all().item() for t in tensors):
        raise FloatingPointError('non-finite model/Adam state')
    return losses/1000, clips/1000


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=100):
    model.eval()
    loss,correct = 0.,0
    for batch in DataLoader(dataset,batch_size=batch_size,shuffle=False):
        batch = tuple(t.to(device) for t in batch)
        logits = model(batch[:3])
        loss += float(torch.nn.functional.cross_entropy(logits,batch[3],reduction='sum'))
        correct += int((logits.argmax(1)==batch[3]).sum())
    return dict(loss=loss/len(dataset),accuracy=correct/len(dataset),examples=len(dataset))


def run(cfg, gpu=0, geometry='scale', steps=310, output=None):
    assert 1 <= steps <= 310
    output = output or RESULTS/f'runs/length{cfg.max_length}_seed{cfg.seed}'
    model,opt,logical,device,hashes = setup(cfg,geometry,gpu)
    train,valid = datasets(cfg.max_length)
    d,S,W,meta,privacy = mechanism(cfg)
    output.mkdir(parents=True,exist_ok=True)
    save_json(output/'config.json',dict(cfg.protocol(),geometry=geometry,**hashes,privacy_calibration=privacy,
                                       coefficients=d.tolist(),matrix_optimization=meta,requested_steps=steps))
    np.savez(output/'matrices.npz',coefficients=d,strategy=S,workload=W)
    records = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(steps):
        loss,clip = logical_step(model,logical,train,(step%62)*1000,cfg,device)
        if (step+1)%62==0 or step+1==steps:
            evaluation = evaluate(model,valid,device)
            mu = cfg.C*fixed_epoch_sensitivity(S,5,62,steps=step+1)/privacy['innovation_std_sum']
            row = dict(step=step+1,epoch=(step+1)/62,train_loss=loss,clip_fraction=clip,**evaluation,
                       epsilon=epsilon_from_mu(mu,1e-5),mu=mu)
            records.append(row)
            print(row,flush=True)
    torch.cuda.synchronize(device)
    result = dict(status='completed',geometry=geometry,**hashes,**records[-1],seed=cfg.seed,max_length=cfg.max_length,
                  physical_batch_size=cfg.physical_batch_size,lr=cfg.lr,C=cfg.C,eps_scale=cfg.eps_scale,
                  optimizer_steps=logical.optimizer_steps,
                  noise_draws=logical.noise.step_count,seconds=time.monotonic()-started,
                  full_run=steps==310,official_validation_used=False,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    save_json(output/'summary.json',result)
    with open(output/'metrics.csv','w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    torch.save(model._module.state_dict(),output/'model.pt')
    model.remove_hooks()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--physical-batch-size',type=int,default=20,choices=(4,8,20,40,50,100,125,200,250,500,1000))
    p.add_argument('--max-length',type=int,default=64,choices=(32,64,128))
    p.add_argument('--geometry',choices=('standard','scale'),default='scale')
    a = p.parse_args()
    run(Config(physical_batch_size=a.physical_batch_size,max_length=a.max_length),a.gpu,a.geometry)

if __name__=='__main__': main()
