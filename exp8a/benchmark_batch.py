"""Isolated GPU benchmarks of the COMPLETE logical DP update."""
import argparse
import csv
from dataclasses import replace
import json
import subprocess
import sys
import time
import torch
from exp8a import ROOT, RESULTS
from exp8a.config import Config, save_json
from exp8a.data import datasets
from exp8a.train import setup, logical_step

FIELDS = ('geometry','max_length','physical_batch_size','status','warmup_steps','measured_steps',
          'seconds','logical_steps_per_second','samples_per_second','peak_allocated_bytes',
          'peak_reserved_bytes','total_vram_bytes','headroom_fraction','optimizer_steps','noise_draws',
          'oom','non_finite','device','error','seed','lr','C','eps_scale','initialization_sha256',
          'classifier_initialization_sha256','checkpoint_sha256','split_sha256')


def benchmark(cfg, geometry, gpu=0, warmup=1, measured=3):
    device = torch.device(f'cuda:{gpu}')
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    row = dict(geometry=geometry,max_length=cfg.max_length,physical_batch_size=cfg.physical_batch_size,
               status='pending',warmup_steps=warmup,measured_steps=measured,oom=False,non_finite=False,
               seed=cfg.seed,lr=cfg.lr,C=cfg.C,eps_scale=cfg.eps_scale,
               device=torch.cuda.get_device_name(device),total_vram_bytes=torch.cuda.get_device_properties(device).total_memory,error='')
    torch.cuda.reset_peak_memory_stats(device)
    started = None
    try:
        model,opt,logical,device,hashes = setup(cfg,geometry,gpu)
        row.update(hashes)
        train,_ = datasets(cfg.max_length)
        for step in range(warmup):
            logical_step(model,logical,train,step*1000,cfg,device)
        torch.cuda.synchronize(device)
        # Include cold Adam/noise allocation in peak, even if warmed measurement reuses it.
        started = time.perf_counter()
        for step in range(warmup,warmup+measured):
            logical_step(model,logical,train,step*1000,cfg,device)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter()-started
        row.update(status='completed',seconds=elapsed,logical_steps_per_second=measured/elapsed,
                   samples_per_second=1000*measured/elapsed,optimizer_steps=logical.optimizer_steps,
                   noise_draws=logical.noise.step_count)
        assert logical.optimizer_steps==logical.noise.step_count==warmup+measured
    except torch.cuda.OutOfMemoryError as e:
        row.update(status='oom',oom=True,error=str(e).splitlines()[0])
    except FloatingPointError as e:
        row.update(status='non_finite',non_finite=True,error=str(e))
    row.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
               peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    row['headroom_fraction'] = 1-max(row['peak_reserved_bytes'],row['peak_allocated_bytes'])/row['total_vram_bytes']
    return row


def isolated(cfg, geometry, gpu=0, warmup=1, measured=3, category='benchmarks'):
    out = RESULTS/category/f'{geometry}_length{cfg.max_length}_batch{cfg.physical_batch_size}.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    command = [sys.executable,'-B','-m','exp8a.benchmark_batch','--worker','--gpu',str(gpu),
               '--physical-batch-size',str(cfg.physical_batch_size),'--max-length',str(cfg.max_length),
               '--geometry',geometry,'--warmup',str(warmup),'--measured',str(measured),'--output',str(out),
               '--lr',str(cfg.lr),'--C',str(cfg.C),'--eps-scale',str(cfg.eps_scale),'--seed',str(cfg.seed)]
    with open(out.with_suffix('.log'),'w') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    row = json.loads(out.read_text())
    print({k:row[k] for k in ('geometry','max_length','physical_batch_size','status','headroom_fraction')},flush=True)
    return row


def write_csv(path,rows,fields=FIELDS):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def select_batch(rows):
    pairs = []
    for batch in sorted({r['physical_batch_size'] for r in rows}):
        group = [r for r in rows if r['physical_batch_size']==batch]
        if len(group)!=2 or {r['geometry'] for r in group}!={'standard','scale'}:
            continue
        if not all(r['status']=='completed' and r['headroom_fraction']>=.2 for r in group):
            continue
        # Harmonic mean of steps/s = throughput of equally weighted complete steps.
        rate = 2/sum(1/r['logical_steps_per_second'] for r in group)
        memory = max(r['peak_reserved_bytes'] for r in group)
        pairs.append(dict(physical_batch_size=batch,combined_steps_per_second=rate,peak_reserved_bytes=memory))
    assert pairs, 'No physical batch passed both mechanisms with 20% VRAM headroom'
    fastest = max(pairs,key=lambda r:r['combined_steps_per_second'])
    near = [r for r in pairs if r['combined_steps_per_second']>=.95*fastest['combined_steps_per_second']]
    chosen = min(near,key=lambda r:(r['peak_reserved_bytes'],r['physical_batch_size']))
    return chosen, pairs


def search(gpu=0,cfg=Config(),length=None,path=None):
    cfg = replace(cfg,max_length=length or cfg.max_length)
    rows = []
    path = path or RESULTS/'physical_batch_benchmark.csv'
    category='benchmarks' if path==RESULTS/'physical_batch_benchmark.csv' else path.stem
    def measure(batch):
        for geometry in ('standard','scale'):
            rows.append(isolated(replace(cfg,physical_batch_size=batch),geometry,gpu,category=category))
            write_csv(path,rows)
    for batch in (8,20,40,50,100):
        measure(batch)
    if any(r['oom'] for r in rows):
        measure(4)
    at100 = [r for r in rows if r['physical_batch_size']==100]
    at50 = [r for r in rows if r['physical_batch_size']==50]
    if all(r['status']=='completed' for r in at100+at50):
        rate100 = 2/sum(1/r['logical_steps_per_second'] for r in at100)
        rate50 = 2/sum(1/r['logical_steps_per_second'] for r in at50)
        if rate100>1.05*rate50:
            for batch in (125,200,250):
                measure(batch)
    chosen,candidates = select_batch(rows)
    return chosen,rows,candidates


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--physical-batch-size',type=int,default=20)
    p.add_argument('--max-length',type=int,default=64,choices=(32,64,128))
    p.add_argument('--geometry',choices=('standard','scale'),default='scale')
    p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--measured',type=int,default=3)
    p.add_argument('--output',type=str)
    p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--C',type=float,default=1.)
    p.add_argument('--eps-scale',type=float,default=.1)
    p.add_argument('--seed',type=int,default=20261011)
    a = p.parse_args()
    cfg = Config(physical_batch_size=a.physical_batch_size,max_length=a.max_length,lr=a.lr,C=a.C,eps_scale=a.eps_scale,seed=a.seed)
    if a.worker:
        save_json(a.output,benchmark(cfg,a.geometry,a.gpu,a.warmup,a.measured))
    else:
        selected,_,candidates = search(a.gpu,cfg)
        save_json(RESULTS/'batch_selection.json',dict(selected=selected,candidates=candidates))

if __name__=='__main__': main()
