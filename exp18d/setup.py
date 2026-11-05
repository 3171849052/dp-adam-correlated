"""Derive data only from local SST-2/cache; freeze inputs and parameters before training."""
import shutil
import subprocess
import sys
import numpy as np
import torch

from exp18d.common import (BASE, ROOT, GROUPS, HISTORICAL, WORKLOADS, SEEDS, T, K,
    EPOCHS, BATCH, configs, read, save, sha, write_csv, now, load_matrix)
from exp18.matrices import workload
from exp18.privacy import target_mu, fixed_epoch_sensitivity, epsilon_from_mu


def prepare():
    assert '/envs/curve/' in sys.executable
    assert not (BASE / 'acceptance.json').exists(), 'Experiment is already frozen'
    torch.set_num_threads(2)
    upstream = []
    for folder in ('exp16e', 'exp16f', 'exp18', 'exp18c'):
        upstream += list((ROOT / folder).glob('*.py'))
        upstream += [ROOT / folder / 'protocol.json']
    upstream += [ROOT / rel for rel in ('exp16e/results/final_freeze.json',
        'exp16e/results/nlp_summary.csv', 'exp16f/results/nlp_summary.csv',
        'exp16e/results/local_assets.json', 'exp16e/results/official_local_assets.json',
        'exp18c/best_hyperparameters.json', 'exp18c/selection_freeze.json', 'stp-ex2.pdf')]
    upstream += list((ROOT / 'cache/bert-tiny').glob('*'))
    upstream += [ROOT / 'data/sst2' / f'{split}.parquet' for split in ('train', 'validation')]
    upstream += [ROOT / 'exp16e/runtime/assets' / name for name in
                 ('nlp_tokens_128.pt', 'nlp_official_tokens_128.pt', 'nlp_split.npz')]
    save(BASE / 'upstream_manifest.json', {str(p.relative_to(ROOT)): sha(p) for p in upstream if p.is_file()})
    methods = {}
    historical = read(ROOT / 'exp16e/results/final_freeze.json')['winners']['nlp']
    stp = read(ROOT / 'exp18c/best_hyperparameters.json')
    assert stp['frozen']
    for group, names in GROUPS.items():
        for method in names:
            source = stp['groups'][group] if group == 'stpex2' else historical[HISTORICAL[method]]
            methods[method] = dict(lr=source['lr'], clip=source['clip'],
                                  eps=source['eps'] if group == 'stpex2' else 1e-8)
    save(BASE / 'hyperparameters.json', dict(frozen=True, methods=methods,
        sources=['exp16e/results/final_freeze.json', 'exp18c/best_hyperparameters.json'],
        stpex2_transfer='Root Exp18c frozen MNIST lr=0.08/clip=100/stability=0.005 transferred unchanged; no NLP tuning',
        root_exp18c_selected=True, historical_results_used_as_formal=False))
    protocol = dict(experiment='Exp18d', dataset='SST-2', model='BERT-Tiny', max_length=128,
        full_finetuning=True, dropout=0.1, precision='FP32; TF32 disabled',
        seeds=SEEDS, split_seed=20261008, train_examples=62000, internal_validation=5349,
        official_validation=872, epochs=EPOCHS, total_steps=T, steps_per_epoch=K,
        logical_batch=BATCH, physical_batch=BATCH, accumulation=1,
        epsilon=8, delta=1e-5, adjacency='add/remove zero-out fixed slot',
        accounting='Exp18 GDP whole fixed-epoch matrix mechanism; no sampling amplification',
        participation='Same per-seed Exp16e permutation repeated five epochs',
        noise_seed='seed+1', dropout_seed='seed+100000+logical_step',
        optimizer='Exact exp18.optimizer.Update for Adam/Momentum/STP-Ex2, beta1=.9, beta2=.999, weight_decay=0',
        stpex2='d0=1; scale by previous completed d; adaptive alpha, 24 bisections, 1e-6 backoff; no final denominator',
        methods=methods, B2LT=dict(T=T,K=K,P=K,Q=5,source='Exp18 solver with new dimensions, no Exp16 matrices'),
        evaluation='Internal validation each epoch; all 872 official validation examples once at final step; no early stopping',
        formal_trainings=60, historical_results='Reference only; all 60 formal trials train anew',
        statistics='10 seeds, sample SD ddof=1, Student-t 95% CI',
        privacy_scope='Per training epsilon=8; 60 trainings are not jointly epsilon=8; internal raw losses/clipping diagnostics are not DP releases',
        gpu='Physical 1,2,3, two independent CUDA-isolated processes per GPU',
        smoke='Six methods, seed 20261111, 3 steps, batch 1000, length128, internal validation64; separate directories',
        selection='All hyperparameters frozen from specified historical sources before training; no search')
    save(BASE / 'protocol.json', protocol)
    inventory = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name','--format=csv,noheader'], text=True)
    gpus = {int(v[0]): v[1].strip() for line in inventory.splitlines() for v in [line.split(',')]}
    save(BASE / 'scheduling.json', dict(physical_gpus=[1,2,3], slots=[1,1,2,2,3,3],
        max_processes=6, max_per_gpu=2, gpu_uuids={str(g):gpus[g] for g in (1,2,3)}, inventory=inventory))

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(str(ROOT / 'cache/bert-tiny'),local_files_only=True)
    assets = BASE / 'assets'; assets.mkdir(exist_ok=True)
    raw = None
    for split, filename in (('train','nlp_tokens_128.pt'),('validation','nlp_official_tokens_128.pt')):
        frame = pd.read_parquet(ROOT / 'data/sst2' / f'{split}.parquet')
        assert len(frame) == (67349 if split == 'train' else 872)
        tokens = dict(tokenizer(frame['sentence'].tolist(),padding='max_length',max_length=128,
            truncation=True,return_tensors='pt'), labels=torch.tensor(frame['label'].tolist()))
        old = torch.load(ROOT / 'exp16e/runtime/assets' / filename,weights_only=True)
        assert tokens.keys() == old.keys()
        assert all(torch.equal(v,old[k]) for k,v in tokens.items()), 'Historical tokens changed'
        torch.save(tokens, assets / filename)
        if split == 'train': raw = frame
    train, valid = train_test_split(np.arange(len(raw)),train_size=62000,random_state=20261008,stratify=raw['label'])
    train = np.random.default_rng(20261009).permutation(train).astype(np.int64)
    valid = np.sort(valid).astype(np.int64)
    with np.load(ROOT / 'exp16e/runtime/assets/nlp_split.npz') as old:
        np.testing.assert_array_equal(train,old['train'])
        np.testing.assert_array_equal(valid,old['search_validation'])
    np.savez(assets / 'nlp_split.npz',train=train,search_validation=valid)
    files = list(assets.glob('*')) + list((ROOT / 'cache/bert-tiny').glob('*'))
    save(BASE / 'assets.json', {str(p.relative_to(ROOT)):sha(p) for p in files if p.is_file()})
    save(BASE / 'data_audit.json', dict(passed=True, time=now(), source='data/sst2/*.parquet + cache/bert-tiny',
        downloads=False, tokens_equal_exp16e=True, split_equal_exp16e=True,
        sizes=[len(train),len(valid),872], tokenizer='BertTokenizer local_files_only=True'))
    for source in ('exp16e','exp16f'):
        dest = BASE / 'history'; dest.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / source / 'results/nlp_summary.csv',dest / f'{source}_nlp_summary.csv')
    from exp18d.matrices import optimize
    for name in WORKLOADS.values(): optimize(name)
    privacy = {}; audits = {}; mu = target_mu(8,1e-5)
    for name in WORKLOADS.values():
        m = load_matrix(name); meta = m['metadata']
        np.testing.assert_allclose(m['A'],workload(name,T),atol=1e-12)
        assert meta['first_order_pass'] and meta['selected']=='b2lt'
        assert (meta['T'],meta['K'],meta['P'],meta['Q']) == (T,K,K,5)
        audits[name] = meta['audit']
    for group,names in GROUPS.items():
        for method in names:
            iid = method.endswith('-iid'); m = None if iid else load_matrix(WORKLOADS[group])
            C = np.eye(T) if iid else m['C']; sensitivity = fixed_epoch_sensitivity(C,EPOCHS,K)
            clip = methods[method]['clip']; sigma = clip*sensitivity/mu
            eps = epsilon_from_mu(clip*sensitivity/sigma,1e-5)
            assert abs(eps-8)<1e-9
            privacy[method] = dict(epsilon=eps,delta=1e-5,mu=mu,sensitivity=sensitivity,
                clip=clip,sigma_sum=sigma,sigma_average=sigma/BATCH,participations=EPOCHS,
                participation_interval=K,T=T,calibration='Exp18 GDP',sampling_amplification=False,
                clip_space='previous-d preconditioned' if group=='stpex2' else 'original',
                matrix_sha256=None if iid else m['metadata']['sha256'],
                adjacency='add/remove zeroed fixed slot',scope='one training, not 60-run composition')
    save(BASE / 'privacy_audit.json',dict(passed=True,methods=privacy,matrices=audits))
    rows = configs(); write_csv(BASE / 'trial_plan.csv',rows,list(rows[0]))
    save(BASE / 'preflight.json',dict(passed=True,time=now(),python=sys.executable,formal_trials=len(rows)))
    print('Prepared offline assets, frozen parameters, new matrices and epsilon=8 GDP audits',flush=True)

if __name__ == '__main__': prepare()
