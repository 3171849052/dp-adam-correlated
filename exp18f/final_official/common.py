"""Frozen official experiment; every output stays in this directory."""
import csv
import hashlib
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo
from exp18f.final_official import BASE, ROOT

PYTHON = '/home/longt29/miniconda3/envs/curve/bin/python'
MODEL_PATH = ROOT / 'cache/bert-tiny'
SEEDS = [20261111, 20261112, 20261113, 20261114, 20261115]
T, K, EPOCHS, BATCH = 310, 62, 5, 1000
METHODS = ['stpex2-iid', 'stpex2-b2lt-momentum-bias']
BASELINES = ['adam-iid', 'adam-b2lt-sgd', 'momentum-iid', 'momentum-b2lt-momentum']
WORKLOADS = {'stpex2': 'momentum-bias'}
PARAMS = dict(lr=.025, clip=100, eps=.001)


def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def read(path): return json.loads(path.read_text())
def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(tmp, path)
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()
def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
def configs(stage='formal'):
    return read(BASE / 'plans' / f'{stage}.json')
def outdir(cfg):
    parent = BASE / cfg['stage']
    if cfg['stage'] == 'smoke': parent = parent / f"gpu_pair_{cfg['gpu_pair']}"
    return parent / f"{cfg['method']}_seed{cfg['seed']}"
def load_matrix(name):
    import numpy as np
    path = BASE / 'matrices' / f'{name}.npz'
    meta = read(path.with_suffix('.json'))
    assert sha(path) == meta['sha256']
    with np.load(path) as f: return dict({k: f[k].copy() for k in f.files}, metadata=meta)
def calibration(cfg):
    import numpy as np
    from exp18f.final_official.privacy import target_mu, fixed_epoch_sensitivity, epsilon_from_mu
    iid = cfg['method'].endswith('-iid')
    m = None if iid else load_matrix('momentum-bias')
    C = np.eye(T) if iid else m['C']
    sens = fixed_epoch_sensitivity(C, EPOCHS, K); mu = target_mu(8, 1e-5)
    sigma = cfg['clip'] * sens / mu
    epsilon = epsilon_from_mu(cfg['clip'] * sens / sigma, 1e-5)
    assert abs(epsilon - 8) < 1e-9
    return dict(epsilon=epsilon, delta=1e-5, mu=mu, sensitivity=sens, clip=cfg['clip'],
        sigma_sum=sigma, sigma_average=sigma/BATCH, participations=5, participation_interval=K, T=T,
        sampling_amplification=False, adjacency='add/remove zeroed fixed slot',
        clip_space='previous-d preconditioned', matrix_sha256=None if iid else m['metadata']['sha256'],
        scope='one full training; not composition of search/formal trials')
def env(gpu):
    e = os.environ.copy()
    e.update(CUDA_VISIBLE_DEVICES=read(BASE/'scheduling.json')['gpu_uuids'][str(gpu)],
        EXP18F_FINAL_GPU=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID', PYTHONDONTWRITEBYTECODE='1')
    return e
def verify_seal():
    for rel, h in read(BASE/'input_manifest.json').items():
        assert sha(ROOT/rel) == h, f'Input changed: {rel}'
    for rel, h in read(BASE/'code_manifest.json').items():
        assert sha(BASE/rel) == h, f'Code changed: {rel}'
def verify_freeze():
    frozen = read(ROOT/'exp18f/rounded_freeze.json')
    assert frozen['frozen'] and not frozen['official_validation_used']
    assert frozen['parameters'] == PARAMS
    assert len(configs()) == 10
    for cfg in configs():
        assert {k: cfg[k] for k in PARAMS} == PARAMS
def completed(cfg):
    out = outdir(cfg)
    if not (out/'complete.json').exists(): return None
    assert read(out/'config.json') == cfg
    for name, h in read(out/'complete.json').items():
        assert sha(out/name) == h, f'Artifact changed: {out/name}'
    trace = [json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in trace] == list(range(1, cfg['steps']+1))
    result = read(out/'summary.json')
    assert result['status'] == 'complete' and result['steps'] == cfg['steps']
    return result
