import csv
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from exp18d import BASE, ROOT
from exp18.common import read, save, sha, digest, PYTHON, GROUPS

MODEL_PATH = ROOT / 'cache/bert-tiny'
SEEDS = list(range(20261111, 20261121))
T, K, EPOCHS, BATCH = 310, 62, 5, 1000
WORKLOADS = {'adam': 'sgd', 'momentum': 'momentum', 'stpex2': 'momentum-bias'}
HISTORICAL = {'adam-iid': 'adam-iid', 'adam-b2lt-sgd': 'adam-mf-sgd-b2lt',
              'momentum-iid': 'momentum-iid', 'momentum-b2lt-momentum': 'momentum-mf-momentum-b2lt'}


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def configs(stage='formal'):
    frozen = read(BASE / 'hyperparameters.json')['methods']
    return [dict(stage=stage, method=method, group=group, seed=seed,
                 steps=3 if stage == 'smoke' else T, **frozen[method])
            for seed in (SEEDS[:1] if stage == 'smoke' else SEEDS)
            for group, methods in GROUPS.items() for method in methods]


def outdir(cfg):
    return BASE / cfg['stage'] / f"{cfg['method']}_seed{cfg['seed']}"


def write_csv(path, rows, fields):
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def load_matrix(name):
    import numpy as np
    path = BASE / 'matrices' / f'{name}.npz'
    meta = read(path.with_suffix('.json'))
    assert sha(path) == meta['sha256']
    with np.load(path) as f:
        return dict({k: f[k].copy() for k in f.files}, metadata=meta)


def env(gpu):
    e = os.environ.copy()
    e.update(CUDA_VISIBLE_DEVICES=read(BASE / 'scheduling.json')['gpu_uuids'][str(gpu)],
             EXP18D_GPU=str(gpu), CUDA_DEVICE_ORDER='PCI_BUS_ID',
             PYTHONDONTWRITEBYTECODE='1')
    return e


def seal():
    paths = list(BASE.glob('*.py')) + list((BASE / 'tests').glob('*.py'))
    paths += [BASE / n for n in ('protocol.json', 'hyperparameters.json', 'scheduling.json',
              'assets.json', 'privacy_audit.json', 'upstream_manifest.json', 'trial_plan.csv', 'start_background.sh')]
    paths += list((BASE / 'matrices').glob('*'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def verify_seal():
    acceptance = read(BASE / 'acceptance.json')
    assert acceptance['passed'] and seal() == acceptance['code'], 'Accepted code/config changed'
    for rel, h in read(BASE / 'upstream_manifest.json').items():
        assert sha(ROOT / rel) == h, f'Upstream changed: {rel}'


def completed(cfg):
    out = outdir(cfg)
    if not (out / 'complete.json').exists():
        return None
    assert read(out / 'config.json') == cfg
    for name, h in read(out / 'complete.json').items():
        assert sha(out / name) == h, f'Changed artifact: {out / name}'
    trace = [json.loads(line) for line in (out / 'steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in trace] == list(range(1,cfg['steps']+1)), 'Incomplete step trace'
    summary = read(out / 'summary.json')
    assert summary['steps'] == cfg['steps'] and summary['status'] == 'complete'
    assert summary['formal'] == (cfg['stage'] == 'formal')
    return summary
