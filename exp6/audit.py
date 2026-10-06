"""Fingerprints include training code, offline assets, protocol and library versions."""
from exp6.runtime import ROOT, EXP, output_path
from functools import lru_cache
import hashlib
from importlib.metadata import version
from pathlib import Path
import json
import math
from exp6.config import FIXED, Trial
from exp6.model import checkpoint_sha256
from exp2.model import checkpoint_path

TRAINING_FILES = ('__init__.py','runtime.py','config.py','data.py','model.py','lora.py',
                  'geometry.py','clipping.py','privacy.py','mechanism.py','train.py','audit.py')


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def environment():
    paths = [EXP / n for n in TRAINING_FILES] + [ROOT / 'exp2' / n for n in ('model.py','privacy.py','bandinvmf.py')]
    return dict(training_source_sha256={str(p.relative_to(ROOT)):checkpoint_sha256(p) for p in paths},
        assets_sha256={str(p.relative_to(ROOT)):checkpoint_sha256(p) for p in
            [checkpoint_path()] + [ROOT / 'data/cifar-100-python' / n for n in ('train','test','meta')]},
        versions={p:version(p) for p in ('torch','torchvision','numpy','scipy','timm','jax','jax_privacy','safetensors')})


def fingerprint(trial, smoke=False):
    return digest(dict(fixed=FIXED,trial=trial.asdict(),smoke=smoke,environment=environment()))


def assert_finite(value):
    if isinstance(value,dict):
        for v in value.values(): assert_finite(v)
    elif isinstance(value,list):
        for v in value: assert_finite(v)
    elif isinstance(value,float):
        assert math.isfinite(value), 'Nonfinite audit value'


def completed(job):
    """A completed trial must have all artifacts and matching full fingerprint."""
    import numpy as np
    from exp6.mechanism import matrices
    from exp6.privacy import calibration
    folder = output_path(job['result_dir'])
    trial = Trial(**job['trial'])
    for name in ('summary.json','config.json','steps.jsonl','geometry.jsonl','matrices.npz','train_order.npy','final.pt'):
        assert (folder / name).is_file(), f'Incomplete trial: missing {folder / name}'
    summary = json.loads((folder / 'summary.json').read_text())
    config = json.loads((folder / 'config.json').read_text())
    assert summary['status'] == 'completed' and summary['smoke'] == job.get('smoke',False)
    assert summary['trial_id'] == config['trial_id'] == trial.identity
    assert summary['trial'] == config['trial'] == trial.asdict()
    assert all(checkpoint_sha256(folder / name) == sha for name,sha in summary['artifact_sha256'].items()), f'Artifact conflict: {folder}'
    assert set(summary['artifact_sha256']) == {'final.pt','matrices.npz','train_order.npy','steps.jsonl','geometry.jsonl'}
    assert config['fixed'] == FIXED, f'Protocol conflict: {folder}'
    assert summary['optimizer_steps'] == summary['noise_draws'] == 250
    expected = fingerprint(trial)
    assert config['fingerprint'] == summary['fingerprint'] == expected, f'Fingerprint conflict: {folder}'
    assert summary['test_examples'] == 10000
    assert summary['final_test_top1'] == summary['test_top1'] and 0 <= summary['final_test_top1'] <= 1
    assert abs(summary['final_privacy']['epsilon']-8) < 1e-7 and summary['final_privacy']['delta'] == 1e-5
    assert len(summary['augmentation_sha256']) == len(summary['innovation_sha256']) == 250
    rows = [json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows] == list(range(1,251))
    assert all(r['step'] == r['noise_draws'] and r['train_examples'] == 1000 for r in rows)
    assert summary['final_train'] == rows[-1]
    geometry = [json.loads(line) for line in (folder / 'geometry.jsonl').read_text().splitlines()]
    assert [r['step'] for r in geometry] == list(range(1,251))
    assert all(len(r['layers']) == 48 for r in geometry)
    coeff, strategy, workload = matrices(trial)
    saved = np.load(folder / 'matrices.npz')
    for name, value in (('coefficients',coeff),('strategy',strategy),('workload_coefficients',workload)):
        np.testing.assert_array_equal(saved[name],value)
    assert config['privacy'] == summary['calibration'] == calibration(strategy,trial)
    assert all(np.isfinite(saved[n]).all() for n in saved.files)
    order = np.load(folder / 'train_order.npy')
    np.testing.assert_array_equal(np.sort(order),np.arange(50000))
    assert hashlib.sha256(order.tobytes()).hexdigest() == summary['order_sha256'] == config['order_sha256']
    for value in (summary,config,rows,geometry): assert_finite(value)
    return summary


def diagnostics(folder):
    rows = [json.loads(line) for line in (output_path(folder) / 'steps.jsonl').read_text().splitlines()]
    clipping = {k:dict(mean=sum(r[k] for r in rows)/len(rows),final=rows[-1][k]) for k in
        ('clipping_fraction','clip_factor_mean','raw_per_example_norm_mean','query_norm','noise_std')}
    if 'scaled_per_example_norm_mean' in rows[-1]:
        clipping['scaled_per_example_norm_mean'] = dict(mean=sum(r['scaled_per_example_norm_mean'] for r in rows)/len(rows),final=rows[-1]['scaled_per_example_norm_mean'])
    keys = [k for k in rows[-1] if any(k.startswith(p) for p in
        ('A_fro','B_fro','actual_weight_energy','PA_','PB_','B_PA_','PB_inverse_'))]
    geometry = {k:dict(mean=sum(r[k] for r in rows)/len(rows),final=rows[-1][k]) for k in keys}
    return dict(clipping=clipping,geometry=geometry,per_layer_artifact=str(Path(folder) / 'geometry.jsonl'))

