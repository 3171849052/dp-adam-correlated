"""Read-only historical audit and explicit, verified result references."""
import json
from functools import lru_cache
import numpy as np
from exp15b import ROOT, BASE, RESULTS
from exp15b.config import Trial, file_hash, array_hash, json_hash, save_json
from exp15b.support import matrices, verify_assets, protected_snapshot, code_hashes
from exp15b.privacy import calibrate

CORE_NAMES = ('config','train','data','cv_model','nlp_model','optimizer','clipping','privacy','noise','bandinvmf','support')


def read(path):
    return json.loads(path.read_text())


def verify_equivalent_sources():
    unchanged = ('data','cv_model','nlp_model','clipping','privacy','noise','bandinvmf')
    checked = {}
    for name in unchanged:
        old = (ROOT/f'exp15/{name}.py').read_text().replace('exp15','exp15b').replace('Exp15','Exp15b')
        if old != (BASE/f'{name}.py').read_text():
            raise ValueError(f'Historical execution changed: {name}')
        checked[name] = 'exact source after package rename'
    old = (ROOT/'exp15/train.py').read_text().replace('exp15','exp15b').replace('Exp15','Exp15b')
    old = old.replace("('search','final')", "('search','review','final')").replace("('0','1','2','3')", "('0','1','2')").replace('=0/1/2/3','=0/1/2')
    if old != (BASE/'train.py').read_text():
        raise ValueError('Training behavior changed beyond authorization and GPU restriction')
    old = (ROOT/'exp15/optimizer.py').read_text().replace('exp15','exp15b').replace('Exp15','Exp15b')
    old = old.replace("method=='momentum'", "method in ('momentum','momentum+iid')")
    if old != (BASE/'optimizer.py').read_text():
        raise ValueError('Optimizer changed beyond identical IID Momentum branch')
    checked.update(train='only stage authorization and GPU allowlist changed',
                   optimizer='only IID method included in uncorrected Momentum branch')
    return checked


@lru_cache(None)
def verify_historical(path_string):
    path = ROOT/path_string
    r = read(path)
    cfg = Trial(**{k:r[k] for k in Trial.__dataclass_fields__})
    if cfg.method == 'momentum+iid' or r['status'] != 'completed':
        raise ValueError('Invalid historical reuse')
    if r['completed_epochs'] != 5 or r['optimizer_steps'] != cfg.total_steps or r['noise_draws'] != cfg.total_steps:
        raise ValueError('Incomplete historical run')
    if r['seed'] != cfg.seed or r['official_evaluation_used'] != cfg.formal:
        raise ValueError('Historical seed/evaluation mismatch')
    for name in CORE_NAMES:
        key = f'exp15/{name}.py'
        if file_hash(ROOT/key) != r['code_sha256'][key]:
            raise ValueError(f'Historical code hash mismatch: {key}')
    for name,h in r['file_sha256'].items():
        if file_hash(path.parent/name) != h:
            raise ValueError(f'Historical artifact hash mismatch: {path.parent/name}')
    resolved = read(path.parent/'config.json')
    actual_hash = resolved.pop('config_sha256')
    if json_hash(resolved) != actual_hash or actual_hash != r['config_sha256']:
        raise ValueError('Historical resolved configuration hash mismatch')
    for key,value in cfg.protocol().items():
        if resolved[key] != value:
            raise ValueError(f'Historical protocol differs: {key}')
    d,S,W,A,provenance = matrices(cfg)
    with np.load(path.parent/'matrices.npz') as saved:
        for key,value in [('coefficients',d),('strategy',S),('workload',W),('actual_workload',A)]:
            np.testing.assert_array_equal(saved[key],value)
    if provenance != r['mechanism']:
        raise ValueError('Historical mechanism provenance mismatch')
    cal = calibrate(S,cfg.protocol())
    for key,value in cal.items():
        if r['calibration'][key] != value:
            raise ValueError(f'Historical calibration mismatch: {key}')
    if abs(r['actual_epsilon']-8) > 1e-7:
        raise ValueError('Historical privacy mismatch')
    if array_hash(np.load(path.parent/'train_order.npy')) != r['train_order_sha256']:
        raise ValueError('Historical train order mismatch')
    return r


def prepare():
    snapshot = RESULTS/'protected_before.json'
    if snapshot.exists():
        if read(snapshot) != protected_snapshot():
            raise ValueError('Protected historical files changed')
    else:
        save_json(snapshot,protected_snapshot())
    sources = verify_equivalent_sources()
    old = ROOT/'exp15/results'
    names = ('frozen_grid','search_results','selected_configs','formal_results','statistics','final_audit')
    manifests = {name:file_hash(old/f'{name}.json') for name in names}
    if read(old/'final_audit.json')['status'] != 'passed':
        raise ValueError('Historical final audit failed')
    if file_hash(old/'frozen_grid.json') != (old/'frozen_grid.sha256').read_text().strip():
        raise ValueError('Historical frozen grid changed')
    assets = verify_assets()
    index = {}
    for stage,filename in [('search','search_results'),('final','formal_results')]:
        for row in read(old/f'{filename}.json'):
            path = old/stage/row['replicate']/row['task']/row['trial_id']/'summary.json'
            relative = str(path.relative_to(ROOT))
            verified = verify_historical(relative)
            if verified != row:
                raise ValueError('Historical aggregate differs from raw summary')
            index[row['trial_id']] = dict(path=relative,sha256=file_hash(path),stage=stage)
    result = dict(status='passed',historical_manifests=manifests,assets=assets,
                  equivalent_sources=sources,verified_runs=len(index),index=index,
                  formal_metrics_used_for_grid_or_selection=False,
                  reuse_requires_bit_exact_smoke=True,code_sha256=code_hashes())
    save_json(RESULTS/'source_audit.json',result)
    return result


def historical_search():
    audit = read(RESULTS/'source_audit.json')
    return [verify_historical(v['path']) for v in audit['index'].values() if v['stage']=='search']


def reuse(cfg):
    if cfg.method == 'momentum+iid':
        return False
    audit = read(RESULTS/'source_audit.json')
    entry = audit['index'].get(cfg.id)
    if entry is None:
        return False
    if read(RESULTS/'historical_equivalence.json')['status'] != 'passed':
        raise ValueError('No numerical equivalence evidence')
    if file_hash(ROOT/entry['path']) != entry['sha256']:
        raise ValueError('Historical source summary changed')
    original = verify_historical(entry['path'])
    if any(original[k] != v for k,v in cfg.values().items()):
        raise ValueError('Historical trial identity mismatch')
    row = dict(original,reuse=dict(source_summary=entry['path'],source_sha256=entry['sha256'],
                                  source_directory=str((ROOT/entry['path']).parent.relative_to(ROOT))))
    if (cfg.output/'summary.json').exists():
        if read(cfg.output/'summary.json') != row:
            raise ValueError('Reuse reference changed')
    else:
        save_json(cfg.output/'summary.json',row)
    return True


def verify_reused(cfg,row):
    source = row['reuse']
    if file_hash(ROOT/source['source_summary']) != source['source_sha256']:
        raise ValueError('Reused summary hash mismatch')
    original = verify_historical(source['source_summary'])
    if {k:v for k,v in row.items() if k!='reuse'} != original or row['trial_id'] != cfg.id:
        raise ValueError('Reused result differs from verified source')


if __name__ == '__main__':
    r=prepare()
    print(json.dumps(dict(status=r['status'],verified_runs=r['verified_runs'])),flush=True)
