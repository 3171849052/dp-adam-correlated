"""Read-only historical provenance and protected-tree audit."""
import json
from exp15d import ROOT, BASE, RESULTS
from exp15d.config import file_hash, save_json
from exp15d.support import protected_snapshot, verify_assets

UNCHANGED=('optimizer','data','cv_model','nlp_model','clipping','noise','privacy','bandinvmf','support')


def equivalent_sources():
    for name in UNCHANGED:
        expected=(ROOT/f'exp15b/{name}.py').read_text().replace('exp15b','exp15d').replace('Exp15b','Exp15d')
        if expected!=(BASE/f'{name}.py').read_text(): raise ValueError(f'Inherited implementation changed: {name}')
    return {name:'exact source after package rename' for name in UNCHANGED}


def prepare():
    equivalent=equivalent_sources();assets=verify_assets()
    path=RESULTS/'protected_before.json';snapshot=protected_snapshot()
    if path.exists():
        if json.loads(path.read_text())!=snapshot: raise ValueError('Protected files changed')
    else: save_json(path,snapshot)
    historical={str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/'exp15b/results').glob('*.json')}
    if json.loads((ROOT/'exp15b/results/final_audit.json').read_text())['status']!='passed': raise ValueError('Exp15b audit failed')
    save_json(RESULTS/'source_audit.json',dict(status='passed',equivalent_sources=equivalent,assets=assets,
        historical_sha256=historical,reuse=False,reason='All Exp15d trainings rerun; no full historical result equivalence certification required'))


def check_protected():
    if protected_snapshot()!=json.loads((RESULTS/'protected_before.json').read_text()): raise ValueError('Protected files changed')
    audit=json.loads((RESULTS/'source_audit.json').read_text())
    for path,h in audit['historical_sha256'].items():
        if file_hash(ROOT/path)!=h: raise ValueError(f'Historical aggregate changed: {path}')


if __name__=='__main__': prepare()
