"""Immutable freeze binds the 14 winners to all search/recheck evidence."""
import json
from exp9 import RESULTS
from exp9.config import METHODS, file_hash, save_json
from exp9.audit import code_hashes

def freeze(winners):
    assert set(winners)=={'cv','nlp'}
    assert all(set(winners[t])==set(METHODS) for t in winners)
    path=RESULTS/'frozen_configs.json'
    if path.exists():
        assert load_frozen()==winners, 'Frozen configuration is immutable'
        return winners
    save_json(path,winners)
    files=('frozen_configs.json','assets_manifest.json','cv_grid.json','nlp_grid.json',
           'grid_results.json','recheck_results.json','top2.json','selection_evidence.json','platform_validation.json')
    # Include every search and recheck summary; each summary binds its checkpoint and logs.
    evidence={str(p.relative_to(RESULTS)):file_hash(p) for stage in ('search','recheck')
              for p in sorted((RESULTS/stage).glob('*/*/summary.json'))}
    assert len(evidence)==377, 'Freeze requires all 321 grid and 56 recheck trial records'
    manifest=dict(schema_version=1,code_sha256=code_hashes(),
                  files_sha256={name:file_hash(RESULTS/name) for name in files},
                  trial_summary_sha256=evidence,
                  selection='Top-2 on seed 20261101 final internal-validation accuracy; '
                            'winner by mean over 20261101/20261102/20261103; canonical-ID tie break',
                  final_results_affect_selection=False,privacy_scope='Per-training accounting only')
    save_json(RESULTS/'frozen_manifest.json',manifest)
    (RESULTS/'frozen_manifest.sha256').write_text(file_hash(RESULTS/'frozen_manifest.json')+'\n')
    return load_frozen()

def load_frozen():
    path=RESULTS/'frozen_manifest.json'
    assert file_hash(path)==(RESULTS/'frozen_manifest.sha256').read_text().strip()
    manifest=json.loads(path.read_text());assert manifest['code_sha256']==code_hashes()
    for name, expected in {**manifest['files_sha256'],**manifest['trial_summary_sha256']}.items():
        assert file_hash(RESULTS/name)==expected, f'Freeze evidence changed: {name}'
    return json.loads((RESULTS/'frozen_configs.json').read_text())
