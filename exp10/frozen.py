"""Freeze four winners, all selection evidence, code and exact MF matrices."""
import json
from exp10 import RESULTS
from exp10.config import file_hash,save_json,METHODS
from exp10.audit import code_hashes
from exp10.matrices import get_matrix

def freeze(winners):
    assert set(winners)=={'cv','nlp'} and all(set(winners[t])==set(METHODS) for t in winners)
    path=RESULTS/'frozen_configs.json'
    if path.exists():
        assert load_frozen()==winners,'Frozen configurations are immutable'
        return winners
    for task in ('cv','nlp'):
        for method in METHODS:
            for n in ((225,250) if task=='cv' else (310,)):
                get_matrix(n,winners[task][method]['module'] if method==METHODS[0] else None)
    save_json(path,winners)
    names=('frozen_configs.json','assets_manifest.json','initial_search_configs.json','initial_search_results.json',
        'refinement_configs.json','refinement_decisions.json','search_results.json','recheck_results.json',
        'top2.json','selection_evidence.json','platform_validation.json')
    summaries=[p for stage in ('search','recheck') for p in sorted((RESULTS/stage).glob('*/*/summary.json'))]
    assert len(summaries)==144
    matrices=sorted((RESULTS/'matrix_cache').glob('*'))
    manifest=dict(code_sha256=code_hashes(),files_sha256={n:file_hash(RESULTS/n) for n in names},
        trial_summary_sha256={str(p.relative_to(RESULTS)):file_hash(p) for p in summaries},
        matrix_sha256={str(p.relative_to(RESULTS)):file_hash(p) for p in matrices},
        selection='Top-2: final internal accuracy, loss, trial ID. Winner: mean over four search/recheck seeds, loss, trial ID.',
        official_evaluation_affects_selection=False,privacy_scope='Per-training accounting only')
    save_json(RESULTS/'frozen_manifest.json',manifest)
    (RESULTS/'frozen_manifest.sha256').write_text(file_hash(RESULTS/'frozen_manifest.json')+'\n')
    return load_frozen()
def load_frozen():
    path=RESULTS/'frozen_manifest.json'
    assert file_hash(path)==(RESULTS/'frozen_manifest.sha256').read_text().strip()
    manifest=json.loads(path.read_text());assert manifest['code_sha256']==code_hashes()
    for name,expected in {**manifest['files_sha256'],**manifest['trial_summary_sha256'],**manifest['matrix_sha256']}.items():
        assert file_hash(RESULTS/name)==expected,f'Freeze evidence changed: {name}'
    return json.loads((RESULTS/'frozen_configs.json').read_text())
