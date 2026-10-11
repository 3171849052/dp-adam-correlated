"""Immutable selection evidence, source, and exact search/formal matrices."""
import json
from exp10b import RESULTS
from exp10b.config import file_hash,immutable_json,save_json
from exp10b.audit import code_hashes
from exp10b.matrices import get_matrix

def freeze(winners):
    path=RESULTS/'frozen_manifest.json'
    if path.exists():
        assert load_frozen()==winners, 'Frozen configurations are immutable'
        return winners
    for task in ('cv','nlp'):
        for n in ((225,250) if task=='cv' else (310,)):
            for lam in (0.,winners[task]['module']): get_matrix(n,lam)
    immutable_json(RESULTS/'frozen_configs.json',winners)
    names=('frozen_configs.json','assets_manifest.json','full_plan.json','stage1_results.json',
           'stage2_results.json','selection_evidence.json','platform_validation.json')
    summaries=[p for stage in ('stage1','stage2') for p in sorted((RESULTS/stage).glob('*/*/summary.json'))]
    assert len(summaries)==120
    matrices=sorted((RESULTS/'matrix_cache').glob('*'))
    manifest=dict(code_sha256=code_hashes(),files_sha256={n:file_hash(RESULTS/n) for n in names},
        trial_summary_sha256={str(p.relative_to(RESULTS)):file_hash(p) for p in summaries},
        matrix_sha256={str(p.relative_to(RESULTS)):file_hash(p) for p in matrices},
        selection='10-seed mean final internal accuracy; ties: smaller lambda; positive winner for formal',
        official_evaluation_affects_selection=False,privacy_scope='Each training separately')
    save_json(path,manifest)
    (RESULTS/'frozen_manifest.sha256').write_text(file_hash(path)+'\n')
    return load_frozen()

def load_frozen():
    path=RESULTS/'frozen_manifest.json'
    assert file_hash(path)==(RESULTS/'frozen_manifest.sha256').read_text().strip()
    manifest=json.loads(path.read_text())
    assert manifest['code_sha256']==code_hashes(), 'Frozen source changed'
    for name,expected in {**manifest['files_sha256'],**manifest['trial_summary_sha256'],**manifest['matrix_sha256']}.items():
        assert file_hash(RESULTS/name)==expected, f'Freeze evidence changed: {name}'
    return json.loads((RESULTS/'frozen_configs.json').read_text())
