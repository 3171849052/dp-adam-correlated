"""Freeze all selection evidence and both horizons before official evaluation."""
import json
from exp12 import BASE, RESULTS
from exp12.config import file_hash, save_json
from exp12.audit import code_hashes


def freeze(winner):
    assert not (RESULTS/'frozen_manifest.json').exists(), 'Freeze is immutable'
    save_json(RESULTS/'frozen_configs.json',winner)
    names=['frozen_configs.json','assets_manifest.json','grid_results.json','recheck_results.json',
           'top2.json','selection_evidence.json','stage1.json']
    paths=[RESULTS/name for name in names]+sorted((RESULTS/'matrices').rglob('*'))
    paths=[p for p in paths if p.is_file()]
    summaries=[p for stage in ('search','recheck') for p in sorted((RESULTS/stage).glob('cv/*/summary.json'))]
    assert len(summaries)==24
    paths+=summaries
    save_json(RESULTS/'frozen_manifest.json',dict(code_sha256=code_hashes(),
        files_sha256={str(p.relative_to(RESULTS)):file_hash(p) for p in paths},
        selection='Exp9: Top-2 by fifth-epoch accuracy descending then canonical trial ID; '
                  'winner by three-seed mean descending then search trial ID',
        final_results_affect_selection=False))
    (RESULTS/'frozen_manifest.sha256').write_text(file_hash(RESULTS/'frozen_manifest.json')+'\n')
    return load_frozen()


def load_frozen():
    assert file_hash(RESULTS/'frozen_manifest.json')==(RESULTS/'frozen_manifest.sha256').read_text().strip()
    manifest=json.loads((RESULTS/'frozen_manifest.json').read_text())
    assert manifest['code_sha256']==code_hashes()
    for name,h in manifest['files_sha256'].items(): assert file_hash(RESULTS/name)==h, name
    return json.loads((RESULTS/'frozen_configs.json').read_text())
