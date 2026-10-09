"""Freeze seven search winners once; verify content hashes before every final trial."""
import json
from exp8b import RESULTS,ROOT
from exp8b.config import *
from exp8b.audit import source_hashes,audit_trial


def freeze(selected):
    assert set(selected)==set(METHODS)
    for m,r in selected.items():
        Config(**{k:r[k] for k in ('method','seed','lr','C','eps_scale')})
        assert m==r['method'] and r['seed']==SEED and r['status']=='completed' and r['completed_epochs']==5
        evidence=audit_trial(ROOT/r['result_dir'])
        assert evidence==r and r['category']=='search' and not r['official_validation_used']
    path=RESULTS/'frozen_configs.json'
    if path.exists():
        assert load_frozen()==selected,'Frozen settings are immutable'
        return
    save_json(path,selected)
    save_json(RESULTS/'frozen_manifest.json',dict(frozen_configs_sha256=file_hash(path),
              selected_configs_sha256=file_hash(RESULTS/'search/selected_configs.json'),source_hashes=source_hashes(),
              split_sha256=file_hash(RESULTS/'split.npz'),tokens_sha256=file_hash(RESULTS/'tokens_128.pt'),
              pretrained_sha256=file_hash(MODEL_PATH/'pytorch_model.bin'),
              selection_objective='search-validation Accuracy at epoch 5 only',search_seed=SEED,
              final_seeds=list(FINAL_SEEDS),official_validation_used=False,
              provenance={m:dict(result_dir=r['result_dir'],summary_sha256=file_hash(ROOT/r['result_dir']/'summary.json'),
                                 checkpoint_sha256=r['checkpoint_sha256']) for m,r in selected.items()}))
    manifest_hash=file_hash(RESULTS/'frozen_manifest.json')
    (RESULTS/'frozen_manifest.sha256').write_text(manifest_hash+'\n')


def load_frozen():
    path=RESULTS/'frozen_configs.json';mp=RESULTS/'frozen_manifest.json'
    assert file_hash(mp)==(RESULTS/'frozen_manifest.sha256').read_text().strip()
    manifest=json.loads(mp.read_text())
    assert manifest['frozen_configs_sha256']==file_hash(path)
    assert manifest['selected_configs_sha256']==file_hash(RESULTS/'search/selected_configs.json')
    assert manifest['source_hashes']==source_hashes()
    for key,p in (('split_sha256',RESULTS/'split.npz'),('tokens_sha256',RESULTS/'tokens_128.pt'),('pretrained_sha256',MODEL_PATH/'pytorch_model.bin')):
        assert manifest[key]==file_hash(p)
    settings=json.loads(path.read_text());assert set(settings)==set(METHODS)
    for m,r in settings.items():
        proof=manifest['provenance'][m]
        assert proof['summary_sha256']==file_hash(ROOT/proof['result_dir']/'summary.json')
        assert proof['checkpoint_sha256']==file_hash(ROOT/proof['result_dir']/'checkpoint.pt')
    return settings
