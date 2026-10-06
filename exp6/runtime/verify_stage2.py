from exp6.runtime import EXP, require_curve
require_curve()
import hashlib
import json
from pathlib import Path
import numpy as np
from exp6.config import METHODS, FIXED, FINAL_SEEDS
from exp6.audit import completed
from exp6.final_runner import final_jobs
from exp6.search import rank

folder=EXP / 'results/search'
selected=json.loads((folder / 'selected_configs.json').read_text())
first=json.loads((folder / 'initial_search_summary.json').read_text())
resume=json.loads((folder / 'search_summary.json').read_text())
assert first['actual_trained_trials'] == first['total_unique_tuning_trials'] == 85
assert resume['actual_trained_trials'] == 0 and resume['reused_preexisting_trials'] == 85
assert resume['actual_trained_trials_total'] == 85
assert hashlib.sha256((folder / 'selected_configs.json').read_bytes()).hexdigest() == json.loads((folder / 'initial_selected_sha256.json').read_text())['sha256']
index=json.loads((folder / 'completed_trials.json').read_text())
assert index['count'] == len(index['trials']) == 85
pool=[json.loads((Path(row['result_dir']) / 'summary.json').read_text()) for row in index['trials']]
chosen=[]; arrays=[]
for method in METHODS:
    f=selected['methods'][method]
    assert f['fixed'] == FIXED and f['status'] == 'frozen'
    configuration=json.loads((Path(f['result_dir']) / 'config.json').read_text())
    s=completed(dict(trial=configuration['trial'],result_dir=f['result_dir']))
    candidates=[r for r in pool if r['trial']['method']==method and
                (not method.endswith('-scale') or r['trial']['geom_eps']==selected['shared_scale_geom_eps'])]
    assert min(candidates,key=rank)['trial_id'] == f['selected_trial_id'] == s['trial_id']
    assert f['selected_fingerprint'] == s['fingerprint']
    if method.endswith('-scale'): assert f['hyperparameters']['geom_eps'] == selected['shared_scale_geom_eps']
    chosen.append(s)
    arrays.append(dict(np.load(Path(f['result_dir']) / 'matrices.npz')))
for key in ('initialization_sha256','order_sha256','augmentation_sha256','innovation_sha256'):
    assert all(s[key] == chosen[0][key] for s in chosen)
for key in ('coefficients','strategy','workload_coefficients','workload','noising_matrix'):
    np.testing.assert_array_equal(arrays[1][key],arrays[3][key])
manifest=json.loads((folder / 'prepared_final_manifest.json').read_text())
assert manifest['status']=='prepared_not_launched' and len(manifest['jobs'])==12
assert manifest['jobs'] == final_jobs(selected,list(FINAL_SEEDS))
assert not (EXP / 'results/final').exists()
result=dict(status='passed',unique_full_tuning_trials=85,first_run_trained=85,
            unique_stage_reused_configs=first['stage_reuse_events'],resume_trained=0,resume_reused=85,
            selected_unchanged=True,all_selected_are_utility_winners=True,
            paired_initialization_order_augmentation_innovations=True,
            raw_scale_bandinvmf_matrices_identical=True,privacy=dict(epsilon=8.,delta=1e-5),
            final_trials_prepared=12,final_launched=False,
            selected={m:dict(hyperparameters=selected['methods'][m]['hyperparameters'],
                final_test_top1=selected['methods'][m]['final_test_top1']) for m in METHODS})
(EXP / 'results/stage2_verification.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(result,indent=2))
