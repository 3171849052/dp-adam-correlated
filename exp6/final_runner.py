"""Explicit launch of 12 full trials after reviewing per-method frozen configs."""
from exp6.runtime import EXP, output_path, require_curve
import argparse
import json
from pathlib import Path
from exp6.config import Trial, METHODS, FIXED, FINAL_SEEDS
from exp6.launch_batch import launch


def final_jobs(selected, seeds):
    assert selected['status'] == 'frozen' and selected['fixed'] == FIXED
    assert seeds == list(FINAL_SEEDS), 'Final validation seeds are fixed'
    assert set(selected['methods']) == set(METHODS)
    eps = selected['shared_scale_geom_eps']
    trials = []
    for seed in seeds:
        for method in METHODS:
            frozen = selected['methods'][method]
            assert frozen['status'] == 'frozen' and frozen['method'] == method and frozen['fixed'] == FIXED
            hp = frozen['hyperparameters']
            assert set(hp) == {'lr','C','geom_eps'}
            if method.endswith('-scale'): assert hp['geom_eps'] == eps
            trials.append(Trial(method,seed,**hp,physical_batch_size=frozen['physical_batch_size']))
    return [dict(trial=t.asdict(),result_dir=str(EXP / 'results/final' / t.identity)) for t in trials]


def run(selected_path, seeds):
    require_curve()
    selected_path = output_path(selected_path)
    selected = json.loads(selected_path.read_text())
    # Verify selected tuning trials under the current training fingerprint.
    from exp6.audit import completed, fingerprint
    for method in METHODS:
        frozen = selected['methods'][method]
        config = json.loads((Path(frozen['result_dir']) / 'config.json').read_text())
        summary = completed(dict(trial=config['trial'],result_dir=frozen['result_dir']))
        assert summary['fingerprint'] == frozen['selected_fingerprint']
        assert summary['trial_id'] == frozen['selected_trial_id']
        trial = Trial(**config['trial'])
        assert fingerprint(trial) == frozen['selected_fingerprint']
        assert {k:getattr(trial,k) for k in ('C','lr','geom_eps')} == frozen['hyperparameters']
    jobs = final_jobs(selected,seeds)
    final = EXP / 'results/final'
    final.mkdir(parents=True,exist_ok=True)
    manifest = dict(selected=selected,seeds=seeds,jobs=jobs)
    path = final / 'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest, 'Frozen final manifest differs'
    else:
        path.write_text(json.dumps(manifest,indent=2,allow_nan=False))
    launch(jobs)
    from exp6.report import report
    report()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=list(FINAL_SEEDS))
    p.add_argument('--gpus',choices=['1,2,3'],default='1,2,3')
    a = p.parse_args()
    run(a.selected,a.seeds)


if __name__ == '__main__':
    main()
