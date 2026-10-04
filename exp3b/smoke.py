"""Two tiny GPU logical steps per mechanism; no expensive full experiment."""
import json
from exp3b import BASE
from exp3b.history import read_inputs
from exp3b.scheduler import run_trials
from exp3b.spec import ADAM, HYBRID, PARAMETERS, RunSpec, write_json


def main():
    inputs = read_inputs()
    specs = []
    for method in ADAM + HYBRID:
        values = ({k: v for k, v in inputs['adam'][method].items() if k != 'num_bands'}
                  if method in ADAM else {k: inputs['choices'][method][k] for k in PARAMETERS})
        specs.append(RunSpec(method=method, result_dir=f'exp3b/results/smoke/{method}', smoke=True,
            capture=method in ADAM + ('mf_muon_standard',), **values))
    summaries = run_trials(specs, [0, 1, 2, 3], 'smoke')
    observed_gpus = [json.loads((spec.directory / 'config.json').read_text())['visible_devices']
                     for spec in specs]
    assert observed_gpus == ['0', '1', '2', '3', '0', '1', '2']
    assert all(summary['optimizer_steps'] == 2 for summary in summaries)
    from exp3b.replay import run_replay
    from exp3b.report import build
    outputs = BASE / 'results/smoke/cancellation'
    import torch
    torch.set_num_threads(2)
    for method, name in (('momentum_standard', 'adam'), ('momentum_scale', 'adam_scale'), ('mf_muon_standard', 'muon')):
        spec = next(s for s in specs if s.method == method)
        run_replay(spec.directory / 'signals', outputs / name, device='cpu', smoke=True)
    groups = build(outputs, steps=2)
    write_json(BASE / 'results/smoke_verification.json', dict(status='passed',
        scope='tiny dim=8 image=32 model, local checkpoint read, real CIFAR100 read, two GPU logical steps per method',
        methods=[s.method for s in specs], gpus=[0, 1, 2, 3], max_concurrency=4,
        observed_gpu_each_method=observed_gpus,
        logical_steps_each=[s['optimizer_steps'] for s in summaries],
        capture_support=48, replay_seeds=8, paired_replay=True,
        cancellation_methods=list(groups), full_experiment_started=False))
    print(json.dumps(dict(status='passed', summary='exp3b/results/smoke_verification.json')))


if __name__ == '__main__':
    main()
