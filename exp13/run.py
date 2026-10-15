"""Explicit launcher. --smoke is small; the default command runs the full experiment."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from exp13 import BASE, ROOT
from exp13.io import file_hash, local_path, save_json


def protocol(smoke, k, holdouts):
    from exp13.math import SEEDS, HORIZONS, METHODS
    return dict(kind='smoke' if smoke else 'full',
                seeds=list(SEEDS[:4] if smoke else SEEDS),
                horizons=[10] if smoke else list(HORIZONS), coordinates=16 if smoke else k,
                holdouts_per_fit=holdouts, methods=list(METHODS),
                beta1=.9, beta2=.999, eps=1e-8, epsilon=8., delta=1e-5,
                epochs=5, clip=1., logical_batch=1000, learning_rate=1.,
                normalization='mean over T and K; cumulative errors at all prefixes',
                optimizer=dict(method='Nelder-Mead', maxiter=1600, xatol=1e-9, fatol=1e-7,
                               max_inverse_abs=1e8,
                               starts=['Exp9 four-band cumulative-momentum BISR', 'IID']),
                dtype='float64', innovation_rng='NumPy PCG64, one fixed draw per seed/horizon',
                holdout_rng='fit_seed + 100000000 + holdout_index*1000000 (index starts at 0)',
                task_scope='pure-noise numerical simulation only',
                calibration='Exp9 GDP, add/remove zero-out, fixed 5 participations, no amplification')


def add_improvements(methods):
    for values in methods.values():
        for baseline in ('iid', 'exp9_momentum_bias'):
            for metric in ('J', 'J_real'):
                values[f'{metric}_improvement_vs_{baseline}_pct'] = 100 * (
                    1 - values[metric] / methods[baseline][metric])


def worker(output, gpu):
    # Parent sets CUDA_VISIBLE_DEVICES before Python imports torch.
    import numpy as np
    import torch
    from exp13.math import Evaluator, configure_torch, exp9_filter, innovations, materialize
    configure_torch()
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    manifest = json.loads((output / 'manifest.json').read_text())
    p = manifest['protocol']
    jobs = manifest['assignments'][str(gpu)]
    device = torch.device('cuda:0')
    original = {}
    for job in jobs:
        seed, n, k = job['seed'], job['T'], p['coordinates']
        started = time.monotonic()
        dest = output / f'T{n}' / f'seed{seed}'
        dest.mkdir(parents=True, exist_ok=False)
        print(f'GPU {gpu}: seed={seed} T={n} K={k} started', flush=True)
        xi, xi_hash = innovations(seed, n, k, device)
        np.save(dest / 'xi.npy', xi.cpu().numpy(), allow_pickle=False)
        evaluator = Evaluator(xi)
        if n not in original:
            original[n] = exp9_filter(n)
        d9, fit9 = original[n]
        fitted = {'iid': np.array([1., 0., 0., 0.]), 'exp9_momentum_bias': d9}
        fits = {'exp9_momentum_bias': dict(starts=fit9, seed_independent_baseline=True)}
        for mode, method in (('expected', 'expected_self_consistent'), ('actual', 'actual_self_consistent')):
            fit_start = time.monotonic()
            with (dest / f'{method}_trajectory.jsonl').open('w') as f:
                def trace(row):
                    f.write(json.dumps(row, allow_nan=False) + '\n')
                fitted[method], fits[method] = evaluator.fit(mode, trace=trace)
            fits[method]['seconds'] = time.monotonic() - fit_start
            print(f'GPU {gpu}: seed={seed} T={n} {method}: '
                  f'J_fit={fits[method]["objective_at_solution"]:.8g}, '
                  f'evaluations={fits[method]["evaluations"]}', flush=True)
        metrics, artifacts = {}, {}
        for method, d in fitted.items():
            values, arrays = evaluator.metrics(d)
            values['J_expected'] = evaluator.evaluate(d, 'expected')
            metrics[method] = values
            path = dest / f'{method}_matrices.npz'
            np.savez_compressed(path, d=d, D=materialize(d, n),
                                C=arrays['C'], Q=arrays['Q'].cpu().numpy(),
                                sigma=np.float64(arrays['sigma']))
            artifacts[path.name] = file_hash(path)
        add_improvements(metrics)
        heldout = []
        for i in range(p['holdouts_per_fit']):
            test_seed = seed + 100_000_000 + i*1_000_000
            test_xi, test_hash = innovations(test_seed, n, k, device)
            test = Evaluator(test_xi)
            values = {method: test.metrics(d)[0] for method, d in fitted.items()}
            add_improvements(values)
            heldout.append(dict(seed=test_seed, xi_sha256=test_hash, methods=values))
        artifacts['xi.npy'] = file_hash(dest / 'xi.npy')
        summary = dict(seed=seed, T=n, K=k, physical_gpu=gpu,
                       device_name=torch.cuda.get_device_name(0), xi_sha256=xi_hash,
                       fits=fits, in_sample=metrics, holdout=heldout, artifacts=artifacts,
                       elapsed_seconds=time.monotonic()-started,
                       peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(0))
        save_json(dest / 'summary.json', summary)
        print(f'GPU {gpu}: seed={seed} T={n} complete ({summary["elapsed_seconds"]:.1f}s)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs=4, default=[0, 1, 2, 3])
    parser.add_argument('--coordinates', type=int, default=4096)
    parser.add_argument('--holdout-seeds', type=int, default=3)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--plan', action='store_true', help='Print assignments without starting any work')
    parser.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        worker(local_path(args.output), args.worker)
        return
    assert set(args.gpus) == {0, 1, 2, 3}
    assert args.coordinates > 0 and args.holdout_seeds >= 2
    p = protocol(args.smoke, args.coordinates, args.holdout_seeds)
    assignments = {str(g): [] for g in args.gpus}
    # A whole seed stays on one GPU; no fitting across seeds or GPUs.
    for index, seed in enumerate(p['seeds']):
        for n in p['horizons']:
            assignments[str(args.gpus[index % 4])].append(dict(seed=seed, T=n))
    plan = dict(protocol=p, assignments=assignments)
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    output = local_path(args.output or BASE / 'results' / p['kind'])
    output.mkdir(parents=True, exist_ok=False)
    from exp13.assets import prepare
    prepare(output, p['coordinates'])
    sources = {str(f.relative_to(BASE)): file_hash(f)
               for f in sorted(BASE.rglob('*.py')) if 'runtime' not in f.parts}
    import numpy as np
    import scipy
    import torch
    plan.update(source_sha256=sources, assets_sha256=file_hash(output / 'assets.json'),
                versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                              torch=torch.__version__, cuda=torch.version.cuda),
                launch_command=sys.argv, output=str(output))
    save_json(output / 'manifest.json', plan)
    processes = []
    try:
        for gpu in args.gpus:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            log = (output / f'gpu{gpu}.log').open('w')
            process = subprocess.Popen([sys.executable, '-B', '-m', 'exp13.run',
                                        '--worker', str(gpu), '--output', str(output)],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            log.close()
            processes.append((gpu, process))
        while processes:
            for gpu, process in processes[:]:
                code = process.poll()
                if code is not None:
                    assert code == 0, f'GPU {gpu} worker exited {code}; see {output / f"gpu{gpu}.log"}'
                    processes.remove((gpu, process))
            time.sleep(.5)
    finally:
        # Expose any failure and stop remaining work; no retries or partial reuse.
        for _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, process in processes:
            process.wait()
    from exp13.report import build
    build(output)
    save_json(output / 'completion.json', dict(complete=True, kind=p['kind'],
              jobs=len(p['seeds'])*len(p['horizons']), fitted_matrices_per_job=2,
              source_sha256=sources))
    print(f'Completed {p["kind"]}: {output / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
