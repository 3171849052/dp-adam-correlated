"""Paired, fixed-signal replay of complete pre-LR optimizer updates."""
import argparse
import csv
import hashlib
import io
import json

import numpy as np
import torch
from exp3.bandinvmf import build_matrices, materialize
from exp3.optimizer import muon_map
from exp3b import offline_runtime
from exp3b.spec import output_path, write_json
from exp3b.support import expected_names

offline_runtime()
REPLAY_SEEDS = tuple(range(20261101, 20261109))
FIELDS = ('step', 'method', 'rmse_mf_mean', 'rmse_mf_std', 'rmse_iid_mean',
          'rmse_iid_std', 'cancellation_mean', 'cancellation_std', 'relative_to_momentum',
          'instant_rmse_mf_mean', 'instant_rmse_mf_std', 'instant_rmse_iid_mean',
          'instant_rmse_iid_std', 'parameter_rmse_mf_mean', 'parameter_rmse_mf_std',
          'parameter_rmse_iid_mean', 'parameter_rmse_iid_std')


def write_csv(path, rows, fields):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def momentum_theory(steps=250):
    coefficients, _, workload = build_matrices('momentum_bandinvmf', 250, 4, .9)
    d = materialize(coefficients, steps)
    w = materialize(workload[:steps], steps)
    h = materialize(.9 ** np.arange(steps), steps)
    mf = np.linalg.norm(w @ d, axis=1)
    iid = np.linalg.norm(w, axis=1)
    instant_mf = np.linalg.norm(h @ d, axis=1)
    instant_iid = np.linalg.norm(h, axis=1)
    rows = [dict(step=i + 1, method='MF-Momentum', rmse_mf_mean=float(mf[i]),
                 rmse_mf_std=0., rmse_iid_mean=float(iid[i]), rmse_iid_std=0.,
                 cancellation_mean=float(mf[i] / iid[i]), cancellation_std=0.,
                 relative_to_momentum=1., instant_rmse_mf_mean=float(instant_mf[i]),
                 instant_rmse_mf_std=0., instant_rmse_iid_mean=float(instant_iid[i]),
                 instant_rmse_iid_std=0., parameter_rmse_mf_mean=float(mf[i]),
                 parameter_rmse_mf_std=0., parameter_rmse_iid_mean=float(iid[i]),
                 parameter_rmse_iid_std=0.) for i in range(steps)]
    return rows


class AdamState:
    def __init__(self, shapes, device):
        self.m = {n: torch.zeros(shape, device=device) for n, shape in shapes.items()}
        self.v = {n: torch.zeros(shape, device=device) for n, shape in shapes.items()}
        self.step = 0

    @torch.no_grad()
    def update(self, signals):
        self.step += 1
        updates = {}
        for name, x in signals.items():
            self.m[name].lerp_(x, .1)
            self.v[name].lerp_(x.square(), .001)
            m = self.m[name] / (1 - .9 ** self.step)
            v = self.v[name] / (1 - .999 ** self.step)
            updates[name] = m / (v.sqrt() + 1e-8)
        return updates


class MuonState:
    beta = .95
    nesterov = True
    ns_steps = 5

    def __init__(self, shapes, device):
        self.buffers = {n: torch.zeros(shape, device=device) for n, shape in shapes.items()}

    @torch.no_grad()
    def update(self, signals):
        updates = {}
        for name, g in signals.items():
            self.buffers[name].lerp_(g, 1 - self.beta)
            h = torch.lerp(g, self.buffers[name], self.beta)
            updates[name] = muon_map(h) * max(1., h.shape[0] / h.shape[1]) ** .5
        return updates


class PairedNoise:
    """One Gaussian draw z feeds BOTH IID and the finite-impulse MF filter."""
    def __init__(self, shapes, coefficients, sigma, seed, device):
        self.shapes, self.coefficients, self.sigma = shapes, coefficients, sigma
        self.device = device
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.history = []
        self.innovation_draws = 0

    @torch.no_grad()
    def next(self):
        z = {n: torch.randn(shape, device=self.device, generator=self.generator) * self.sigma
             for n, shape in self.shapes.items()}
        self.innovation_draws += 1
        mf = {n: value * float(self.coefficients[0]) for n, value in z.items()}
        for lag, previous in enumerate(self.history, 1):
            for n in z:
                mf[n].add_(previous[n], alpha=float(self.coefficients[lag]))
        self.history.insert(0, z)
        del self.history[len(self.coefficients) - 1:]
        return mf, z


def replay_inputs(tensors, mf_noise, iid_noise, scaled):
    clean, mf, iid = {}, {}, {}
    for name, saved in tensors.items():
        if scaled:
            s, q = saved['scale'], saved['q']
            assert torch.all(s > 0)
            g = q / s
            assert torch.equal(g, saved['g']), 'Clean must be the realized scaled-clipped signal'
            # Scale inversion on the noise path only. Clean never applies a new scale.
            mf[name] = g + mf_noise[name] / s
            iid[name] = g + iid_noise[name] / s
        else:
            g = saved['g']
            mf[name], iid[name] = g + mf_noise[name], g + iid_noise[name]
        clean[name] = g
    return clean, mf, iid


def load_signal(directory, entry, step, shapes, device):
    data = (directory / entry['file']).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry['sha256'], 'Source signal changed'
    payload = torch.load(io.BytesIO(data), map_location=device, weights_only=True)
    assert payload['step'] == step
    tensors = payload['tensors']
    assert tuple(tensors) == expected_names()
    for name, saved in tensors.items():
        assert tuple(saved['g'].shape) == tuple(shapes[name])
        assert all(torch.isfinite(value).all() for value in saved.values())
    return tensors


@torch.no_grad()
def replay_one(manifest, loader, seed, device):
    shapes = manifest['shapes']
    assert tuple(shapes) == expected_names() and manifest['support'] == list(expected_names())
    dimension = sum(int(np.prod(shape)) for shape in shapes.values())
    assert dimension == manifest['dimension']
    scaled = manifest['method'] == 'momentum_scale'
    muon = manifest['method'] == 'mf_muon_standard'
    state_type = MuonState if muon else AdamState
    clean, mf, iid = (state_type(shapes, device) for _ in range(3))
    if not muon:
        assert all(len({branch.m[n].data_ptr() for branch in (clean, mf, iid)}) == 3
                   and len({branch.v[n].data_ptr() for branch in (clean, mf, iid)}) == 3 for n in shapes)
    noise = PairedNoise(shapes, manifest['coefficients'], manifest['innovation_std_gradient'], seed, device)
    cumulative = [{n: torch.zeros(shape, device=device, dtype=torch.float64)
                   for n, shape in shapes.items()} for _ in range(2)]
    records = []
    for step in range(1, manifest['steps'] + 1):
        tensors = loader(step)
        mf_noise, iid_noise = noise.next()
        inputs = replay_inputs(tensors, mf_noise, iid_noise, scaled)
        updates = [state.update(x) for state, x in zip((clean, mf, iid), inputs)]
        values = []
        for branch, noisy in enumerate(updates[1:]):
            instant_sq = torch.zeros((), device=device, dtype=torch.float64)
            cumulative_sq = torch.zeros_like(instant_sq)
            for name in shapes:
                difference = (noisy[name] - updates[0][name]).double()
                cumulative[branch][name].add_(difference)
                instant_sq.add_(difference.square().sum())
                cumulative_sq.add_(cumulative[branch][name].square().sum())
            values.extend([float((cumulative_sq / dimension).sqrt()),
                           float((instant_sq / dimension).sqrt())])
        rmse_mf, instant_mf, rmse_iid, instant_iid = values
        assert rmse_iid > 0
        row = dict(step=step, replay_seed=seed, rmse_mf=rmse_mf, rmse_iid=rmse_iid,
                   instant_rmse_mf=instant_mf, instant_rmse_iid=instant_iid,
                   cancellation=rmse_mf / rmse_iid,
                   parameter_rmse_mf=rmse_mf * manifest['actual_lr'],
                   parameter_rmse_iid=rmse_iid * manifest['actual_lr'])
        assert all(np.isfinite(value) for value in row.values())
        records.append(row)
    assert noise.innovation_draws == manifest['steps']
    return records


def aggregate(samples, method, theory):
    assert len(samples) >= 2
    steps = len(theory)
    assert all([row['step'] for row in sample] == list(range(1, steps + 1)) for sample in samples)
    rows = []
    for index in range(steps):
        row = dict(step=index + 1, method=method)
        for metric in ('rmse_mf', 'rmse_iid', 'cancellation', 'instant_rmse_mf',
                       'instant_rmse_iid', 'parameter_rmse_mf', 'parameter_rmse_iid'):
            values = [sample[index][metric] for sample in samples]
            row[metric + '_mean'] = float(np.mean(values))
            row[metric + '_std'] = float(np.std(values, ddof=1))
        row['relative_to_momentum'] = row['cancellation_mean'] / theory[index]['cancellation_mean']
        rows.append(row)
    return rows


def run_replay(source, result, device='cuda:0', seeds=REPLAY_SEEDS, smoke=False):
    source, result = output_path(source), output_path(result)
    manifest = json.loads((source / 'manifest.json').read_text())
    assert manifest['status'] == 'completed' and manifest['steps'] == (2 if smoke else 250)
    assert manifest['smoke'] == smoke
    if not smoke:
        expected, _, _ = build_matrices('momentum_bandinvmf', 250, 4, .9)
        np.testing.assert_array_equal(manifest['coefficients'], expected)
        assert manifest['initial_optimizer_state'] == 'zero'
        assert manifest['logical_batch_size'] == 1000
    assert len(seeds) >= 2 and len(set(seeds)) == len(seeds)
    labels = {'momentum_standard': 'MF-Adam', 'momentum_scale': 'MF-Adam-Scale',
              'mf_muon_standard': 'MF-Muon'}
    theory = momentum_theory(manifest['steps'])
    samples = []
    for seed in seeds:
        samples.append(replay_one(manifest, lambda step: load_signal(
            source, manifest['files'][step - 1], step, manifest['shapes'], device), seed, device))
    rows = aggregate(samples, labels[manifest['method']], theory)
    write_csv(result.with_suffix('.csv'), rows, FIELDS)
    flat = [row for sample in samples for row in sample]
    write_csv(result.parent / (result.name + '_paired_samples.csv'), flat, tuple(flat[0]))
    write_json(result.with_suffix('.json'), dict(
        status='completed', source=str(source), source_manifest=manifest,
        replay_seeds=list(seeds), steps=manifest['steps'], support_count=48,
        support=manifest['support'], dimension=manifest['dimension'],
        paired_noise='MF = D z, IID = z, exact same Gaussian tensors; same MF innovation scale',
        metric='pre-LR complete optimizer update; cumulative sum of noisy minus clean updates',
        std='sample standard deviation across paired replay seeds, ddof=1',
        cancellation='mean of paired RMSE_MF / RMSE_IID ratios',
        source_trajectory_modified=False, forward_backward_in_replay=False,
        mf_workload_beta=.9, muon_nesterov_beta=.95 if manifest['method'] == 'mf_muon_standard' else None,
        final=rows[-1]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--result', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.device.startswith('cuda'):
        import os
        assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0', '1', '2', '3')
        assert torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    run_replay(args.source, args.result, args.device, smoke=args.smoke)


if __name__ == '__main__':
    main()
