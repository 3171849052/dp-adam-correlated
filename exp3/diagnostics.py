"""Actual update-space JVP and saved frozen-state MF diagnostic, analysis only."""
import csv
import hashlib
import json
import math
import numpy as np
import torch
from exp3.optimizer import muon_map
from exp3.geometry import snapshot_geometry, snapshot_inverse

LAYERS = ('blocks.0.attn.qkv.weight', 'blocks.5.attn.proj.weight', 'blocks.11.mlp.2.weight')
TEMPORAL_PROBE_SEED = 31415926


def statistics(values):
    a = np.asarray(values, dtype=float)
    mean, std = float(a.mean()), float(a.std())
    return dict(mean=mean, std=std, cv=std / mean if mean else 0.,
                p10=float(np.quantile(a, .1)), p50=float(np.quantile(a, .5)),
                p90=float(np.quantile(a, .9)))


def shape_factor(shape):
    return max(1., shape[0] / shape[1]) ** .5


def probe_gain(h, probe, geometry):
    _, derivative = torch.func.jvp(muon_map, (h.detach(),), (geometry.inverse(probe),))
    return float(derivative.norm() / probe.norm())


def gain_metrics(phi_gains, shape, innovation_std_sum, logical_batch_size):
    update = [shape_factor(shape) * g for g in phi_gains]
    result = {}
    for label, values in (('phi_gain', phi_gains), ('update_gain', update)):
        result.update({f'{label}_{k}': v for k, v in statistics(values).items()})
    weighted = None if innovation_std_sum is None else statistics(
        [innovation_std_sum / logical_batch_size * g for g in update])
    result.update({f'noise_weighted_update_gain_{k}': None if weighted is None else weighted[k]
                   for k in statistics([0])})
    return result


def layer_seed(name, base=0):
    return (int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little') + base) % 2**63


def temporal_probes(name, shape, steps, probes, seed=TEMPORAL_PROBE_SEED):
    """CPU-generated sequences are method/GPU independent and never use training RNG."""
    rng = torch.Generator().manual_seed(layer_seed(name, seed))
    return torch.randn((probes, steps, *shape), generator=rng)


def nesterov_tangent(gradient_perturbation, momentum_perturbation, beta=.95):
    momentum = torch.lerp(momentum_perturbation, gradient_perturbation, 1 - beta)
    return momentum, torch.lerp(gradient_perturbation, momentum, beta)


def frozen_trajectory_diagnostic(trajectory, d, innovation_std_sum, logical_size, muon_lr,
                                 device, probes=2, seed=TEMPORAL_PROBE_SEED):
    """Frozen-state first-order diagnostic, not exact nonlinear training dynamics.

    Actual Nesterov .95 propagates gradient perturbations; MF D was designed
    with ordinary momentum .9. Each final RMS includes LR and sigma/batch.
    """
    reports, final_all, prefix_all, hashes = {}, [], [], {}
    for name, frames in trajectory.items():
        steps = len(frames)
        shape = tuple(frames[0]['h'].shape)
        z = temporal_probes(name, shape, steps, probes, seed)
        hashes[name] = hashlib.sha256(z.numpy().tobytes()).hexdigest()
        correlated = torch.einsum('ts,psmn->ptmn', torch.tensor(d[:steps, :steps], dtype=z.dtype), z)
        correlated.mul_(innovation_std_sum / logical_size)
        finals, prefixes = [], []
        for probe in range(probes):
            momentum = torch.zeros(shape, device=device)
            cumulative = torch.zeros_like(momentum)
            per_prefix = []
            for t, frame in enumerate(frames):
                e = snapshot_inverse(frame['geometry'], correlated[probe, t].to(device))
                momentum, dh = nesterov_tangent(e, momentum, .95)
                h = frame['h'].to(device)
                _, derivative = torch.func.jvp(muon_map, (h,), (dh,))
                cumulative.add_(derivative, alpha=-muon_lr * shape_factor(shape))
                per_prefix.append(float(cumulative.square().mean().sqrt()))
            finals.append(per_prefix[-1])
            prefixes.append(float(np.mean(per_prefix)))
        reports[name] = dict(final_cumulative_rmse=statistics(finals), mean_prefix_rmse=statistics(prefixes),
                             per_probe_final_cumulative_rmse=finals, per_probe_mean_prefix_rmse=prefixes)
        final_all.extend(finals)
        prefix_all.extend(prefixes)
    return dict(name='frozen_trajectory_muon_mf', interpretation='frozen-state first-order diagnostic; not exact nonlinear training dynamics',
                actual_dynamics='EMA Nesterov beta=.95', mf_design_workload='ordinary momentum beta=.9',
                steps=len(next(iter(trajectory.values()))), layers=list(trajectory), probes=probes, probe_seed=seed,
                temporal_probe_sha256=hashes, innovation_std_sum=innovation_std_sum, logical_batch_size=logical_size,
                muon_lr=muon_lr, per_layer=reports,
                aggregate=dict(final_cumulative_rmse=statistics(final_all), mean_prefix_rmse=statistics(prefix_all)))


class MuonDiagnostics:
    def __init__(self, optimizer, interval=25, probes=4, layers=LAYERS,
                 innovation_std_sum=None, logical_batch_size=1000, save_trajectory=False):
        self.optimizer, self.interval = optimizer, interval
        self.layers, self.rows, self.probes = layers, [], {}
        self.sigma, self.logical_size = innovation_std_sum, logical_batch_size
        self.trajectory = {name: [] for name in layers} if save_trajectory else {}
        for name in layers:
            p = optimizer.muon[name]
            rng = torch.Generator(device=p.device).manual_seed(layer_seed(name))
            self.probes[name] = [torch.randn(p.shape, generator=rng, device=p.device) for _ in range(probes)]

    def record(self, step, geometries):
        for name in self.trajectory:
            p = self.optimizer.muon[name]
            self.trajectory[name].append(dict(step=step, h=self.optimizer.state[p]['pre_ns'].detach().cpu().clone(),
                                              geometry=snapshot_geometry(geometries[p])))
        if step != 1 and step % self.interval:
            return
        for name in self.layers:
            p = self.optimizer.muon[name]
            h = self.optimizer.state[p]['pre_ns']
            gains = [probe_gain(h, e, geometries[p]) for e in self.probes[name]]
            self.rows.append(dict(step=step, layer=name, shape_factor=shape_factor(p.shape),
                                  **gain_metrics(gains, p.shape, self.sigma, self.logical_size)))

    def save(self, directory):
        with (directory / 'diagnostics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        by_step, temporal = {}, {}
        for row in self.rows:
            by_step.setdefault(row['step'], []).append(row['update_gain_mean'])
            temporal.setdefault(row['layer'], []).append(row['update_gain_mean'])
        summary = dict(interval=self.interval, probe_count=len(next(iter(self.probes.values()))), layers=list(self.layers),
                       records=self.rows, layer_update_gain_cv={str(s): statistics(v)['cv'] for s,v in by_step.items()},
                       temporal_update_gain_cv={n: statistics(v)['cv'] for n,v in temporal.items()})
        (directory / 'diagnostics.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        torch.save(self.trajectory, directory / 'muon_trajectory.pt')
        return summary


class UpdateStatistics:
    def __init__(self):
        self.rows = []

    @torch.no_grad()
    def record(self, optimizer, before):
        for label, parameters in (('muon', list(optimizer.muon.values())), ('adam', list(optimizer.adam.values()))):
            grad = float(torch.stack([p.grad.square().sum() for p in parameters]).sum().sqrt())
            update = float(torch.stack([(p - before[p]).square().sum() for p in parameters]).sum().sqrt())
            weight = float(torch.stack([before[p].square().sum() for p in parameters]).sum().sqrt())
            elements = sum(p.numel() for p in parameters)
            self.rows.append(dict(step=optimizer.step_count, group=label, gradient_norm=grad, update_norm=update,
                                  update_rms=update/math.sqrt(elements), relative_update_norm=update/weight))

    def save(self, directory):
        with (directory / 'update_statistics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        summary = {label: {key: statistics([r[key] for r in self.rows if r['group'] == label])
                          for key in ('gradient_norm', 'update_norm', 'update_rms', 'relative_update_norm')}
                   for label in ('muon', 'adam')}
        (directory / 'update_statistics.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        return summary
