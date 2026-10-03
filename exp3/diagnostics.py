"""Fixed random JVP probes of the actual FP32 Muon map, outside the mechanism."""
import csv
import hashlib
import json
import numpy as np
import torch
from exp3.optimizer import muon_map

LAYERS = ('blocks.0.attn.qkv.weight', 'blocks.5.attn.proj.weight', 'blocks.11.mlp.2.weight')


def statistics(values):
    a = np.asarray(values, dtype=float)
    mean, std = float(a.mean()), float(a.std())
    return dict(mean=mean, std=std, cv=std / mean if mean else 0.,
                p10=float(np.quantile(a, .1)), p50=float(np.quantile(a, .5)),
                p90=float(np.quantile(a, .9)))


def probe_gain(h, probe, geometry):
    tangent = geometry.inverse(probe)
    _, derivative = torch.func.jvp(muon_map, (h.detach(),), (tangent,))
    return float(derivative.norm() / probe.norm())


class MuonDiagnostics:
    def __init__(self, optimizer, interval=25, probes=4, layers=LAYERS):
        self.optimizer, self.interval = optimizer, interval
        self.layers, self.rows, self.probes = layers, [], {}
        for name in layers:
            p = optimizer.muon[name]
            seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
            rng = torch.Generator(device=p.device).manual_seed(seed)
            self.probes[name] = [torch.randn(p.shape, generator=rng, device=p.device)
                                 for _ in range(probes)]

    def record(self, step, geometries):
        if step != 1 and step % self.interval:
            return
        for name in self.layers:
            p = self.optimizer.muon[name]
            h = self.optimizer.state[p]['pre_ns']
            gains = [probe_gain(h, e, geometries[p]) for e in self.probes[name]]
            self.rows.append(dict(step=step, layer=name, **statistics(gains)))

    def save(self, directory):
        with (directory / 'diagnostics.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'layer', 'mean', 'std', 'cv', 'p10', 'p50', 'p90'])
            writer.writeheader()
            writer.writerows(self.rows)
        by_step = {}
        temporal = {}
        for row in self.rows:
            by_step.setdefault(row['step'], []).append(row['mean'])
            temporal.setdefault(row['layer'], []).append(row['mean'])
        summary = dict(interval=self.interval, probe_count=len(next(iter(self.probes.values()))),
                       layers=list(self.layers), records=self.rows,
                       layer_gain_cv={str(step): statistics(v)['cv'] for step, v in by_step.items()},
                       temporal_gain_cv={name: statistics(v)['cv'] for name, v in temporal.items()})
        (directory / 'diagnostics.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        return summary
