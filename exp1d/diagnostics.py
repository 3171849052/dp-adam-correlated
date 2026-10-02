"""Exact epoch quantiles; raw values live in RAM only and are never persisted."""
import csv
import numpy as np
import torch

QUANTILES = ('p10', 'p25', 'p50', 'p75', 'p90', 'p99')


def statistics(chunks, clip=None):
    values = np.concatenate(chunks)
    result = dict(zip(QUANTILES, map(float, np.quantile(values, [.1,.25,.5,.75,.9,.99]))))
    result['mean'] = float(values.mean(dtype=np.float64))
    if clip is not None:
        result['clip_fraction'] = float((values > clip).mean())
    return result


class EpochDiagnostics:
    def __init__(self, clip):
        self.clip = clip
        self.norms = {'unscaled_norm': [], 'scaled_norm': []}
        self.coordinates = {'sqrt_vhat': [], 'scale': []}

    def add_norms(self, unscaled, scaled):
        self.norms['unscaled_norm'].append(unscaled.cpu().numpy())
        self.norms['scaled_norm'].append(scaled.cpu().numpy())

    @torch.no_grad()
    def add_state(self, parameters):
        # Called exactly once at logical-step start, after freezing vhat_{t-1}.
        self.coordinates['sqrt_vhat'].append(torch.cat([p._logical_sqrt_vhat.flatten() for p in parameters]).cpu().numpy())
        self.coordinates['scale'].append(torch.cat([p._logical_scale.flatten() for p in parameters]).cpu().numpy())

    def aggregate(self):
        return ({key: statistics(values, self.clip if key == 'scaled_norm' else None)
                 for key, values in self.norms.items()},
                {key: statistics(values) for key, values in self.coordinates.items()})


def append_statistics(path, epoch, groups):
    fields = ['epoch', 'group', *QUANTILES, 'mean', 'clip_fraction']
    exists = path.exists()
    with path.open('a', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for group, stats in groups.items():
            writer.writerow(dict(epoch=epoch, group=group, **stats))
