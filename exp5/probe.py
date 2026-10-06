"""Nonprivate, bounded-memory logarithmic histogram of nonzero per-example |g|."""
from exp5.runtime import EXP
import argparse
import json
import os
import numpy as np
import torch
from exp5.config import SEARCH_SEED, FIXED
from exp5.model import pretrained_vit, initialization_digest
from exp5.data import assets, loader, EpochData
from exp5.mechanism import PerExample
from exp5.train import seed_all, write_json

QUANTILES = (.1, .25, .5, .75, .9)


def histogram_quantiles(histogram, lower=-50., upper=10.):
    histogram = np.asarray(histogram, dtype=np.int64)
    cumulative = histogram.cumsum()
    if cumulative[-1] == 0:
        raise ValueError('Probe found no nonzero gradient coordinates')
    width = (upper - lower) / len(histogram)
    result = []
    for p in QUANTILES:
        rank = p * cumulative[-1]
        index = int(np.searchsorted(cumulative, rank))
        previous = cumulative[index-1] if index else 0
        fraction = (rank - previous) / histogram[index]
        result.append(float(10 ** (lower + (index + fraction) * width)))
    return result


def run(path=EXP / 'results/search/scale_probe.json', physical_batches=3):
    if path.exists():
        existing = json.loads(path.read_text())
        assert existing['fixed'] == FIXED and existing['seed'] == SEARCH_SEED
        assert existing['physical_batches'] == physical_batches
        return existing
    if os.environ['CUDA_VISIBLE_DEVICES'] not in ('0', '1', '2'):
        raise ValueError('Probe must expose exactly one GPU from 0,1,2')
    seed_all(SEARCH_SEED)
    torch.set_num_threads(2)
    model, cfg = pretrained_vit()
    initial = initialization_digest(model)
    model.cuda().train()
    train, _ = assets(cfg)
    order = torch.randperm(50000, generator=torch.Generator().manual_seed(SEARCH_SEED))[:100*physical_batches]
    per_example = PerExample(model)
    bins, lower, upper = 8192, -50., 10.
    histogram = torch.zeros(bins, device='cuda', dtype=torch.int64)
    total = 0
    for inputs, targets in loader(EpochData(train, order, SEARCH_SEED, 0), SEARCH_SEED):
        for gradients, losses, logits in per_example.chunks(inputs.cuda(), targets.cuda()):
            for g in gradients.values():
                values = g.abs().flatten()
                nonzero = values[values != 0]
                assert bool(torch.isfinite(nonzero).all())
                logs = nonzero.log10()
                assert bool(((logs >= lower) & (logs < upper)).all())
                indices = ((logs - lower) * (bins / (upper - lower))).long()
                histogram.add_(torch.bincount(indices, minlength=bins))
                total += values.numel()
                del values, nonzero, logs, indices
            del gradients, losses, logits, g
    hist = histogram.cpu().numpy()
    values = histogram_quantiles(hist, lower, upper)
    result = dict(fixed=FIXED, seed=SEARCH_SEED, physical_batches=physical_batches,
                  samples=100*physical_batches, initialization_sha256=initial, pretrained=cfg,
                  quantiles={f'p{int(p*100)}': v for p, v in zip(QUANTILES, values)},
                  tau_grid=sorted(set(values)), nonzero_coordinates=int(hist.sum()),
                  zero_coordinates=int(total-hist.sum()), approximation='streaming_log10_histogram',
                  histogram=dict(lower=lower, upper=upper, bins=bins, counts=hist.tolist(),
                                 multiplicative_bin_width=float(10**((upper-lower)/bins))),
                  nonprivate=True, included_in_dp_guarantee=False,
                  privacy_scope='DP calibration is conditional on fixed public tau/lr; data-dependent probe and test selection are not privatized',
                  gradients_saved=False)
    write_json(path, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'histogram'}), flush=True)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--physical-batches', type=int, default=3)
    run(physical_batches=p.parse_args().physical_batches)
