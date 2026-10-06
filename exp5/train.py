"""Single-GPU coordinate-normalized trials; resume at completed epoch boundaries."""
from exp5.runtime import ROOT, output_path
import argparse
import csv
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import Subset
from exp5.config import FIXED, METHODS, Trial, CHUNK_SIZE
from exp5.data import assets, loader, EpochData
from exp5.model import pretrained_vit, initialization_digest
from exp5.privacy import build, spent, materialize
from exp5.mechanism import BandInvMFNoise, LogicalSGDM, PerExample, clip_microbatch

DIAGNOSTICS = ('mean_raw_sample_norm', 'mean_transformed_sample_norm', 'transformed_norm_std',
               'clip_fraction', 'mean_clip_factor', 'raw_mean_gradient_norm', 'query_norm',
               'batch_coherence', 'noise_std', 'momentum_norm', 'update_norm', 'logical_step_seconds')


def write_json(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def sample_metrics(stats):
    n = stats['count']
    mean = stats['h_norm_sum'] / n
    return dict(mean_raw_sample_norm=stats['raw_norm_sum'] / n,
                mean_transformed_sample_norm=mean,
                transformed_norm_std=max(0., stats['h_norm_sq_sum'] / n - mean ** 2) ** .5,
                clip_fraction=stats['clipped'] / n, mean_clip_factor=stats['factor_sum'] / n)


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    loss, correct, count = 0., 0, 0
    for inputs, targets in batches:
        inputs, targets = inputs.cuda(), targets.cuda()
        logits = model(inputs)
        loss += float(torch.nn.functional.cross_entropy(logits, targets, reduction='sum'))
        correct += int((logits.argmax(1) == targets).sum())
        count += targets.numel()
    return loss / count, correct / count


def epoch_diagnostics(totals, steps):
    metrics = {k: float(np.mean([r[k] for r in steps])) for k in DIAGNOSTICS}
    metrics.update(sample_metrics(totals))
    return metrics


def write_csv(path, rows):
    with output_path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(trial, result_dir, smoke=False):
    started = time.monotonic()
    result_dir = output_path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    if os.environ['CUDA_VISIBLE_DEVICES'] not in ('0', '1', '2'):
        raise ValueError('Expose exactly one physical GPU from 0,1,2')
    seed_all(trial.seed)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    model, pretrained_cfg = pretrained_vit()
    initial = initialization_digest(model)
    model.cuda()
    assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())
    coefficients, strategy, workload, privacy = build(trial.method, trial.C)
    configuration = dict(fixed=FIXED, trial=trial.asdict(), trial_id=trial.identity,
                         smoke=smoke, pretrained=pretrained_cfg,
                         initialization_sha256=initial, privacy=privacy,
                         chunk_size=CHUNK_SIZE, visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                         device_name=torch.cuda.get_device_name(), download=False)
    config_path = result_dir / 'config.json'
    if config_path.exists():
        old = json.loads(config_path.read_text())
        assert old['trial_id'] == trial.identity and old['fixed'] == FIXED and old['smoke'] == smoke
    write_json(config_path, configuration)
    np.savez(result_dir / 'matrices.npz', noising_coefficients=coefficients,
             strategy=strategy, workload_coefficients=workload, workload=materialize(workload, 250))
    train, test = assets(pretrained_cfg)
    permutation = torch.randperm(len(train), generator=torch.Generator().manual_seed(trial.seed))
    np.save(result_dir / 'train_order.npy', permutation.numpy())
    order = permutation[:1000] if smoke else permutation
    noise = BandInvMFNoise(model.parameters(), coefficients, privacy['innovation_std'], 250, trial.seed + 1)
    logical = LogicalSGDM(model.parameters(), noise, trial.lr)
    per_example = PerExample(model)
    checkpoint_path = result_dir / 'resume.pt'
    step_rows, epoch_rows, start_epoch, prior_seconds = [], [], 0, 0.
    if checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        assert state['trial_id'] == trial.identity
        model.load_state_dict(state['model'])
        for m, saved in zip(logical.momentum, state['momentum']):
            m.copy_(saved)
        logical.steps = noise.step_count = state['steps']
        noise.history = state['noise_history']
        noise.generator.set_state(state['noise_rng'].cpu())
        step_rows, epoch_rows = state['step_rows'], state['epoch_rows']
        start_epoch, prior_seconds = len(epoch_rows), state['wall_seconds']
        del state
    common = trial.asdict()
    for epoch in range(start_epoch, 1 if smoke else 5):
        model.train()
        totals = dict(loss_sum=0., correct=0, count=0, clipped=0, raw_norm_sum=0.,
                      h_norm_sum=0., h_norm_sq_sum=0., factor_sum=0., q_norm_sum=0.)
        window = dict(totals)
        epoch_steps = []
        torch.cuda.synchronize()
        step_started = time.monotonic()
        for inputs, targets in loader(EpochData(train, order, trial.seed, epoch), trial.seed):
            model.zero_grad(set_to_none=True)
            stats, raw_sums = clip_microbatch(per_example, inputs.cuda(), targets.cuda(), trial.tau, trial.C)
            for key, value in stats.items():
                window[key] += value
                totals[key] += value
            metrics = logical.finish_microbatch(raw_sums, stats['q_norm_sum'], stats['count'])
            del raw_sums
            if metrics is not None:
                torch.cuda.synchronize()
                row = dict(common, epoch=epoch + 1, step=logical.steps,
                           train_loss=window['loss_sum'] / 1000, train_top1=window['correct'] / 1000,
                           **sample_metrics(window), **metrics,
                           logical_step_seconds=time.monotonic() - step_started,
                           peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                           peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                           **spent(strategy, privacy, logical.steps),
                           per_step_sensitivity=privacy['per_step_sensitivity'])
                assert all(np.isfinite(v) for v in row.values() if isinstance(v, (float, int)))
                step_rows.append(row)
                epoch_steps.append(row)
                write_csv(result_dir / 'steps.csv', step_rows)
                print(json.dumps(dict(event='logical_step', **row)), flush=True)
                window = {k: 0 for k in window}
                step_started = time.monotonic()
        assert logical.micro_steps == 0
        test_loss, test_top1 = evaluate(model, loader(Subset(test, range(100)) if smoke else test, trial.seed + 2))
        record = dict(common, epoch=epoch + 1, logical_steps=logical.steps,
                      train_loss=totals['loss_sum'] / totals['count'],
                      train_top1=totals['correct'] / totals['count'], test_loss=test_loss, test_top1=test_top1,
                      **epoch_diagnostics(totals, epoch_steps),
                      **spent(strategy, privacy, logical.steps), per_step_sensitivity=privacy['per_step_sensitivity'],
                      seconds=prior_seconds + time.monotonic() - started)
        assert all(np.isfinite(v) for v in record.values() if isinstance(v, (float, int)))
        epoch_rows.append(record)
        write_csv(result_dir / 'metrics.csv', epoch_rows)
        print(json.dumps(dict(event='epoch', **record)), flush=True)
        temporary = checkpoint_path.with_suffix('.tmp')
        torch.save(dict(trial_id=trial.identity, model=model.state_dict(), momentum=logical.momentum,
                        steps=logical.steps, noise_history=noise.history, noise_rng=noise.generator.get_state(),
                        step_rows=step_rows, epoch_rows=epoch_rows,
                        wall_seconds=prior_seconds + time.monotonic() - started), temporary)
        temporary.replace(checkpoint_path)
    assert noise.step_count == logical.steps == (1 if smoke else 250)
    write_csv(result_dir / 'steps.csv', step_rows)
    diagnostics = {k: float(np.mean([r[k] for r in step_rows])) for k in DIAGNOSTICS}
    torch.save(dict(model=model.state_dict(), logical_steps=logical.steps), result_dir / 'final.pt')
    summary = dict(status='completed', trial_id=trial.identity, fixed=FIXED, **common,
                   smoke=smoke, optimizer_steps=logical.steps, noise_steps=noise.step_count,
                   physical_batches=logical.steps * 10, initialization_sha256=initial,
                   privacy=privacy, epochs=epoch_rows, final=epoch_rows[-1], diagnostics=diagnostics,
                   peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                   peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                   wall_seconds=prior_seconds + time.monotonic() - started)
    write_json(result_dir / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--tau', type=float, required=True)
    p.add_argument('--C', type=float, default=1.)
    p.add_argument('--result-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    run(Trial(a.method, a.seed, a.lr, a.tau, a.C), a.result_dir, a.smoke)

if __name__ == '__main__':
    main()
