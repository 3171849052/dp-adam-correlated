"""Offline single-GPU trial; ten physical batches per private logical query."""
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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from opacus.grad_sample import GradSampleModuleFastGradientClipping
from exp5.config import FIXED, METHODS, Trial
from exp5.model import pretrained_vit, initialization_digest
from exp5.privacy import build, spent, materialize
from exp5.mechanism import BandInvMFNoise, LogicalSGDM, clip_microbatch


def write_json(path, value):
    output_path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def image_transforms(cfg):
    interpolation = transforms.InterpolationMode.BICUBIC
    normalize = transforms.Normalize(cfg['mean'], cfg['std'])
    train = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=interpolation),
                                transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    test = transforms.Compose([transforms.Resize(int(224 / cfg['crop_pct']), interpolation=interpolation),
                               transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    return train, test


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss, correct, count = 0., 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        logits = model(inputs)
        loss += float(torch.nn.functional.cross_entropy(logits, targets, reduction='sum'))
        correct += int((logits.argmax(1) == targets).sum())
        count += targets.numel()
    return loss / count, correct / count


def run(trial, result_dir, smoke=False):
    started = time.monotonic()
    result_dir = output_path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    if set(p.name for p in result_dir.iterdir()) - {'train.log'}:
        raise RuntimeError(f'Trial output already exists: {result_dir}')
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
    write_json(result_dir / 'config.json', dict(fixed=FIXED, trial=trial.asdict(),
               trial_id=trial.identity, smoke=smoke, pretrained=pretrained_cfg,
               initialization_sha256=initial, privacy=privacy,
               visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
               device_name=torch.cuda.get_device_name(), download=False))
    np.savez(result_dir / 'matrices.npz', noising_coefficients=coefficients,
             strategy=strategy, workload_coefficients=workload,
             workload=materialize(workload, 250))
    train_transform, test_transform = image_transforms(pretrained_cfg)
    train = datasets.CIFAR100(ROOT / 'data', train=True, download=False, transform=train_transform)
    test = datasets.CIFAR100(ROOT / 'data', train=False, download=False, transform=test_transform)
    assert len(train) == 50000 and len(test) == 10000
    permutation = torch.randperm(len(train), generator=torch.Generator().manual_seed(trial.seed))
    np.save(result_dir / 'train_order.npy', permutation.numpy())
    order = permutation[:1000] if smoke else permutation
    train_loader = DataLoader(Subset(train, order.tolist()), batch_size=100, shuffle=False,
                              num_workers=2, pin_memory=True, multiprocessing_context='spawn',
                              persistent_workers=True,
                              generator=torch.Generator().manual_seed(trial.seed))
    test_loader = DataLoader(Subset(test, range(100)) if smoke else test, batch_size=100,
                             shuffle=False, num_workers=2, pin_memory=True,
                             multiprocessing_context='spawn', persistent_workers=True,
                             generator=torch.Generator().manual_seed(trial.seed + 2))
    model = GradSampleModuleFastGradientClipping(model, loss_reduction='sum',
                max_grad_norm=trial.C, use_ghost_clipping=True)
    noise = BandInvMFNoise(model.parameters(), coefficients, privacy['innovation_std'], 250, trial.seed + 1)
    logical = LogicalSGDM(model.parameters(), noise, trial.lr)
    step_rows, epoch_rows = [], []
    common = dict(method=trial.method, seed=trial.seed, lr=trial.lr, C=trial.C)
    step_fields = list(common) + ['epoch', 'step', 'train_loss', 'train_top1', 'clip_fraction',
                   'mean_unclipped_norm', 'mean_clip_factor', 'query_norm', 'noise_std',
                   'momentum_norm', 'update_norm', 'epsilon', 'delta', 'mu', 'sensitivity']
    with (result_dir / 'steps.csv').open('w', newline='') as sf, (result_dir / 'metrics.csv').open('w', newline='') as ef:
        sw = csv.DictWriter(sf, fieldnames=step_fields)
        sw.writeheader()
        ew = None
        for epoch in range(1 if smoke else 5):
            model.train()
            totals = dict(loss_sum=0., clipped=0, norm_sum=0., factor_sum=0., count=0, correct=0)
            window = dict(totals)
            epoch_steps = []
            for inputs, targets in train_loader:
                model.zero_grad(set_to_none=True)
                stats = clip_microbatch(model, inputs.cuda(), targets.cuda(), trial.C)
                for key, value in stats.items():
                    window[key] += value
                    totals[key] += value
                metrics = logical.finish_microbatch()
                if metrics is not None:
                    row = dict(common, epoch=epoch + 1, step=logical.steps,
                               train_loss=window['loss_sum'] / 1000, train_top1=window['correct'] / 1000,
                               clip_fraction=window['clipped'] / 1000,
                               mean_unclipped_norm=window['norm_sum'] / 1000,
                               mean_clip_factor=window['factor_sum'] / 1000, **metrics,
                               **spent(strategy, privacy, logical.steps), sensitivity=privacy['sensitivity'])
                    assert all(np.isfinite(v) for v in row.values() if isinstance(v, (float, int)))
                    sw.writerow(row)
                    sf.flush()
                    step_rows.append(row)
                    epoch_steps.append(row)
                    window = {k: 0 for k in window}
            assert logical.micro_steps == 0
            test_loss, test_top1 = evaluate(model, test_loader)
            record = dict(common, epoch=epoch + 1, logical_steps=logical.steps,
                          train_loss=totals['loss_sum'] / totals['count'],
                          train_top1=totals['correct'] / totals['count'],
                          test_loss=test_loss, test_top1=test_top1,
                          clip_fraction=totals['clipped'] / totals['count'],
                          mean_unclipped_norm=totals['norm_sum'] / totals['count'],
                          mean_clip_factor=totals['factor_sum'] / totals['count'],
                          **{k: float(np.mean([r[k] for r in epoch_steps])) for k in
                             ('query_norm', 'noise_std', 'momentum_norm', 'update_norm')},
                          **spent(strategy, privacy, logical.steps), sensitivity=privacy['sensitivity'],
                          seconds=time.monotonic() - started)
            assert all(np.isfinite(v) for v in record.values() if isinstance(v, (float, int)))
            if ew is None:
                ew = csv.DictWriter(ef, fieldnames=list(record))
                ew.writeheader()
            ew.writerow(record)
            ef.flush()
            epoch_rows.append(record)
            print(json.dumps(record), flush=True)
    assert noise.step_count == logical.steps == (1 if smoke else 250)
    diagnostics = {k: float(np.mean([r[k] for r in step_rows])) for k in
                   ('clip_fraction', 'mean_unclipped_norm', 'mean_clip_factor',
                    'query_norm', 'noise_std', 'momentum_norm', 'update_norm')}
    torch.save(dict(model=model._module.state_dict(), logical_steps=logical.steps), result_dir / 'final.pt')
    summary = dict(status='completed', trial_id=trial.identity, fixed=FIXED, **common,
                   smoke=smoke, optimizer_steps=logical.steps, noise_steps=noise.step_count,
                   physical_batches=logical.steps * 10, initialization_sha256=initial,
                   privacy=privacy, epochs=epoch_rows, final=epoch_rows[-1], diagnostics=diagnostics,
                   wall_seconds=time.monotonic() - started)
    write_json(result_dir / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--lr', type=float, required=True)
    p.add_argument('--C', type=float, required=True)
    p.add_argument('--result-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    run(Trial(a.method, a.seed, a.lr, a.C), a.result_dir, a.smoke)


if __name__ == '__main__':
    main()
