"""One trial, one visible GPU, ordinary minibatch backpropagation."""
from exp4.runtime import ROOT, EXP, output_path
import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import yaml

from exp4.config import METHODS, load_config
from exp4.bandinvmf import build_matrices, materialize
from exp4.model import pretrained_vit, initialization_digest
from exp4.privacy import calibrate, full_temporal_sensitivity, epsilon_from_mu
from exp4.mechanism import TemporalGaussianNoise, ZeroNoise
from exp4.optimizer_uc import UCAdam


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
    return (
        transforms.Compose([transforms.RandomResizedCrop(224, interpolation=interpolation),
                            transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]),
        transforms.Compose([transforms.Resize(int(224 / cfg['crop_pct']), interpolation=interpolation),
                            transforms.CenterCrop(224), transforms.ToTensor(), normalize]),
    )


def backward_microbatch(model, inputs, targets, logical_batch_size):
    """Sum divided by logical size: four 250-example batches give mean g_t."""
    loss = torch.nn.functional.cross_entropy(model(inputs), targets, reduction='sum')
    (loss / logical_batch_size).backward()
    return float(loss.detach())


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss, correct, count = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss += float(torch.nn.functional.cross_entropy(logits, targets, reduction='sum'))
        correct += int((logits.argmax(1) == targets).sum())
        count += targets.numel()
    return loss / count, correct / count


def run(args):
    started = time.monotonic()
    assert Path(sys.prefix).name == 'curve', 'Run in conda environment curve'
    visible = os.environ['CUDA_VISIBLE_DEVICES']
    assert visible in ('0', '2', '3'), 'Expose exactly one of physical GPUs 0,2,3'
    assert torch.cuda.device_count() == 1
    config = load_config(args.config)
    if args.seed is not None:
        config['seed'] = args.seed
    for name in ('lr', 'update_clip_norm'):
        value = getattr(args, name)
        if value is not None:
            assert value > 0
            config['optimizer'][name] = value
    result_dir = output_path(args.result_dir or EXP / 'results' /
                             ('smoke' if args.smoke else 'trials') / args.method /
                             f"seed_{config['seed']}")
    result_dir.mkdir(parents=True, exist_ok=True)
    assert {p.name for p in result_dir.iterdir()} <= {'train.log'}, 'Use a fresh trial directory'
    seed_all(config['seed'])
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    model, pretrained_cfg = pretrained_vit()
    init_digest = initialization_digest(model)
    head_digest = initialization_digest(model.head)
    model = model.to(device)
    coefficients, strategy, workload = build_matrices(
        config['total_steps'], config['bandinvmf_num_bands'],
        identity=args.method == 'dp-adam-iid-uc')
    opt = config['optimizer']
    if args.scale_probe:
        assert not args.smoke and args.method == 'dp-adam-iid-uc'
        assert config['seed'] == 20261001 and opt['lr'] == 1e-3
        opt['update_clip_norm'] = None
        privacy = dict(accountant='nonprivate_scale_probe', innovation_std=0.0,
                       target_epsilon=None, delta=None, total_steps=250,
                       privacy_applied=False, sampling_amplification=False)
    else:
        privacy = calibrate(strategy, opt['update_clip_norm'], **{
            k: config['privacy'][k] for k in ('epsilon', 'delta')})
    epochs = 1 if args.smoke else config['epochs']
    steps_per_epoch = 2 if args.smoke else config['dataset_size'] // config['logical_batch_size']
    resolved = dict(config, method=args.method, smoke=args.smoke,
                    scale_probe=args.scale_probe, update_clipping_enabled=not args.scale_probe,
                    dp_noise_enabled=not args.scale_probe,
                    effective_epochs=epochs, effective_steps_per_epoch=steps_per_epoch,
                    effective_total_steps=epochs * steps_per_epoch,
                    visible_devices=visible, device_name=torch.cuda.get_device_name(),
                    initialization_sha256=init_digest, classifier_initialization_sha256=head_digest,
                    pretrained_cfg=pretrained_cfg, local_data_root=str(ROOT / 'data'),
                    download=False, fixed_epoch_order=True, dtype='float32',
                    privacy_calibration=privacy, noising_coefficients=coefficients.tolist(),
                    trainable_parameters=sum(p.numel() for p in model.parameters()),
                    test_examples=100 if args.smoke else 10000,
                    augmentation_rng_convention='seeded DataLoader + worker seeds, fixed permutation every epoch',
                    diagnostic_release='private research diagnostics; GDP claim covers noisy parameter trajectory only',
                    versions={p: version(p) for p in
                              ('torch', 'torchvision', 'jax', 'jax_privacy', 'numpy', 'scipy', 'timm', 'PyYAML')})
    (result_dir / 'config.yaml').write_text(yaml.safe_dump(resolved, sort_keys=False))
    np.savez(result_dir / 'matrices.npz', noising_coefficients=coefficients, strategy=strategy,
             workload_coefficients=workload, W=materialize(workload, config['total_steps']),
             D=materialize(coefficients, config['total_steps']),
             innovation_std=privacy['innovation_std'])

    train_transform, test_transform = image_transforms(pretrained_cfg)
    train = datasets.CIFAR100(ROOT / 'data', train=True, transform=train_transform, download=False)
    test = datasets.CIFAR100(ROOT / 'data', train=False, transform=test_transform, download=False)
    assert len(train) == 50000 and len(test) == 10000
    permutation = torch.randperm(len(train), generator=torch.Generator().manual_seed(config['seed']))
    np.save(result_dir / 'train_order.npy', permutation.numpy())
    order = permutation[:steps_per_epoch * config['logical_batch_size']] if args.smoke else permutation
    train_loader = DataLoader(Subset(train, order.tolist()),
                              batch_size=config['physical_batch_size'], shuffle=False,
                              num_workers=config['num_workers'], pin_memory=True,
                              multiprocessing_context='spawn',
                              generator=torch.Generator().manual_seed(config['seed']))
    test_loader = DataLoader(Subset(test, range(100)) if args.smoke else test,
                             batch_size=config['physical_batch_size'], shuffle=False,
                             num_workers=config['num_workers'], pin_memory=True,
                             multiprocessing_context='spawn',
                             generator=torch.Generator().manual_seed(config['seed'] + 2))
    noise = ZeroNoise(model.parameters(), config['total_steps']) if args.scale_probe else TemporalGaussianNoise(
        model.parameters(), coefficients, privacy['innovation_std'], config['total_steps'], config['seed'] + 1)
    optimizer = UCAdam(model.parameters(), noise=noise, **{
        k: opt[k] for k in ('lr', 'update_clip_norm', 'beta1', 'beta2', 'eps')})
    records = []
    step_fields = ['epoch', 'train_loss', 'step', 'raw_update_norm', 'clipped_update_norm',
                   'clip_scale', 'clip_indicator', 'update_rms', 'clipped_update_rms',
                   'noise_marginal_std', 'parameter_noise_std', 'signal_parameter_norm',
                   'noise_signal_rms_ratio', 'realized_noise_rms', 'realized_noise_signal_rms_ratio',
                   'adam_m_norm', 'vhat_min', 'vhat_max', 'vhat_mean', 'vhat_rms']
    epoch_fields = ['epoch', 'logical_steps', 'train_examples', 'train_loss', 'test_loss',
                    'test_top1', 'epoch_clip_fraction', 'gdp_mu', 'gdp_epsilon',
                    'noise_marginal_std', 'parameter_noise_std', 'augmentation_trace_sha256', 'seconds']
    print(json.dumps(dict(method=args.method, calibration=privacy,
                          coefficients=coefficients.tolist())), flush=True)
    with (result_dir / 'steps.csv').open('w', newline='') as sf, \
            (result_dir / 'metrics.csv').open('w', newline='') as ef:
        sw = csv.DictWriter(sf, fieldnames=step_fields)
        ew = csv.DictWriter(ef, fieldnames=epoch_fields)
        sw.writeheader()
        ew.writeheader()
        for epoch in range(epochs):
            epoch_start = time.monotonic()
            model.train()
            optimizer.zero_grad()
            loss_sum, logical_loss, clipped, count = 0.0, 0.0, 0, 0
            augmentation = hashlib.sha256()
            for micro, (inputs, targets) in enumerate(train_loader, start=1):
                if (micro - 1) % config['gradient_accumulation'] == 0:
                    augmentation.update(inputs[:4].contiguous().numpy().tobytes())
                inputs, targets = inputs.to(device), targets.to(device)
                loss = backward_microbatch(model, inputs, targets, config['logical_batch_size'])
                loss_sum += loss
                logical_loss += loss
                count += targets.numel()
                if micro % config['gradient_accumulation'] == 0:
                    diagnostic = optimizer.step()
                    nonfinite = {key: value for key, value in diagnostic.items()
                                 if value is not None and not np.isfinite(value)}
                    assert not nonfinite, f'Nonfinite diagnostics at step {optimizer.step_count}: {nonfinite}'
                    clipped += diagnostic['clip_indicator']
                    sw.writerow(dict(epoch=epoch + 1, train_loss=logical_loss / config['logical_batch_size'],
                                     **diagnostic))
                    sf.flush()
                    optimizer.zero_grad()
                    logical_loss = 0.0
            assert optimizer.step_count == (epoch + 1) * steps_per_epoch == noise.step_count
            assert count == steps_per_epoch * config['logical_batch_size']
            test_loss, top1 = evaluate(model, test_loader, device)
            mu = None if args.scale_probe else 2 * opt['update_clip_norm'] * full_temporal_sensitivity(
                strategy, steps=optimizer.step_count) / privacy['innovation_std']
            record = dict(epoch=epoch + 1, logical_steps=optimizer.step_count, train_examples=count,
                          train_loss=loss_sum / count, test_loss=test_loss, test_top1=top1,
                          epoch_clip_fraction=clipped / steps_per_epoch, gdp_mu=mu,
                          gdp_epsilon=None if args.scale_probe else epsilon_from_mu(mu, privacy['delta']),
                          noise_marginal_std=noise.marginal_std(optimizer.step_count - 1),
                          parameter_noise_std=opt['lr'] * noise.marginal_std(optimizer.step_count - 1),
                          augmentation_trace_sha256=augmentation.hexdigest(),
                          seconds=time.monotonic() - epoch_start)
            assert all(np.isfinite(record[k]) for k in ('train_loss', 'test_loss', 'test_top1'))
            records.append(record)
            ew.writerow(record)
            ef.flush()
            print(json.dumps(record), flush=True)
    assert optimizer.step_count == resolved['effective_total_steps']
    torch.save(dict(model=model.state_dict(), logical_steps=optimizer.step_count), result_dir / 'final.pt')
    summary = dict(method=args.method, status='completed', smoke=args.smoke, seed=config['seed'],
                   scale_probe=args.scale_probe,
                   lr=opt['lr'], update_clip_norm=opt['update_clip_norm'],
                   optimizer_steps=optimizer.step_count, noise_steps=0 if args.scale_probe else noise.step_count,
                   planned_total_steps=config['total_steps'], physical_batches=optimizer.step_count * 4,
                   initialization_sha256=init_digest, classifier_initialization_sha256=head_digest,
                   calibration=privacy, final=records[-1], epochs=records,
                   wall_seconds=time.monotonic() - started)
    (result_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(dict(status='completed', results=str(result_dir))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--config', type=Path, default=EXP / 'config.yaml')
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--update-clip-norm', type=float)
    parser.add_argument('--smoke', action='store_true', help='2 logical steps; 8 physical batches; test 100 examples')
    parser.add_argument('--scale-probe', action='store_true', help='Stage 0: full nonprivate raw Adam, no clip, no noise; lr=1e-3')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
