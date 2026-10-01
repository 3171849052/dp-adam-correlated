"""Run one method on the single GPU exposed by CUDA_VISIBLE_DEVICES."""
import argparse
import csv
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import yaml
from opacus.grad_sample import GradSampleModuleFastGradientClipping

from exp1.bandinvmf import BandInvMFNoise, build_matrices
from exp1.model import ViTTiny
from exp1.privacy import calibrate, epsilon_from_mu, fixed_epoch_sensitivity
from exp1.scale import LogicalBatch, ScaledGhostModule, clipped_microbatch

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('adam', 'dp_adam', 'dp_adam_bandinvmf_momentum', 'dp_adam_bandinvmf_scale')


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    assert config['dataset_size'] == 50000
    assert config['epochs'] == config['privacy']['k'] == 5
    assert config['logical_batch_size'] == 1000
    assert config['physical_batch_size'] == 50
    assert config['gradient_accumulation'] == 20
    assert config['logical_batch_size'] == config['physical_batch_size'] * config['gradient_accumulation']
    assert config['dataset_size'] // config['logical_batch_size'] == config['privacy']['b_participation'] == 50
    assert config['total_steps'] == config['epochs'] * config['privacy']['b_participation'] == 250
    assert config['privacy']['sampling_amplification'] is False
    assert config['privacy']['adjacency'] == 'add_remove_zero_out'
    return config


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


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
    config = load_config(args.config)
    seed_all(config['seed'])
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    is_dp = args.method != 'adam'
    scaled = args.method == 'dp_adam_bandinvmf_scale'
    result_dir = (ROOT / 'exp1/results' / ('smoke' if args.smoke else '') / args.method)
    result_dir.mkdir(parents=True, exist_ok=True)
    coefficients, strategy, workload = build_matrices(
        args.method, config['total_steps'], config['bandinvmf']['num_bands'],
        config['optimizer']['beta1'])
    privacy = calibrate(strategy, config) if is_dp else None
    resolved = dict(config, method=args.method, smoke=args.smoke,
                    visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                    device_name=torch.cuda.get_device_name(),
                    effective_epochs=1 if args.smoke else config['epochs'],
                    effective_steps_per_epoch=2 if args.smoke else 50,
                    test_examples=100 if args.smoke else 10000,
                    fixed_epoch_order=True, download=False,
                    dtype='float32', privacy_calibration=privacy,
                    noising_coefficients=coefficients.tolist(),
                    versions={package: version(package) for package in
                              ('torch', 'torchvision', 'opacus', 'jax', 'jax_privacy',
                               'numpy', 'scipy', 'PyYAML')})
    with open(result_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(resolved, f, sort_keys=False)
    np.savez(result_dir / 'matrices.npz', noising_coefficients=coefficients,
             strategy=strategy, workload_coefficients=workload)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    train = datasets.CIFAR100(ROOT / config['data_root'], train=True,
                              transform=transform, download=False)
    test = datasets.CIFAR100(ROOT / config['data_root'], train=False,
                             transform=transform, download=False)
    assert len(train) == config['dataset_size'] and len(test) == 10000
    # Shuffle ONCE, identically for all methods. Reuse exact logical batches
    # across epochs, implementing fixed-epoch (k,b), rather than random reshuffles.
    permutation = torch.randperm(len(train), generator=torch.Generator().manual_seed(config['seed']))
    np.save(result_dir / 'train_order.npy', permutation.numpy())
    order = permutation[:2000] if args.smoke else permutation
    train_loader = DataLoader(Subset(train, order.tolist()),
                              batch_size=config['physical_batch_size'], shuffle=False,
                              num_workers=config['num_workers'], pin_memory=True,
                              multiprocessing_context='spawn',
                              generator=torch.Generator().manual_seed(config['seed']))
    test_data = Subset(test, range(100)) if args.smoke else test
    test_loader = DataLoader(test_data, batch_size=config['physical_batch_size'],
                             shuffle=False, num_workers=config['num_workers'], pin_memory=True,
                             multiprocessing_context='spawn')
    model = ViTTiny(**config['model']).to(device)
    if scaled:
        model = ScaledGhostModule(model, config['privacy']['max_grad_norm'])
    elif is_dp:
        model = GradSampleModuleFastGradientClipping(
            model, loss_reduction='sum', max_grad_norm=config['privacy']['max_grad_norm'],
            use_ghost_clipping=True)
    opt = config['optimizer']
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'],
                                 betas=(opt['beta1'], opt['beta2']), eps=opt['eps'],
                                 weight_decay=opt['weight_decay'])
    noise = BandInvMFNoise(model.parameters(), coefficients,
                          privacy['innovation_std_sum'], config['total_steps'],
                          config['seed'] + 1) if is_dp else None
    logical = LogicalBatch(model, optimizer, config['gradient_accumulation'],
                           config['logical_batch_size'], noise=noise, scaled=scaled,
                           eps_scale=config['scale']['eps_scale'])
    print(json.dumps({'method': args.method, 'smoke': args.smoke,
                      'calibration': privacy, 'coefficients': coefficients.tolist()}), flush=True)
    fields = ['epoch', 'logical_steps', 'train_examples', 'train_loss', 'test_loss',
              'test_top1', 'clip_fraction', 'gdp_mu', 'gdp_epsilon', 'target_mu',
              'target_epsilon', 'delta', 'noise_std', 'noise_std_space',
              'innovation_std_sum', 'seconds']
    records = []
    started = time.monotonic()
    with open(result_dir / 'metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(resolved['effective_epochs']):
            epoch_start = time.monotonic()
            model.train()
            loss_sum, clipped_count, count = 0.0, 0, 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logical.begin_microbatch()
                if is_dp:
                    loss, clipped = clipped_microbatch(model, inputs, targets,
                                                       config['privacy']['max_grad_norm'])
                    clipped_count += clipped
                else:
                    losses = torch.nn.functional.cross_entropy(model(inputs), targets, reduction='sum')
                    losses.backward()
                    loss = float(losses.detach())
                loss_sum += loss
                count += targets.numel()
                logical.finish_microbatch()
            assert logical.micro_steps == 0
            assert logical.optimizer_steps == (epoch + 1) * resolved['effective_steps_per_epoch']
            test_loss, top1 = evaluate(model, test_loader, device)
            if is_dp:
                sens = fixed_epoch_sensitivity(strategy, config['privacy']['k'],
                                               config['privacy']['b_participation'],
                                               steps=logical.optimizer_steps)
                mu = config['privacy']['max_grad_norm'] * sens / privacy['innovation_std_sum']
                epsilon = epsilon_from_mu(mu, config['privacy']['delta'])
                assert noise.step_count == logical.optimizer_steps
            else:
                mu, epsilon = None, None
            record = dict(epoch=epoch + 1, logical_steps=logical.optimizer_steps,
                          train_examples=count, train_loss=loss_sum/count,
                          test_loss=test_loss, test_top1=top1,
                          clip_fraction=clipped_count/count if is_dp else 0.0,
                          gdp_mu=mu, gdp_epsilon=epsilon,
                          target_mu=privacy['target_mu'] if is_dp else None,
                          target_epsilon=config['privacy']['epsilon'] if is_dp else None,
                          delta=config['privacy']['delta'] if is_dp else None,
                          noise_std=noise.marginal_std(logical.optimizer_steps - 1)/config['logical_batch_size'] if is_dp else 0.0,
                          noise_std_space='scaled' if scaled else 'gradient',
                          innovation_std_sum=privacy['innovation_std_sum'] if is_dp else 0.0,
                          seconds=time.monotonic() - epoch_start)
            records.append(record)
            writer.writerow(record)
            f.flush()
            print(json.dumps(record), flush=True)
    expected_steps = 2 if args.smoke else config['total_steps']
    assert logical.optimizer_steps == expected_steps
    if is_dp:
        assert noise.step_count == expected_steps
    summary = dict(method=args.method, smoke=args.smoke, status='completed',
                   optimizer_steps=logical.optimizer_steps,
                   noise_steps=noise.step_count if is_dp else 0,
                   bandinvmf_steps=noise.step_count if 'bandinvmf' in args.method else 0,
                   planned_total_steps=config['total_steps'],
                   calibration=privacy, final=records[-1], epochs=records,
                   wall_seconds=time.monotonic() - started)
    with open(result_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'logical_steps': logical.optimizer_steps}, result_dir / 'final.pt')
    print(json.dumps({'status': 'completed', 'results': str(result_dir)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'exp1/config.yaml')
    parser.add_argument('--smoke', action='store_true', help='2 logical steps, full-size microbatches/model; test 100 examples')
    run(parser.parse_args())
