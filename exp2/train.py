"""Run one method on the single GPU exposed by CUDA_VISIBLE_DEVICES."""
import argparse
import csv
import hashlib
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

from exp2.bandinvmf import BandInvMFNoise, build_matrices, materialize
from exp2.model import pretrained_vit, initialization_digest
from exp2.privacy import calibrate, epsilon_from_mu, fixed_epoch_sensitivity
from exp2.diagnostics import MechanismTrace
from exp2.cells import CELLS, FAMILIES, cell_for
from exp2.scale import LogicalBatch, ScaledGhostModule, clipped_microbatch

ROOT = Path(__file__).resolve().parents[1]
METHODS = tuple(CELLS) + tuple(f'{family}_standard_matched' for family in FAMILIES)


class AugmentationTrace:
    """Hash four CPU transformed examples from the start of every logical batch."""
    def __init__(self, accumulation):
        self.accumulation = accumulation
        self.physical_batches = 0
        self.digest = hashlib.sha256()

    def update(self, inputs):
        if self.physical_batches % self.accumulation == 0:
            assert inputs.device.type == 'cpu'
            self.digest.update(inputs[:4].contiguous().numpy().tobytes())
        self.physical_batches += 1

    def hexdigest(self):
        assert self.physical_batches % self.accumulation == 0
        return self.digest.hexdigest()


def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    assert config['logical_batch_size'] % config['physical_batch_size'] == 0
    config['gradient_accumulation'] = config['logical_batch_size'] // config['physical_batch_size']
    assert config['epochs'] == 5 and config['dataset_size'] == 50000
    assert config['physical_batch_size'] == 250 and config['logical_batch_size'] == 1000
    assert config['gradient_accumulation'] == 4
    assert config['optimizer']['beta1'] == .9 and config['optimizer']['beta2'] == .999
    assert config['optimizer']['eps'] == 1e-8 and config['optimizer']['weight_decay'] == 0
    assert config['dataset_size'] % config['logical_batch_size'] == 0
    config['privacy']['k'] = config['epochs']
    config['privacy']['b_participation'] = config['dataset_size'] // config['logical_batch_size']
    config['total_steps'] = config['epochs'] * config['privacy']['b_participation']
    assert config['bandinvmf']['num_bands'] == 4
    assert config['privacy']['epsilon'] == 8 and config['privacy']['delta'] == 1e-5
    assert config['data_root'] == 'data'
    assert config['privacy']['sampling_amplification'] is False
    assert config['privacy']['adjacency'] == 'add_remove_zero_out'
    assert config['scale']['eps_scale'] == .1
    return config


def image_transforms(cfg):
    interpolation = transforms.InterpolationMode.BICUBIC
    normalize = transforms.Normalize(cfg['mean'], cfg['std'])
    train = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=interpolation),
                                transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    test = transforms.Compose([transforms.Resize(int(224 / cfg['crop_pct']), interpolation=interpolation),
                               transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    return train, test


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
    started = time.monotonic()
    config = load_config(args.config)
    if args.seed is not None:
        config['seed'] = args.seed
    seed_all(config['seed'])
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    cell = cell_for(args.method)
    scaled = cell['geometry'] == 'scale'
    result_dir = args.result_dir or (ROOT / 'exp2/results' / ('smoke' if args.smoke else '') / args.method)
    result_dir = result_dir.resolve()
    assert result_dir.is_relative_to(ROOT / 'exp2')
    for value, section, key in ((args.lr, 'optimizer', 'lr'),
                                (args.eps_scale, 'scale', 'eps_scale'),
                                (args.max_grad_norm, 'privacy', 'max_grad_norm')):
        if value is not None:
            config[section][key] = value
    assert config['scale']['eps_scale'] == .1
    assert config['model'] == dict(architecture='vit_tiny_patch16_224', pretrained=True,
                                   num_classes=100, image_size=224)
    model, pretrained_cfg = pretrained_vit()
    init_digest = initialization_digest(model)
    head_digest = initialization_digest(model.head)
    model = model.to(device)
    if result_dir.exists():
        if {p.name for p in result_dir.iterdir()} != {'train.log'}:
            raise RuntimeError(f'Incomplete trial directory: {result_dir}\nRemove it manually before restarting.')
    else:
        result_dir.mkdir(parents=True)
    coefficients, strategy, workload = build_matrices(
        cell['noise'], config['total_steps'], config['bandinvmf']['num_bands'],
        config['optimizer']['beta1'])
    privacy = calibrate(strategy, config)
    resolved = dict(config, method=args.method, **cell, smoke=args.smoke,
                    visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                    device_name=torch.cuda.get_device_name(),
                    effective_epochs=1 if args.smoke else config['epochs'],
                    effective_steps_per_epoch=1 if args.smoke else config['privacy']['b_participation'],
                    initialization_sha256=init_digest, classifier_initialization_sha256=head_digest,
                    pretrained_checkpoint_sha256=pretrained_cfg['checkpoint_sha256'],
                    pretrained_cfg=pretrained_cfg,
                    test_examples=100 if args.smoke else 10000,
                    fixed_epoch_order=True, download=False,
                    augmentation_rng_convention='seeded train DataLoader generator + torch worker seeds; fixed permutation each epoch',
                    diagnostic_coordinates=2048, coordinate_seed=0,
                    mechanism_note='p from completed current vhat; s is actual previous-vhat scale; r=p/s; not exact Adam Jacobian',
                    augmentation_trace_convention='rolling SHA256 of first 4 CPU transformed examples of every logical batch, reset each epoch',
                    dtype='float32', privacy_calibration=privacy,
                    noising_coefficients=coefficients.tolist(),
                    versions={package: version(package) for package in
                              ('torch', 'torchvision', 'opacus', 'jax', 'jax_privacy',
                               'numpy', 'scipy', 'PyYAML', 'timm')})
    with open(result_dir / 'config.yaml', 'w') as f:
        yaml.safe_dump(resolved, f, sort_keys=False)
    M = materialize(coefficients, config['total_steps']) * privacy['innovation_std_sum']
    W = materialize(workload, config['total_steps'])
    np.savez(result_dir / 'matrices.npz', noising_coefficients=coefficients,
             strategy=strategy, workload_coefficients=workload, M=M, W=W,
             innovation_std_sum=privacy['innovation_std_sum'])

    train_transform, test_transform = image_transforms(pretrained_cfg)
    train = datasets.CIFAR100(ROOT / config['data_root'], train=True,
                              transform=train_transform, download=False)
    test = datasets.CIFAR100(ROOT / config['data_root'], train=False,
                             transform=test_transform, download=False)
    assert len(train) == config['dataset_size'] and len(test) == 10000
    # Shuffle ONCE, identically for all methods. Reuse exact logical batches
    # across epochs, implementing fixed-epoch (k,b), rather than random reshuffles.
    permutation = torch.randperm(len(train), generator=torch.Generator().manual_seed(config['seed']))
    np.save(result_dir / 'train_order.npy', permutation.numpy())
    order = permutation[:config['logical_batch_size']] if args.smoke else permutation
    train_loader = DataLoader(Subset(train, order.tolist()),
                              batch_size=config['physical_batch_size'], shuffle=False,
                              num_workers=config['num_workers'], pin_memory=True,
                              multiprocessing_context='spawn',
                              generator=torch.Generator().manual_seed(config['seed']))
    test_data = Subset(test, range(100)) if args.smoke else test
    test_loader = DataLoader(test_data, batch_size=config['physical_batch_size'],
                             shuffle=False, num_workers=config['num_workers'], pin_memory=True,
                             multiprocessing_context='spawn',
                             generator=torch.Generator().manual_seed(config['seed'] + 2))
    if scaled:
        model = ScaledGhostModule(model, config['privacy']['max_grad_norm'])
    else:
        model = GradSampleModuleFastGradientClipping(
            model, loss_reduction='sum', max_grad_norm=config['privacy']['max_grad_norm'],
            use_ghost_clipping=True)
    assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())
    assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in model.trainable_parameters}
    opt = config['optimizer']
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'],
                                 betas=(opt['beta1'], opt['beta2']), eps=opt['eps'],
                                 weight_decay=opt['weight_decay'])
    noise = BandInvMFNoise(model.parameters(), coefficients,
                          privacy['innovation_std_sum'], config['total_steps'],
                          config['seed'] + 1)
    logical = LogicalBatch(model, optimizer, config['gradient_accumulation'],
                           config['logical_batch_size'], noise=noise, scaled=scaled,
                           eps_scale=config['scale']['eps_scale'])
    trace = MechanismTrace(model.parameters(), scaled)
    print(json.dumps({'method': args.method, 'smoke': args.smoke,
                      'calibration': privacy, 'coefficients': coefficients.tolist()}), flush=True)
    fields = ['epoch', 'logical_steps', 'train_examples', 'train_loss', 'test_loss',
              'test_top1', 'clip_fraction', 'gdp_mu', 'gdp_epsilon', 'target_mu',
              'target_epsilon', 'delta', 'noise_std', 'noise_std_space',
              'innovation_std_sum', 'augmentation_trace_sha256', 'seconds']
    records = []
    augmentation_digests = []
    with open(result_dir / 'metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for epoch in range(resolved['effective_epochs']):
            epoch_start = time.monotonic()
            model.train()
            loss_sum, clipped_count, count = 0.0, 0, 0
            augmentation_trace = AugmentationTrace(config['gradient_accumulation'])
            for inputs, targets in train_loader:
                augmentation_trace.update(inputs)
                inputs, targets = inputs.to(device), targets.to(device)
                logical.begin_microbatch()
                loss, clipped = clipped_microbatch(model, inputs, targets,
                                                   config['privacy']['max_grad_norm'])
                clipped_count += clipped
                loss_sum += loss
                count += targets.numel()
                if logical.finish_microbatch():
                    trace.record(optimizer)
            assert logical.micro_steps == 0
            assert logical.optimizer_steps == (epoch + 1) * resolved['effective_steps_per_epoch']
            test_loss, top1 = evaluate(model, test_loader, device)
            sens = fixed_epoch_sensitivity(strategy, config['privacy']['k'],
                                           config['privacy']['b_participation'],
                                           steps=logical.optimizer_steps)
            mu = config['privacy']['max_grad_norm'] * sens / privacy['innovation_std_sum']
            epsilon = epsilon_from_mu(mu, config['privacy']['delta'])
            assert noise.step_count == logical.optimizer_steps
            augmentation_digests.append(augmentation_trace.hexdigest())
            record = dict(epoch=epoch + 1, logical_steps=logical.optimizer_steps,
                          train_examples=count, train_loss=loss_sum/count,
                          test_loss=test_loss, test_top1=top1,
                          clip_fraction=clipped_count/count,
                          gdp_mu=mu, gdp_epsilon=epsilon,
                          target_mu=privacy['target_mu'],
                          target_epsilon=config['privacy']['epsilon'],
                          delta=config['privacy']['delta'],
                          noise_std=noise.marginal_std(logical.optimizer_steps - 1)/config['logical_batch_size'],
                          noise_std_space='scaled' if scaled else 'gradient',
                          innovation_std_sum=privacy['innovation_std_sum'],
                          augmentation_trace_sha256=augmentation_digests[-1],
                          seconds=time.monotonic() - epoch_start)
            records.append(record)
            writer.writerow(record)
            f.flush()
            print(json.dumps(record), flush=True)
    expected_steps = 1 if args.smoke else config['total_steps']
    assert logical.optimizer_steps == expected_steps
    assert noise.step_count == expected_steps
    trace.save(result_dir)
    summary = dict(method=args.method, **cell, smoke=args.smoke, status='completed',
                   lr=opt['lr'], eps_scale=config['scale']['eps_scale'] if scaled else None,
                   max_grad_norm=config['privacy']['max_grad_norm'],
                   seed=config['seed'], initialization_sha256=init_digest,
                   classifier_initialization_sha256=head_digest,
                   pretrained_checkpoint_sha256=pretrained_cfg['checkpoint_sha256'],
                   num_bands=config['bandinvmf']['num_bands'] if cell['noise'] != 'iid' else None,
                   optimizer_steps=logical.optimizer_steps,
                   noise_steps=noise.step_count,
                   bandinvmf_steps=noise.step_count if cell['noise'] != 'iid' else 0,
                   planned_total_steps=config['total_steps'],
                   augmentation_trace_sha256=augmentation_digests,
                   physical_batches=logical.optimizer_steps * config['gradient_accumulation'],
                   calibration=privacy, final=records[-1], epochs=records,
                   wall_seconds=time.monotonic() - started)
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'logical_steps': logical.optimizer_steps}, result_dir / 'final.pt')
    with open(result_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    print(json.dumps({'status': 'completed', 'results': str(result_dir)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'exp2/config.yaml')
    parser.add_argument('--result-dir', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--eps-scale', type=float)
    parser.add_argument('--max-grad-norm', type=float)
    parser.add_argument('--smoke', action='store_true', help='1 logical step, 4 full-size microbatches; test 100 examples')
    run(parser.parse_args())
