"""Single-GPU FP32 Adam/clipped Adam diagnostic trial."""
import argparse
import csv
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
from exp1a.model import pretrained_vit, initialization_digest
from exp1a.privacy import calibrate, epsilon_from_mu, fixed_epoch_sensitivity
from exp1a.clipping import LogicalBatch, IIDNoise, EpochNorms, clipped_microbatch
ROOT = Path(__file__).resolve().parents[1]
METHODS = ('adam', 'dp_adam', 'clipped_adam_no_noise')

def load_config(path):
    with open(path) as f:
        config = yaml.safe_load(f)
    for key in ('physical_batch_size', 'gradient_accumulation'):
        value = config[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f'{key} must be a positive integer; got {value!r}')
    if config['logical_batch_size'] != config['physical_batch_size'] * config['gradient_accumulation']:
        raise ValueError('physical_batch_size * gradient_accumulation must equal '
                         f"logical_batch_size ({config['logical_batch_size']})")
    assert config['dataset_size'] % config['logical_batch_size'] == 0
    config['privacy']['k'] = config['epochs']
    config['privacy']['b_participation'] = config['dataset_size'] // config['logical_batch_size']
    config['total_steps'] = config['epochs'] * config['privacy']['b_participation']
    assert config['privacy']['sampling_amplification'] is False
    assert config['privacy']['adjacency'] == 'add_remove_zero_out'
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
    if args.lr is not None:
        config['optimizer']['lr'] = args.lr
    if args.max_grad_norm is not None:
        config['privacy']['max_grad_norm'] = args.max_grad_norm
    seed_all(config['seed'])
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    device = torch.device('cuda:0')
    clipped = args.method != 'adam'
    private = args.method == 'dp_adam'
    directory = args.result_dir.resolve()
    assert directory.is_relative_to(ROOT / 'exp1a')
    directory.mkdir(parents=True, exist_ok=True)
    assert config['model'] == dict(architecture='vit_tiny_patch16_224', pretrained=True,
                                   num_classes=100, image_size=224)
    model, pretrained_cfg = pretrained_vit()
    digest = initialization_digest(model)
    assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())
    model.to(device)
    strategy = np.eye(config['total_steps'])
    privacy = calibrate(strategy, config) if private else None
    resolved = dict(config, method=args.method, smoke=args.smoke,
                    initialization_sha256=digest, pretrained_cfg=pretrained_cfg,
                    privacy_calibration=privacy, dtype='float32', download=False,
                    fixed_epoch_order=True, device_name=torch.cuda.get_device_name(),
                    effective_epochs=1 if args.smoke else config['epochs'],
                    effective_steps_per_epoch=1 if args.smoke else config['privacy']['b_participation'])
    (directory / 'config.yaml').write_text(yaml.safe_dump(resolved, sort_keys=False))
    train_transform, test_transform = image_transforms(pretrained_cfg)
    assert config['data_root'] == 'data'
    train = datasets.CIFAR100(ROOT / 'data', train=True, transform=train_transform, download=False)
    test = datasets.CIFAR100(ROOT / 'data', train=False, transform=test_transform, download=False)
    assert len(train) == config['dataset_size'] and len(test) == 10000
    permutation = torch.randperm(len(train), generator=torch.Generator().manual_seed(config['seed']))
    np.save(directory / 'train_order.npy', permutation.numpy())
    order = permutation[:config['logical_batch_size']] if args.smoke else permutation
    train_loader = DataLoader(Subset(train, order.tolist()), batch_size=config['physical_batch_size'],
                              shuffle=False, num_workers=config['num_workers'], pin_memory=True,
                              multiprocessing_context='spawn',
                              generator=torch.Generator().manual_seed(config['seed']))
    test_loader = DataLoader(Subset(test, range(100)) if args.smoke else test,
                             batch_size=config['physical_batch_size'], shuffle=False,
                             num_workers=config['num_workers'], pin_memory=True,
                             multiprocessing_context='spawn',
                             generator=torch.Generator().manual_seed(config['seed'] + 2))
    clip = config['privacy']['max_grad_norm']
    if clipped:
        model = GradSampleModuleFastGradientClipping(model, loss_reduction='sum',
                                                     max_grad_norm=clip, use_ghost_clipping=True)
        assert set(model.trainable_parameters) == set(model.parameters())
    opt = config['optimizer']
    optimizer = torch.optim.Adam(model.parameters(), lr=opt['lr'], betas=(opt['beta1'], opt['beta2']),
                                 eps=opt['eps'], weight_decay=opt['weight_decay'])
    noise = IIDNoise(model.parameters(), privacy['innovation_std_sum'], config['seed'] + 1) if private else None
    logical = LogicalBatch(model, optimizer, config['gradient_accumulation'], config['logical_batch_size'], noise)
    records = []
    norm_records = []
    norm_fields = ['epoch', 'norm_group', 'p10', 'p25', 'p50', 'p75', 'p90', 'p99', 'mean', 'clip_fraction']
    fields = ['epoch', 'logical_steps', 'train_examples', 'train_loss', 'test_loss', 'test_top1',
              'clip_fraction', 'gdp_mu', 'gdp_epsilon', 'target_mu', 'delta', 'noise_std', 'seconds']
    with open(directory / 'metrics.csv', 'w', newline='') as mf, open(directory / 'norm_stats.csv', 'w', newline='') as nf:
        mw = csv.DictWriter(mf, fieldnames=fields); mw.writeheader()
        nw = csv.DictWriter(nf, fieldnames=norm_fields); nw.writeheader()
        for epoch in range(resolved['effective_epochs']):
            epoch_start = time.monotonic()
            model.train()
            stats = EpochNorms(clip) if clipped else None
            loss_sum, count = 0., 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logical.begin_microbatch()
                if clipped:
                    loss, norms = clipped_microbatch(model, inputs, targets, clip)
                    stats.add(norms)
                else:
                    losses = torch.nn.functional.cross_entropy(model(inputs), targets, reduction='sum')
                    losses.backward()
                    loss = float(losses.detach())
                loss_sum += loss
                count += targets.numel()
                logical.finish_microbatch()
            assert logical.micro_steps == 0
            assert logical.optimizer_steps == (epoch + 1) * resolved['effective_steps_per_epoch']
            aggregates = stats.aggregate() if clipped else None
            norm_records.append(aggregates)
            if clipped:
                for group, values in aggregates.items():
                    nw.writerow(dict(epoch=epoch + 1, norm_group=group, **values))
                nf.flush()
            test_loss, top1 = evaluate(model, test_loader, device)
            mu = clip * fixed_epoch_sensitivity(strategy, config['privacy']['k'],
                     config['privacy']['b_participation'], steps=logical.optimizer_steps) / privacy['innovation_std_sum'] if private else None
            record = dict(epoch=epoch + 1, logical_steps=logical.optimizer_steps, train_examples=count,
                          train_loss=loss_sum/count, test_loss=test_loss, test_top1=top1,
                          clip_fraction=aggregates['full_model_norm']['clip_fraction'] if clipped else None,
                          gdp_mu=mu, gdp_epsilon=epsilon_from_mu(mu, config['privacy']['delta']) if private else None,
                          target_mu=privacy['target_mu'] if private else None,
                          delta=config['privacy']['delta'] if private else None,
                          noise_std=privacy['innovation_std_sum']/config['logical_batch_size'] if private else 0.,
                          seconds=time.monotonic()-epoch_start)
            records.append(record); mw.writerow(record); mf.flush()
            print(json.dumps(record), flush=True)
    assert logical.optimizer_steps == (1 if args.smoke else config['total_steps'])
    if private:
        assert noise.step_count == logical.optimizer_steps
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    logical_steps=logical.optimizer_steps), directory / 'final.pt')
    summary = dict(method=args.method, lr=opt['lr'], max_grad_norm=clip if clipped else None,
                   seed=config['seed'], smoke=args.smoke, status='completed', exit_code=0,
                   initialization_sha256=digest, calibration=privacy, epochs=records, final=records[-1],
                   final_norm_stats=norm_records[-1], optimizer_steps=logical.optimizer_steps,
                   noise_steps=noise.step_count if private else 0, wall_seconds=time.monotonic()-started)
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'exp1a/config.yaml')
    parser.add_argument('--result-dir', type=Path, required=True)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--max-grad-norm', type=float)
    parser.add_argument('--smoke', action='store_true', help='One logical batch of 20 x 50 examples, one epoch, 100 test examples')
    run(parser.parse_args())
