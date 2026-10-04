"""One trial; full runs use 250 logical steps, explicit smoke uses two."""
import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = str(Path(__file__).resolve().parent / 'cache/huggingface')
os.environ['TORCH_HOME'] = str(Path(__file__).resolve().parent / 'cache/torch')

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import yaml
from exp3.bandinvmf import BandInvMFNoise, build_matrices, materialize
from exp3.diagnostics import MuonDiagnostics, UpdateStatistics, frozen_trajectory_diagnostic
from exp3.mechanism import GeometryGhostModule, LogicalBatch, clipped_microbatch
from exp3.model import pretrained_vit, initialization_digest
from exp3.optimizer import HybridOptimizer
from exp3.privacy import calibrate, fixed_epoch_sensitivity, epsilon_from_mu
from exp3.spec import ROOT, EXP3, METHODS, TrialSpec, read_completed


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
    norm = transforms.Normalize(cfg['mean'], cfg['std'])
    train = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=interpolation),
                                transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    test = transforms.Compose([transforms.Resize(int(224 / cfg['crop_pct']), interpolation=interpolation),
                               transforms.CenterCrop(224), transforms.ToTensor(), norm])
    return train, test


def load_data(cfg):
    train_transform, test_transform = image_transforms(cfg)
    train = datasets.CIFAR100(ROOT / 'data', train=True, transform=train_transform, download=False)
    test = datasets.CIFAR100(ROOT / 'data', train=False, transform=test_transform, download=False)
    assert len(train) == 50000 and len(test) == 10000
    return train, test


def training_order(seed):
    return torch.randperm(50000, generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss, correct, count = 0., 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss += float(torch.nn.functional.cross_entropy(logits, y, reduction='sum'))
        correct += int((logits.argmax(1) == y).sum())
        count += len(y)
    return loss / count, correct / count


def execute(spec):
    started = time.monotonic()
    config = yaml.safe_load((EXP3 / 'config.yaml').read_text())
    assert config['logical_batch_size'] == 1000 and config['physical_batch_size'] == 250
    assert config['gradient_accumulation'] == 4 and config['total_steps'] == 250
    assert config['dataset_size'] == 50000 and config['epochs'] == 5
    assert config['muon'] == dict(momentum=.95, nesterov=True, ns_steps=5, dtype='float32')
    assert config['adam'] == dict(beta1=.9, beta2=.999, eps=1e-8, weight_decay=0.)
    assert config['bandinvmf'] == dict(beta_mf=.9, num_bands=4)
    assert config['privacy'] == dict(epsilon=8., delta=1e-5, k=5, b_participation=50,
                                    adjacency='add_remove_zero_out', sampling_amplification=False)
    config['privacy']['max_grad_norm'] = spec.max_grad_norm
    seed_all(spec.seed)
    torch.set_num_threads(2)
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('1', '2', '3'), 'Expose exactly one of GPUs 1,2,3'
    assert torch.cuda.device_count() == 1
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    model, metadata = pretrained_vit()
    init_sha = initialization_digest(model)
    head_sha = initialization_digest(model.head)
    model.to(device)
    optimizer = HybridOptimizer(model, spec.muon_lr, spec.adam_lr)
    private = spec.method != 'nonprivate_hybrid'
    kind = 'momentum_bandinvmf' if spec.method.startswith('mf_') else 'iid'
    coefficients, strategy, workload = build_matrices(kind, 250, 4, .9)
    privacy = calibrate(strategy, config) if private else None
    sigma = privacy['innovation_std_sum'] if private else 0.
    directory = spec.directory
    resolved = dict(config, trial_spec=spec.mapping(), spec_sha256=spec.fingerprint(),
                    initialization_sha256=init_sha, classifier_initialization_sha256=head_sha,
                    checkpoint_sha256=metadata['checkpoint_sha256'], pretrained_cfg=metadata,
                    privacy_calibration=privacy, noising_coefficients=coefficients.tolist(),
                    download=False, visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                    device_name=torch.cuda.get_device_name(), effective_steps=2 if spec.smoke else 250,
                    augmentation_rng_convention='dedicated seeded DataLoader generators; spawn workers; fixed permutation reused each epoch',
                    diagnostic_note='Phi/update-space JVP at H_t with frozen S_t from H_(t-1); noise weighted and frozen-state trajectory diagnostics are analysis only')
    (directory / 'config.yaml').write_text(yaml.safe_dump(resolved, sort_keys=False))
    np.savez(directory / 'matrices.npz', noising_coefficients=coefficients, strategy=strategy,
             workload_coefficients=workload, W=materialize(workload, 250),
             D=materialize(coefficients, 250), M=materialize(coefficients, 250) * sigma,
             innovation_std_sum=sigma)
    train, test = load_data(metadata)
    order = training_order(spec.seed)
    np.save(directory / 'train_order.npy', order.numpy())
    train_subset = Subset(train, order[:2000].tolist() if spec.smoke else order.tolist())
    test_subset = Subset(test, range(100)) if spec.smoke else test
    train_loader = DataLoader(train_subset, batch_size=250, shuffle=False, drop_last=True,
                              num_workers=2, pin_memory=True, multiprocessing_context='spawn',
                              generator=torch.Generator().manual_seed(spec.seed + 1))
    test_loader = DataLoader(test_subset, batch_size=250, shuffle=False, num_workers=2,
                             pin_memory=True, multiprocessing_context='spawn',
                             generator=torch.Generator().manual_seed(spec.seed + 2))
    base_model = model
    if private:
        model = GeometryGhostModule(model, spec.max_grad_norm)
    noise = BandInvMFNoise(model.parameters(), coefficients, sigma, 250, spec.seed + 3) if private else None
    logical = LogicalBatch(model, optimizer, noise, spec.method, spec.max_grad_norm,
                           spec.lambda_parallel, spec.kappa, spec.rho)
    diagnostics = MuonDiagnostics(optimizer, spec.diagnostic_interval, spec.diagnostic_probes,
                                  innovation_std_sum=sigma if private else None,
                                  save_trajectory=spec.method.startswith('mf_'))
    updates = UpdateStatistics()
    epochs = 1 if spec.smoke else 5
    records, augmentation_digests = [], []
    print(json.dumps(dict(status='running', method=spec.method, seed=spec.seed,
                          calibration=privacy, coefficients=coefficients.tolist())), flush=True)
    with (directory / 'metrics.csv').open('w', newline='') as f:
        fields = ['epoch', 'logical_steps', 'train_examples', 'train_loss', 'test_loss',
                  'test_top1', 'clip_fraction', 'epsilon', 'delta', 'seconds']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for epoch in range(epochs):
            model.train()
            count = clipped = 0
            loss_sum = 0.
            augmentation = hashlib.sha256()
            for physical_step, (x, y) in enumerate(train_loader):
                if physical_step % 4 == 0:
                    augmentation.update(x[:4].contiguous().numpy().tobytes())
                x, y = x.to(device), y.to(device)
                logical.begin_microbatch()
                if private:
                    loss, clip_count = clipped_microbatch(model, x, y, spec.max_grad_norm)
                else:
                    total_loss = torch.nn.functional.cross_entropy(model(x), y, reduction='sum')
                    total_loss.backward()
                    loss, clip_count = float(total_loss.detach()), 0
                count += len(y)
                loss_sum += loss
                clipped += clip_count
                next_step = logical.optimizer_steps + 1
                sample_update = logical.micro_steps == 3 and (next_step == 1 or next_step % spec.diagnostic_interval == 0)
                before = {p: p.detach().clone() for p in model.parameters()} if sample_update else None
                if logical.finish_microbatch():
                    diagnostics.record(logical.optimizer_steps, logical.geometries)
                    if sample_update:
                        updates.record(optimizer, before)
                        print(json.dumps(dict(event='logical_progress', step=logical.optimizer_steps,
                                              train_loss=loss_sum/count, clip_fraction=clipped/count if private else None)), flush=True)
            assert logical.micro_steps == 0
            assert logical.optimizer_steps == (epoch + 1) * (2 if spec.smoke else 50)
            test_loss, accuracy = evaluate(model, test_loader, device)
            if private:
                assert noise.step_count == logical.optimizer_steps
                sens = fixed_epoch_sensitivity(strategy, 5, 50, logical.optimizer_steps)
                epsilon = epsilon_from_mu(spec.max_grad_norm * sens / sigma, 1e-5)
            else:
                epsilon = None
            augmentation_digests.append(augmentation.hexdigest())
            record = dict(epoch=epoch+1, logical_steps=logical.optimizer_steps, train_examples=count,
                          train_loss=loss_sum/count, test_loss=test_loss, test_top1=accuracy,
                          clip_fraction=clipped/count if private else None, epsilon=epsilon,
                          delta=1e-5 if private else None, seconds=time.monotonic()-started)
            writer.writerow(record)
            f.flush()
            records.append(record)
            print(json.dumps(record), flush=True)
    assert optimizer.step_count == logical.optimizer_steps == (2 if spec.smoke else 250)
    diagnostic_summary = diagnostics.save(directory)
    update_summary = updates.save(directory)
    frozen_diagnostic = None
    if spec.method.startswith('mf_'):
        frozen_diagnostic = frozen_trajectory_diagnostic(
            diagnostics.trajectory, materialize(coefficients, 250), sigma, 1000, spec.muon_lr, device)
    (directory / 'frozen_trajectory_muon_mf.json').write_text(json.dumps(frozen_diagnostic, indent=2, allow_nan=False))
    torch.save(dict(model=base_model.state_dict(), optimizer=optimizer.state_dict(),
                    logical_steps=logical.optimizer_steps), directory / 'final.pt')
    summary = dict(status='completed', **spec.mapping(), spec_sha256=spec.fingerprint(),
                   final_test_top1=records[-1]['test_top1'], best_test_top1=max(r['test_top1'] for r in records),
                   train_loss=records[-1]['train_loss'], epoch_test_top1=[r['test_top1'] for r in records],
                   epoch_clip_fraction=[r['clip_fraction'] for r in records], epsilon=records[-1]['epsilon'],
                   target_epsilon=8. if private else None, delta=1e-5 if private else None,
                   innovation_std_sum=sigma, optimizer_steps=logical.optimizer_steps,
                   noise_steps=noise.step_count if private else 0, physical_batches=logical.optimizer_steps*4,
                   initialization_sha256=init_sha, classifier_initialization_sha256=head_sha,
                   checkpoint_sha256=metadata['checkpoint_sha256'], augmentation_trace_sha256=augmentation_digests,
                   train_order_sha256=hashlib.sha256(order.numpy().tobytes()).hexdigest(),
                   planned_total_steps=250, epochs=records, diagnostics=diagnostic_summary,
                   update_statistics=update_summary, frozen_trajectory_muon_mf=frozen_diagnostic,
                   muon_parameters=list(optimizer.muon), adam_parameters=list(optimizer.adam),
                   wall_seconds=time.monotonic()-started)
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(dict(status='completed', result_dir=str(directory))), flush=True)
    return summary


def run(spec):
    directory = spec.directory
    reserved = os.environ.get('EXP3_LAUNCHER_RESERVED') == str(directory)
    if directory.exists() and not reserved:
        return read_completed(spec)
    if reserved:
        assert {p.name for p in directory.iterdir()} == {'train.log'}, 'Incomplete reserved trial'
        return execute(spec)
    directory.mkdir(parents=True)
    with (directory / 'train.log').open('x', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = execute(spec)
    print(json.dumps(dict(status='completed', result_dir=str(directory))), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', required=True, choices=METHODS)
    parser.add_argument('--result-dir', required=True)
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--muon-lr', type=float, default=.01)
    parser.add_argument('--adam-lr', type=float, default=.0005)
    parser.add_argument('--max-grad-norm', type=float, default=100.)
    parser.add_argument('--lambda-parallel', type=float, default=1.)
    parser.add_argument('--kappa', type=float, default=4.)
    parser.add_argument('--rho', type=float, default=.1)
    parser.add_argument('--diagnostic-interval', type=int, default=25)
    parser.add_argument('--diagnostic-probes', type=int, default=4)
    parser.add_argument('--smoke', action='store_true')
    run(TrialSpec(**vars(parser.parse_args())))


if __name__ == '__main__':
    main()
