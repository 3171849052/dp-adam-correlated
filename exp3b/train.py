"""One real private source trajectory, with optional streaming signal capture."""
import argparse
import contextlib
import csv
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import yaml
from opacus.grad_sample import GradSampleModuleFastGradientClipping
from exp2.scale import ScaledGhostModule
from exp3.bandinvmf import BandInvMFNoise, build_matrices, materialize
from exp3.mechanism import GeometryGhostModule, clipped_microbatch
from exp3.model import ViTTiny, pretrained_vit, initialization_digest
from exp3.optimizer import HybridOptimizer
from exp3.privacy import calibrate
from exp3b import BASE, ROOT, offline_runtime
from exp3b.capture import CapturedAdamBatch, CapturedMuonBatch, SignalWriter
from exp3b.spec import ADAM, RunSpec, read_completed, write_json
from exp3b.support import support

# Reused model modules set their own cache variables on import. Redirect all
# runtime cache and temporary paths before any execution creates an artifact.
offline_runtime()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def loaders(spec, metadata):
    size = 32 if spec.smoke else 224
    norm = transforms.Normalize(metadata['mean'], metadata['std'])
    interpolation = transforms.InterpolationMode.BICUBIC
    training = transforms.Compose([transforms.RandomResizedCrop(size, interpolation=interpolation),
                                   transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    testing = transforms.Compose([transforms.Resize(int(size / metadata['crop_pct']), interpolation=interpolation),
                                  transforms.CenterCrop(size), transforms.ToTensor(), norm])
    train = datasets.CIFAR100(ROOT / spec.data_root, train=True, transform=training, download=False)
    test = datasets.CIFAR100(ROOT / spec.data_root, train=False, transform=testing, download=False)
    assert len(train) == 50000 and len(test) == 10000
    order = torch.randperm(50000, generator=torch.Generator().manual_seed(spec.seed))
    np.save(spec.directory / 'train_order.npy', order.numpy())
    train_set = Subset(train, order[:16].tolist() if spec.smoke else order.tolist())
    test_set = Subset(test, range(4)) if spec.smoke else test
    workers = {} if spec.smoke else dict(num_workers=2, multiprocessing_context='spawn')
    physical_size = 2 if spec.smoke else 250
    train_seed = spec.seed if spec.method in ADAM else spec.seed + 1
    train_loader = DataLoader(train_set, batch_size=physical_size, shuffle=False, drop_last=True,
                              pin_memory=True, generator=torch.Generator().manual_seed(train_seed), **workers)
    test_loader = DataLoader(test_set, batch_size=physical_size, shuffle=False, pin_memory=True,
                             generator=torch.Generator().manual_seed(spec.seed + 2), **workers)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss = correct = count = 0
    for inputs, labels in loader:
        logits = model(inputs.to(device))
        labels = labels.to(device)
        loss += float(torch.nn.functional.cross_entropy(logits, labels, reduction='sum'))
        correct += int((logits.argmax(1) == labels).sum())
        count += len(labels)
    return loss / count, correct / count


def execute(spec):
    started = time.monotonic()
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('0', '1', '2', '3')
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    seed_all(spec.seed)
    cfg = yaml.safe_load((ROOT / ('exp2/config.yaml' if spec.method in ADAM else 'exp3/config.yaml')).read_text())
    assert cfg['logical_batch_size'] == 1000 and cfg['physical_batch_size'] == 250
    assert cfg['dataset_size'] == 50000 and cfg['epochs'] == 5 and cfg['num_workers'] == 2
    assert cfg['model'] == dict(architecture='vit_tiny_patch16_224', pretrained=True, num_classes=100, image_size=224)
    assert cfg['privacy']['epsilon'] == 8. and cfg['privacy']['delta'] == 1e-5
    assert cfg['privacy']['adjacency'] == 'add_remove_zero_out' and not cfg['privacy']['sampling_amplification']
    assert cfg['bandinvmf']['num_bands'] == 4
    if spec.method in ADAM:
        assert cfg['optimizer']['beta1'] == .9 and cfg['optimizer']['beta2'] == .999
        assert cfg['optimizer']['eps'] == 1e-8 and cfg['optimizer']['weight_decay'] == 0.
        assert cfg['scale']['eps_scale'] == spec.eps_scale
    else:
        assert cfg['muon'] == dict(momentum=.95, nesterov=True, ns_steps=5, dtype='float32')
        assert cfg['adam'] == dict(beta1=.9, beta2=.999, eps=1e-8, weight_decay=0.)
        assert cfg['bandinvmf']['beta_mf'] == .9 and cfg['total_steps'] == 250
    cfg['privacy'].update(k=5, b_participation=50, max_grad_norm=spec.max_grad_norm)
    model, metadata = pretrained_vit()
    checkpoint_sha = metadata['checkpoint_sha256']
    if spec.smoke:
        # Deliberate explicitly-labelled tiny integration mode, never a utility trial.
        model = ViTTiny(embed_dim=8, depth=12, heads=2, mlp_ratio=2, image_size=32)
    initialization_sha = initialization_digest(model)
    model.to(device)
    selected = support(model)
    adam = spec.method in ADAM
    optimizer = (torch.optim.Adam(model.parameters(), lr=spec.lr, betas=(.9, .999), eps=1e-8,
                                  weight_decay=0.) if adam else HybridOptimizer(model, spec.muon_lr, spec.adam_lr))
    if not adam:
        assert set(optimizer.muon) == set(selected)
    private = spec.method != 'nonprivate_hybrid'
    coefficients, strategy, workload = build_matrices(
        'momentum_bandinvmf' if adam or spec.method.startswith('mf_') else 'iid', 250, 4, .9)
    privacy = calibrate(strategy, cfg) if private else None
    sigma = privacy['innovation_std_sum'] if private else 0.
    logical_size = 8 if spec.smoke else 1000
    steps = 2 if spec.smoke else 250
    resolved = dict(spec=spec.mapping(), spec_sha256=spec.fingerprint(), original_config=cfg,
                    privacy_calibration=privacy, checkpoint=metadata, initialization_sha256=initialization_sha,
                    visible_devices=os.environ['CUDA_VISIBLE_DEVICES'], device_name=torch.cuda.get_device_name(),
                    effective_steps=steps, effective_logical_batch_size=logical_size,
                    smoke_model='12 blocks, dim=8, image=32' if spec.smoke else None,
                    data_root=str(ROOT / 'data'), cache_root=str(ROOT / 'cache'), download=False,
                    mf_workload_beta=.9, muon_nesterov_beta=.95, ns_steps=5,
                    support=list(selected), source_initial_optimizer_state='zero')
    write_json(spec.directory / 'config.json', resolved)
    np.savez(spec.directory / 'matrices.npz', coefficients=coefficients, strategy=strategy,
             W=materialize(workload, 250), D=materialize(coefficients, 250), innovation_std_sum=sigma)
    train_loader, test_loader = loaders(spec, metadata)
    scaled = spec.method == 'momentum_scale'
    if adam:
        model = (ScaledGhostModule(model, spec.max_grad_norm) if scaled else
                 GradSampleModuleFastGradientClipping(model, loss_reduction='sum',
                    max_grad_norm=spec.max_grad_norm, use_ghost_clipping=True))
    elif private:
        model = GeometryGhostModule(model, spec.max_grad_norm)
    source_seed = spec.seed + (1 if adam else 3)
    noise = BandInvMFNoise(model.parameters(), coefficients, sigma, 250, source_seed) if private else None
    writer = SignalWriter(spec.directory / 'signals', selected, spec.method, dict(
        coefficients=coefficients.tolist(), innovation_std_sum=sigma,
        innovation_std_gradient=sigma / logical_size, logical_batch_size=logical_size,
        actual_lr=spec.lr if adam else spec.muon_lr, source_seed=spec.seed,
        source_noise_seed=source_seed, source_spec=spec.mapping(),
        initialization_sha256=initialization_sha, checkpoint_sha256=checkpoint_sha,
        initial_optimizer_state='zero', smoke=spec.smoke)) if spec.capture else None
    if adam:
        logical = CapturedAdamBatch(model, optimizer, 4, logical_size, noise,
                                    scaled=scaled, eps_scale=.1, writer=writer)
    else:
        logical = CapturedMuonBatch(model, optimizer, noise, spec.method, spec.max_grad_norm,
            spec.lambda_parallel, spec.kappa, spec.rho, accumulation=4, logical_size=logical_size, writer=writer)
    records = []
    print(json.dumps(dict(status='running', spec=spec.mapping(), calibration=privacy)), flush=True)
    with (spec.directory / 'metrics.csv').open('w', newline='') as stream:
        fields = ('epoch', 'logical_steps', 'train_loss', 'test_loss', 'test_top1', 'clip_fraction', 'seconds')
        csv_writer = csv.DictWriter(stream, fieldnames=fields)
        csv_writer.writeheader()
        for epoch in range(1 if spec.smoke else 5):
            model.train()
            count = clipped = 0
            loss_sum = 0.
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                logical.begin_microbatch()
                if private:
                    loss, clip_count = clipped_microbatch(model, x, y, spec.max_grad_norm)
                else:
                    loss_tensor = torch.nn.functional.cross_entropy(model(x), y, reduction='sum')
                    loss_tensor.backward()
                    loss, clip_count = float(loss_tensor.detach()), 0
                assert np.isfinite(loss)
                count += len(y)
                clipped += clip_count
                loss_sum += loss
                if logical.finish_microbatch() and logical.optimizer_steps % 25 == 0:
                    print(json.dumps(dict(step=logical.optimizer_steps, train_loss=loss_sum/count)), flush=True)
            assert logical.micro_steps == 0
            assert logical.optimizer_steps == (epoch + 1) * (2 if spec.smoke else 50)
            test_loss, top1 = evaluate(model, test_loader, device)
            row = dict(epoch=epoch + 1, logical_steps=logical.optimizer_steps, train_loss=loss_sum/count,
                       test_loss=test_loss, test_top1=top1, clip_fraction=clipped/count,
                       seconds=time.monotonic() - started)
            assert all(np.isfinite(value) for value in row.values())
            csv_writer.writerow(row)
            stream.flush()
            records.append(row)
            print(json.dumps(row), flush=True)
    assert logical.optimizer_steps == steps
    assert noise is None or noise.step_count == steps
    assert all(torch.isfinite(p).all() for p in model.parameters())
    if writer is not None:
        writer.finish(steps)
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), logical_steps=steps),
               spec.directory / 'final.pt')
    summary = dict(status='completed', spec_sha256=spec.fingerprint(), **spec.mapping(),
                   final_test_top1=records[-1]['test_top1'], epoch_test_top1=[r['test_top1'] for r in records],
                   optimizer_steps=steps, noise_steps=noise.step_count if private else 0,
                   innovation_std_sum=sigma, calibration=privacy, support=list(selected),
                   initialization_sha256=initialization_sha, checkpoint_sha256=checkpoint_sha,
                   epochs=records, wall_seconds=time.monotonic() - started)
    write_json(spec.directory / 'summary.json', summary)
    return summary


def run(spec):
    if spec.directory.exists():
        return read_completed(spec)
    (BASE / 'runtime/tmp').mkdir(parents=True, exist_ok=True)
    spec.directory.mkdir(parents=True)
    with (spec.directory / 'train.log').open('x', buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            result = execute(spec)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    run(RunSpec(**json.loads((ROOT / args.spec).read_text())))


if __name__ == '__main__':
    main()
