"""One isolated GPU worker using Exp10's standard Adam and exact Ghost clipping."""
import argparse
import csv
import hashlib
import json
import os
import random
import time
import traceback
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from exp11 import RESULTS
from exp11.config import T, CLIP, LR, SEED, protocol, trial_output, save_json, file_hash, array_hash, code_hashes
from exp11.clipping import StandardGhostModule, LogicalBatch, clipped_microbatch
from exp11.cv_model import pretrained_vit, initialization_digest
from exp11.data import cv_datasets
from exp11.matrices import get_matrix
from exp11.noise import BandInvMFNoise
from exp11.privacy import calibrate, fixed_epoch_sensitivity, epsilon_from_mu


def seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss = correct = count = 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss += float(torch.nn.functional.cross_entropy(logits, targets, reduction='sum'))
        correct += int((logits.argmax(1) == targets).sum())
        count += len(targets)
    if not np.isfinite(loss):
        raise FloatingPointError('non-finite internal-validation loss')
    assert count == 5000
    return dict(accuracy=correct / count, validation_loss=loss / count, validation_examples=count)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(lam, smoke=False):
    started = time.monotonic()
    physical_gpu = int(os.environ['CUDA_VISIBLE_DEVICES'])
    assert physical_gpu in (0, 1, 2, 3)
    assert torch.cuda.device_count() == 1
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    seed_all()
    output = trial_output(lam, smoke)
    output.mkdir(parents=True, exist_ok=True)
    resolved = dict(protocol(lam, smoke), physical_gpu=physical_gpu, local_device='cuda:0',
                    code_sha256=code_hashes(), assets_manifest_sha256=file_hash(RESULTS / 'assets_manifest.json'))
    save_json(output / 'config.json', resolved)
    model = None
    try:
        base, metadata = pretrained_vit()
        init_hash, head_hash = initialization_digest(base), initialization_digest(base.head)
        training, validation, order = cv_datasets(metadata)
        model = StandardGhostModule(base.to(device), CLIP)
        assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(.9, .999), eps=1e-8, weight_decay=0.)
        d, S, W, matrix_meta = get_matrix(lam)
        privacy = calibrate(S, protocol(lam, smoke))
        full_epsilon = epsilon_from_mu(CLIP * privacy['sensitivity'] / privacy['innovation_std_sum'], 1e-5)
        assert abs(full_epsilon - 8.) < 1e-7
        assert np.isclose(privacy['innovation_std_sum'] / 1000, matrix_meta['sigma_avg'], rtol=1e-10)
        privacy.update(verified_full_epsilon=full_epsilon, delta=1e-5, target_epsilon=8.)
        np.savez(output / 'matrices.npz', coefficients=d, strategy=S, workload=W)
        np.save(output / 'train_order.npy', order)
        noise = BandInvMFNoise(model.parameters(), d, privacy['innovation_std_sum'], T, SEED + 1)
        logical = LogicalBatch(model, optimizer, 4, 1000, noise)
        if smoke:
            training = Subset(training, range(3000))
        loader = DataLoader(training, batch_size=250, shuffle=False, pin_memory=True,
                            num_workers=2, generator=torch.Generator().manual_seed(SEED),
                            multiprocessing_context='spawn')
        eval_loader = DataLoader(validation, batch_size=250, shuffle=False)
        torch.cuda.reset_peak_memory_stats(device)
        records, rng_trace = [], []
        micro_count = 0
        train_seconds = 0.
        with (output / 'steps.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['step', 'epoch', 'step_seconds', 'noise_marginal_std_avg'])
            writer.writeheader()
            for epoch in range(1 if smoke else 5):
                model.train()
                augmentation = hashlib.sha256()
                for images, targets in loader:
                    if logical.micro_steps == 0:
                        torch.cuda.synchronize()
                        step_started = time.monotonic()
                        augmentation.update(images[:4].contiguous().numpy().tobytes())
                    images, targets = images.to(device), targets.to(device)
                    assert len(targets) == 250
                    logical.begin_microbatch()
                    clipped_microbatch(model, images, targets, CLIP)
                    micro_count += 1
                    if logical.finish_microbatch():
                        tensors = list(model.parameters()) + [state[key] for state in optimizer.state.values()
                                                              for key in ('exp_avg', 'exp_avg_sq')]
                        if not all(torch.isfinite(t).all().item() for t in tensors):
                            raise FloatingPointError('non-finite model/Adam state')
                        torch.cuda.synchronize()
                        elapsed = time.monotonic() - step_started
                        train_seconds += elapsed
                        row = dict(step=logical.optimizer_steps, epoch=epoch + 1, step_seconds=elapsed,
                                   noise_marginal_std_avg=noise.marginal_std(logical.optimizer_steps - 1) / 1000)
                        writer.writerow(row)
                        f.flush()
                        if logical.optimizer_steps % 10 == 0 or smoke:
                            print(json.dumps(row), flush=True)
                assert logical.optimizer_steps == (epoch + 1) * (3 if smoke else 45)
                assert logical.micro_steps == 0
                mu = CLIP * fixed_epoch_sensitivity(S, 5, 45, steps=logical.optimizer_steps) / privacy['innovation_std_sum']
                record = dict(epoch=epoch + 1, step=logical.optimizer_steps, epsilon=epsilon_from_mu(mu, 1e-5))
                if not smoke:
                    record.update(evaluate(model, eval_loader, device))
                assert record['epsilon'] <= 8. + 1e-7
                records.append(record)
                rng_trace.append(augmentation.hexdigest())
                write_csv(output / 'metrics.csv', records)
                print(json.dumps(record), flush=True)
        steps = 3 if smoke else T
        assert logical.optimizer_steps == noise.step_count == steps and micro_count == steps * 4
        if not smoke:
            assert abs(records[-1]['epsilon'] - 8.) < 1e-7
        result = dict(lambda_value=lam, seed=SEED, status='completed', smoke=smoke,
                      completed_epochs=0 if smoke else 5, optimizer_steps=steps, noise_draws=noise.step_count,
                      physical_batches=micro_count, planned_total_steps=T, finite=True, physical_gpu=physical_gpu,
                      initialization_sha256=init_hash, classifier_initialization_sha256=head_hash,
                      train_order_sha256=file_hash(output / 'train_order.npy'), rng_trace=rng_trace,
                      strategy_sha256=array_hash(S), coefficients=d.tolist(), calibration=privacy,
                      actual_epsilon=records[-1]['epsilon'], epochs=records, matrix_optimization=matrix_meta,
                      official_evaluation_used=False, seconds=time.monotonic() - started,
                      train_seconds=train_seconds, logical_step_seconds=train_seconds / steps,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                      device=torch.cuda.get_device_name(device), code_sha256=resolved['code_sha256'],
                      assets_manifest_sha256=resolved['assets_manifest_sha256'], pretrained=metadata)
        if not smoke:
            result.update(accuracy=records[-1]['accuracy'], validation_loss=records[-1]['validation_loss'])
        result['file_sha256'] = {name: file_hash(output / name) for name in
                                ('config.json', 'matrices.npz', 'train_order.npy', 'metrics.csv', 'steps.csv')}
        save_json(output / 'summary.json', result)
        return result
    except Exception as error:
        save_json(output / 'summary.json', dict(lambda_value=lam, seed=SEED, status='failed', smoke=smoke,
                                               error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc(),
                                               seconds=time.monotonic() - started,
                                               completed_optimizer_steps=logical.optimizer_steps if 'logical' in locals() else 0))
        raise
    finally:
        if model is not None:
            model.remove_hooks()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lambda-value', type=float, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.lambda_value, args.smoke)
