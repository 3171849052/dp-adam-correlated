"""One isolated GPU trial; every optimizer input is an unscaled noisy average."""
from exp6.runtime import ROOT, EXP, output_path, require_curve
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.data import Subset
from exp6.config import Trial, METHODS, FIXED, SMOKE_STEPS
from exp6.model import create_model, initialization_digest, checkpoint_sha256
from exp6.data import assets, EpochData, loader
from exp6.geometry import Geometry
from exp6.clipping import per_example, clipped_sum, global_norm
from exp6.mechanism import TemporalNoise, matrices, materialize
from exp6.privacy import calibration, spent
from exp6.audit import fingerprint, environment


def write_json(path, payload):
    output_path(path).write_text(json.dumps(payload, indent=2, allow_nan=False))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def provenance():
    paths = list(EXP.glob('*.py')) + [ROOT / 'exp2' / name for name in ('model.py', 'privacy.py', 'bandinvmf.py')]
    return {str(p.relative_to(ROOT)): checkpoint_sha256(p) for p in sorted(paths)}


def finite_tensors(values):
    assert all(torch.isfinite(t).all() for t in values), 'Non-finite tensor'


@torch.no_grad()
def evaluate(model, dataset, count, seed):
    model.eval()
    loss, correct = 0., 0
    for x, y in loader(Subset(dataset, range(count)), 100, seed):
        x, y = x.cuda(), y.cuda()
        logits = model(x)
        finite_tensors([logits])
        loss += float(F.cross_entropy(logits, y, reduction='sum'))
        correct += int((logits.argmax(1) == y).sum())
    model.train()
    return dict(test_loss=loss/count, test_top1=correct/count, test_examples=count)


def statistics(values, prefix):
    return {prefix+'_mean': float(values.mean()), prefix+'_min': float(values.min()),
            prefix+'_max': float(values.max())}


def run(trial, result_dir, smoke=False):
    require_curve()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in ('1', '2', '3'), 'Expose one physical GPU 1..3'
    folder = output_path(result_dir)
    folder.mkdir(parents=True, exist_ok=True)
    assert not any(p.name != 'train.log' for p in folder.iterdir()), f'Trial directory already used: {folder}'
    started = time.monotonic()
    seed_all(trial.seed)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    model, model_meta = create_model()
    init_digest = initialization_digest(model)
    model = model.cuda()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert len(params) == 98
    optimizer = torch.optim.Adam(params.values(), lr=trial.lr, betas=(.9, .999), eps=1e-8, weight_decay=0)
    coefficients, strategy, workload = matrices(trial)
    privacy = calibration(strategy, trial)
    noise = TemporalNoise(params, coefficients, privacy['innovation_std_sum'], trial.seed+1)
    train, test = assets(model_meta)
    order = torch.randperm(50000, generator=torch.Generator().manual_seed(trial.seed)).numpy()
    order_digest = hashlib.sha256(order.tobytes()).hexdigest()
    np.save(folder / 'train_order.npy', order)
    data_hashes = {name: checkpoint_sha256(ROOT / 'data/cifar-100-python' / name) for name in ('train','test','meta')}
    np.savez(folder / 'matrices.npz', coefficients=coefficients, strategy=strategy,
        workload_coefficients=workload, workload=materialize(workload, 250),
        noising_matrix=materialize(coefficients, 250), innovation_std_sum=privacy['innovation_std_sum'])
    config = dict(trial=trial.asdict(), trial_id=trial.identity, fixed=FIXED, smoke=smoke,
        fingerprint=fingerprint(trial,smoke), audit_environment=environment(),
        actual_steps=SMOKE_STEPS if smoke else 250, model=model_meta, initialization_sha256=init_digest,
        order_sha256=order_digest, privacy=privacy, coefficients=coefficients.tolist(),
        gaussian_seed=trial.seed+1, gaussian_convention='one randn tensor per named trainable parameter in registration order per logical step',
        trainable_names=list(params), data_sha256=data_hashes, source_sha256=provenance(),
        geometry_note='FP64 eigendecomposition; detached FP32 P and inverse; fixed within logical batch; head identity',
        noise_space='scaled_sum' if trial.scaled else 'raw_sum',
        optimizer_input='unscale(noisy query sum) / 1000; standard torch.optim.Adam',
        physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']), device=torch.cuda.get_device_name(),
        torch_cuda_version=torch.version.cuda, versions={p: version(p) for p in
            ('torch','torchvision','numpy','scipy','timm','jax','jax_privacy','safetensors')},
        deterministic_algorithms=True, tf32=False, sdpa='math', workers=2,
        evaluation='official CIFAR100 test; outside training privacy population')
    write_json(folder / 'config.json', config)
    records, augmentation_digests = [], []
    with sdpa_kernel(SDPBackend.MATH), (folder / 'steps.jsonl').open('w') as step_file, (folder / 'geometry.jsonl').open('w') as geometry_file:
        for epoch in range(1 if smoke else 5):
            steps_epoch = SMOKE_STEPS if smoke else 50
            dataset = EpochData(train, order[:steps_epoch*1000], trial.seed, epoch)
            train_loader = iter(loader(dataset, trial.physical_batch_size, trial.seed + epoch))
            for batch in range(steps_epoch):
                t0 = time.monotonic()
                # A/B and the detached metrics below refer to the beginning of this step.
                geometry = Geometry(model, trial.geom_eps)
                query = {n: torch.zeros_like(p) for n, p in params.items()}
                raw_norms, query_norms, factors, losses, correct = [], [], [], [], []
                augmentation = hashlib.sha256()
                for _ in range(1000 // trial.physical_batch_size):
                    x, y = next(train_loader)
                    augmentation.update(x.numpy().tobytes())
                    g, loss, accuracy = per_example(model, x.cuda(), y.cuda())
                    raw_norms.append(global_norm(g))
                    transformed = geometry.transform(g) if trial.scaled else g
                    q, norms, f = clipped_sum(transformed, trial.C)
                    for name in query:
                        query[name].add_(q[name])
                    query_norms.append(norms)
                    factors.append(f)
                    losses.append(loss)
                    correct.append(accuracy)
                finite_tensors(query.values())
                query_norm = float(global_norm(query, per_sample=False) / 1000)
                before_draw = noise.step_count
                perturbation = noise.draw()  # Exactly one DP vector draw, after all physical batches.
                assert noise.step_count == before_draw + 1
                noisy = {n: (q + perturbation[n]) / 1000 for n, q in query.items()}
                adam_input = geometry.transform(noisy, inverse=True) if trial.scaled else noisy
                finite_tensors(adam_input.values())
                previous = {n: p.detach().clone() for n, p in params.items()}
                optimizer.zero_grad(set_to_none=True)
                for n, p in params.items():
                    p.grad = adam_input[n]
                optimizer.step()
                finite_tensors(params.values())
                finite_tensors([v for state in optimizer.state.values() for v in state.values() if torch.is_tensor(v)])
                update_norm = float(sum((p.detach()-previous[n]).square().sum() for n,p in params.items()).sqrt())
                norms = torch.cat(raw_norms)
                qnorms, clip_factors = torch.cat(query_norms), torch.cat(factors)
                aug_digest = augmentation.hexdigest()
                augmentation_digests.append(aug_digest)
                row = dict(step=noise.step_count, epoch=epoch+1, train_examples=1000,
                    train_loss=float(torch.cat(losses).mean()), train_top1=float(torch.cat(correct).mean()),
                    clipping_fraction=float((clip_factors < 1).float().mean()),
                    query_norm=query_norm, query_sum_norm=query_norm*1000,
                    noise_std=noise.marginal_std(noise.step_count-1)/1000,
                    noise_std_space=config['noise_space'].replace('_sum','_average'),
                    innovation_std_sum=privacy['innovation_std_sum'], adam_update_norm=update_norm,
                    adam_input_norm=float(global_norm(adam_input, per_sample=False)),
                    noise_draws=noise.step_count, augmentation_sha256=aug_digest,
                    **statistics(norms, 'raw_per_example_norm'), **statistics(clip_factors, 'clip_factor'),
                    **spent(strategy, noise.step_count, trial, privacy), **geometry.diagnostics(),
                    seconds=time.monotonic()-t0)
                if trial.scaled:
                    row.update(statistics(qnorms, 'scaled_per_example_norm'))
                # Finite serialization is required for metrics as well as tensors.
                line = json.dumps(row, allow_nan=False)
                step_file.write(line+'\n'); step_file.flush()
                geometry_file.write(json.dumps(dict(step=noise.step_count, layers=geometry.layers), allow_nan=False)+'\n')
                geometry_file.flush()
                records.append(row)
                print(json.dumps({k: row[k] for k in ('step','train_loss','train_top1','epsilon','noise_std','seconds')}), flush=True)
        evaluation = evaluate(model, test, 100 if smoke else 10000, trial.seed)
    assert noise.step_count == (SMOKE_STEPS if smoke else 250)
    assert not smoke or len(records) >= 3
    torch.save(dict(trainable={n:p.detach().cpu() for n,p in params.items()}, optimizer=optimizer.state_dict(),
        noise_generator_state=noise.generator.get_state(), noise_history=noise.history,
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), numpy_rng=np.random.get_state(),
        python_rng=random.getstate(), steps=noise.step_count, trial=trial.asdict()), folder / 'final.pt')
    summary = dict(status='completed', trial_id=trial.identity, trial=trial.asdict(), smoke=smoke,
        fingerprint=config['fingerprint'], final_test_top1=evaluation['test_top1'],
        artifact_sha256={name:checkpoint_sha256(folder / name) for name in
            ('final.pt','matrices.npz','train_order.npy','steps.jsonl','geometry.jsonl')},
        initialization_sha256=init_digest, order_sha256=order_digest, augmentation_sha256=augmentation_digests,
        innovation_sha256=noise.innovation_digests, optimizer_steps=len(records), noise_draws=noise.step_count,
        planned_total_steps=250, final_privacy=spent(strategy, noise.step_count, trial, privacy),
        target_epsilon=8., target_delta=1e-5, calibrated_mu=privacy['target_mu'],
        calibration=privacy, final_train=records[-1], **evaluation, wall_seconds=time.monotonic()-started)
    write_json(folder / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--seed', type=int, default=20261001)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--C', type=float, default=1.)
    p.add_argument('--geom-eps', type=float, default=.1)
    p.add_argument('--physical-batch-size', type=int, default=8)
    p.add_argument('--result-dir', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    run(Trial(a.method, a.seed, a.lr, a.C, a.geom_eps, a.physical_batch_size), a.result_dir, a.smoke)


if __name__ == '__main__':
    main()
