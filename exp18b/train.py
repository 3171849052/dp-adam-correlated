"""Exp18 training loop with fixed initialization and validation-only grid trials."""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from opacus import GradSampleModule

from exp18.model import CNN
from exp18.noise import OnlineNoise
from exp18.optimizer import Update
from exp18b import BASE, DATA, ROOT
from exp18b.common import (WORKLOADS, load_matrix, now, read, save, sha, tensor_hash,
                           validate_config, verify_seal)


@torch.no_grad()
def evaluate(model, x, y):
    loss, correct = 0., 0
    for start in range(0, len(y), 1000):
        xx, yy = x[start:start + 1000].cuda(), y[start:start + 1000].cuda()
        logits = model(xx)
        loss += float(torch.nn.functional.cross_entropy(logits, yy, reduction="sum"))
        correct += int((logits.argmax(1) == yy).sum())
    return dict(loss=loss / len(y), accuracy=correct / len(y), n=len(y))


def run(path):
    cfg, out = read(path), path.parent
    validate_config(cfg)
    if cfg["stage"] != "smoke":
        verify_seal()
    assets = read(BASE / "assets.json")
    # The training worker never opens official test data.
    for name in ("train.pt", "split.npz"):
        assert sha(DATA / name) == assets[f"data/{name}"]
    torch.set_num_threads(2)
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.device_count() == 1
    process_start = time.perf_counter()
    initial_meta = read(BASE / "initializations.json")[str(cfg["seed"])]
    assert sha(ROOT / initial_meta["path"]) == initial_meta["sha256"]
    initial = torch.load(ROOT / initial_meta["path"], map_location="cpu", weights_only=True)
    model = CNN()
    model.load_state_dict(initial, strict=True)
    assert tensor_hash(model.state_dict()) == initial_meta["tensor_sha256"]
    model = model.cuda()
    group = cfg["group"]
    opt = Update(model, group, cfg["lr"], cfg["eps"])
    model = GradSampleModule(model, loss_reduction="sum")
    data = torch.load(DATA / "train.pt", map_location="cpu", weights_only=True)
    x, y = data["images"], data["labels"]
    with np.load(DATA / "split.npz") as split:
        train, valid = split["train"].copy(), split["valid"].copy()
    order = train[np.random.default_rng(cfg["seed"]).permutation(len(train))]
    np.save(out / "order.npy", order)
    vx, vy = x[valid], y[valid]
    iid = cfg["method"].endswith("-iid")
    matrix = {"metadata": {}} if iid else load_matrix(WORKLOADS[group])
    privacy = read(BASE / "privacy_audit.json")["methods"][cfg["method"]]
    noise = OnlineNoise(opt.ps, matrix, "iid" if iid else "b2lt", privacy["sigma_sum"],
                        162, cfg["seed"] + 1000000)
    # Inspect the same underlying unit innovations without consuming the live RNG.
    probe_rng = torch.Generator(device="cuda").manual_seed(cfg["seed"] + 1000000)
    probe = {str(i): torch.randn(p.shape, device="cuda", generator=probe_rng)
             for i, p in enumerate(opt.ps)}
    pairing = dict(initial_tensor_sha256=tensor_hash(initial), order_sha256=sha(out / "order.npy"),
                   innovation_seed=cfg["seed"] + 1000000, first_unit_innovation_sha256=tensor_hash(probe))
    del probe
    save(out / "pairing.json", pairing)
    save(out / "privacy.json", privacy)
    save(out / "process.json", dict(pid=os.getpid(), started=now(),
        physical_gpu=int(os.environ["CUDA_VISIBLE_DEVICES"]), visible_devices=torch.cuda.device_count(),
        device_name=torch.cuda.get_device_name(0), device_uuid=str(torch.cuda.get_device_properties(0).uuid),
        batch=1000, precision="FP32", accumulation=1))
    if cfg["stage"] == "smoke":
        save(out / "ready.json", dict(pid=os.getpid(), time=now(),
            physical_gpu=int(os.environ["CUDA_VISIBLE_DEVICES"]), cuda_memory=torch.cuda.memory_allocated()))
        deadline = time.monotonic() + 120
        while len(list((BASE / "smoke").glob("*/ready.json"))) < 6:
            if time.monotonic() > deadline:
                raise RuntimeError("Six-worker CUDA readiness timeout")
            time.sleep(0.2)
        time.sleep(3)
    torch.cuda.synchronize()
    training_start = time.perf_counter()
    train_seconds, validation_seconds, curve = 0., 0., []
    torch.cuda.reset_peak_memory_stats()
    with (out / "steps.jsonl").open("w", buffering=1) as log:
        for step in range(cfg["steps"]):
            start = time.perf_counter()
            ids = order[(step % 54) * 1000:(step % 54 + 1) * 1000]
            xx, yy = x[ids].cuda(), y[ids].cuda()
            model.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(xx), yy, reduction="sum")
            loss.backward()
            gs = torch.cat([p.grad_sample.reshape(1000, -1) for p in opt.ps], 1)
            if group == "stpex2":
                gs.div_(opt.d)
            norms = gs.norm(dim=1)
            weights = (cfg["clip"] / norms.clamp_min(1e-30)).clamp_max(1)
            q = (gs * weights[:, None]).sum(0)
            q.add_(torch.cat([z.flatten() for z in noise.next()]))
            q.div_(1000)
            alpha = opt.step(q)
            if not torch.isfinite(loss) or not all(torch.isfinite(p).all() for p in opt.ps):
                raise FloatingPointError("nonfinite training")
            train_loss = float(loss.detach()) / 1000
            clipped_fraction = float((weights < 1).float().mean())
            del gs, q, xx, yy, loss, norms, weights
            for p in opt.ps:
                p.grad_sample = None
            torch.cuda.synchronize()
            train_seconds += time.perf_counter() - start
            log.write(json.dumps(dict(step=step + 1, loss=train_loss, alpha=alpha,
                clipped_fraction=clipped_fraction, elapsed_seconds=time.perf_counter() - training_start)) + "\n")
            if (step + 1) % 54 == 0 or step + 1 == cfg["steps"]:
                start = time.perf_counter()
                metrics = evaluate(model, vx, vy)
                torch.cuda.synchronize()
                validation_seconds += time.perf_counter() - start
                curve.append(dict(step=step + 1, **metrics))
                save(out / "curve.json", curve)
    torch.save(dict(model=model._module.state_dict(), optimizer=opt.state_dict(), config=cfg,
                    privacy=privacy, pairing=pairing, initial=initial_meta), out / "checkpoint.pt")
    final_rng = tensor_hash({"rng": noise.generator.get_state()})
    summary = dict(status="complete", config=cfg, steps=cfg["steps"], metrics=curve[-1],
        train_loss=train_loss, seconds=time.perf_counter() - process_start,
        train_seconds=train_seconds, validation_seconds=validation_seconds,
        peak_memory=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
        training_examples_per_second=cfg["steps"] * 1000 / train_seconds,
        evaluation="internal_validation", physical_gpu=int(os.environ["CUDA_VISIBLE_DEVICES"]),
        pairing=pairing, final_innovation_rng_sha256=final_rng, ended=now())
    save(out / "summary.json", summary)
    save(out / "complete.json", {p.name: sha(p) for p in out.iterdir()
         if p.is_file() and p.name not in ("complete.json", "worker.log")})
    print(f"Completed {cfg['method']} seed={cfg['seed']} lr={cfg['lr']}: {metrics}", flush=True)


if __name__ == "__main__":
    path = Path(sys.argv[1])
    try:
        run(path)
    except Exception as error:
        save(path.parent / "failure.json", dict(type=type(error).__name__, message=str(error),
             oom=isinstance(error, torch.cuda.OutOfMemoryError), time=now(), traceback=traceback.format_exc()))
        raise
