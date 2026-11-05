"""One official test pass for a previously selected, frozen training checkpoint."""
import os
import sys
import time
import traceback
from pathlib import Path

import torch

from exp18.model import CNN
from exp18b import BASE, DATA
from exp18b.common import now, read, save, sha, verify_seal
from exp18b.train import evaluate


def run(out):
    verify_seal()
    intent = read(out / "intent.json")
    assert sha(BASE / "best_hyperparameters.json") == intent["selection_sha256"]
    assert read(BASE / "selection_freeze.json")["sha256"] == intent["selection_sha256"]
    selection = read(BASE / "best_hyperparameters.json")
    cfg = read(BASE / "trials" / intent["trial_id"] / "config.json")
    assert cfg["lr"] == selection["groups"][cfg["group"]]["lr"]
    checkpoint = BASE / "trials" / intent["trial_id"] / "checkpoint.pt"
    assert sha(checkpoint) == intent["checkpoint_sha256"]
    assert sha(DATA / "test.pt") == read(BASE / "assets.json")["data/test.pt"]
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.device_count() == 1
    torch.manual_seed(cfg["seed"])
    model = CNN()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    model = model.cuda().eval()
    data = torch.load(DATA / "test.pt", map_location="cpu", weights_only=True)
    save(out / "started.json", dict(time=now(), pid=os.getpid(),
                                    physical_gpu=int(os.environ["CUDA_VISIBLE_DEVICES"])))
    torch.cuda.synchronize()
    start = time.perf_counter()
    metrics = evaluate(model, data["images"], data["labels"])
    torch.cuda.synchronize()
    result = dict(**intent, metrics=metrics, seconds=time.perf_counter() - start,
                  evaluation="official_test", time=now(),
                  physical_gpu=int(os.environ["CUDA_VISIBLE_DEVICES"]))
    assert metrics["n"] == 10000
    save(out / "result.json", result)
    save(out / "complete.json", {p.name: sha(p) for p in out.iterdir()
         if p.is_file() and p.name not in ("complete.json", "worker.log")})
    print(result, flush=True)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    try:
        run(out)
    except Exception as error:
        save(out / "failure.json", dict(time=now(), type=type(error).__name__, message=str(error),
                                        traceback=traceback.format_exc()))
        raise
