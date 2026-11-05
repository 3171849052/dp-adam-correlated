import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from exp18c import BASE, ROOT
from exp18c.common import GRID, GROUPS, SEEDS, plan, read, save, sha, tensor_hash, trial_id
from exp18c import report


class ExtensionTests(unittest.TestCase):
    def test_only_authorized_protocol_changes(self):
        old, new = [read(ROOT / name / "protocol.json") for name in ("exp18b", "exp18c")]
        allowed = {"experiment", "authorization_revision", "initialization", "B2LT", "grid", "smoke", "comparison", "gpu"}
        self.assertEqual({k: v for k, v in old.items() if k not in allowed},
                         {k: v for k, v in new.items() if k not in allowed})
        self.assertEqual(GRID, {"adam": [.045, .05, .06], "momentum": [.5, .6, .8], "stpex2": [.05, .06, .08]})
        for group in GRID:
            self.assertEqual({k: v for k, v in old["grid"][group].items() if k != "lr"},
                             {k: v for k, v in new["grid"][group].items() if k != "lr"})
            self.assertEqual(new["grid"][group]["lr"], GRID[group])
        self.assertEqual({k: v for k, v in old["B2LT"].items() if k != "source"},
                         {k: v for k, v in new["B2LT"].items() if k != "source"})

    def test_initializations_matrices_privacy_exactly_reused(self):
        old = read(ROOT / "exp18b/initializations.json")
        new = read(BASE / "initializations.json")
        for seed in SEEDS:
            self.assertEqual(sha(ROOT / old[str(seed)]["path"]), sha(ROOT / new[str(seed)]["path"]))
        for source in (ROOT / "exp18b/matrices").iterdir():
            self.assertEqual(sha(source), sha(BASE / "matrices" / source.name))
        self.assertEqual(read(ROOT / "exp18b/privacy_audit.json")["methods"],
                         read(BASE / "privacy_audit.json")["methods"])

    def test_four_discovered_uuid_bindings(self):
        from exp18c.common import env
        policy = read(BASE / "scheduling.json")
        self.assertEqual(len(policy["physical_gpus"]), 4)
        self.assertEqual(len(set(policy["gpu_uuids"].values())), 4)
        for gpu in policy["physical_gpus"]:
            self.assertEqual(env(gpu)["CUDA_VISIBLE_DEVICES"], policy["gpu_uuids"][str(gpu)])
            self.assertEqual(env(gpu)["EXP18C_PHYSICAL_GPU"], str(gpu))

    def test_selection_equal_noise_seed_weight_and_loss_tie(self):
        # IID alone prefers the highest LR, but the equal-noise average ties all
        # LRs. Lower averaged validation loss must then select the middle LR.
        with tempfile.TemporaryDirectory(dir=BASE / "runtime") as temporary:
            base = Path(temporary)
            save(base / "protocol.json", read(BASE / "protocol.json"))
            rows = []
            for cfg in plan():
                index = GRID[cfg["group"]].index(cfg["lr"])
                accuracy = [.2, .4, .6][index] if cfg["method"].endswith("-iid") else [.6, .4, .2][index]
                row = dict(config=cfg, metrics=dict(accuracy=accuracy, loss=[.3, .1, .2][index]),
                    train_loss=999, train_seconds=1, seconds=2, peak_memory=1,
                    training_examples_per_second=1, pairing={"same": cfg["seed"]},
                    final_innovation_rng_sha256="same")
                save(base / trial_id(cfg) / "summary.json", row)
                rows.append(row)
            with patch.object(report, "BASE", base), patch.object(report, "trial_dir", lambda cfg: base / trial_id(cfg)):
                selected = report.select(rows)
            self.assertEqual(len(selected), 30)
            for group, entry in read(base / "best_hyperparameters.json")["groups"].items():
                self.assertEqual(entry["lr"], GRID[group][1])
            self.assertFalse(read(base / "best_hyperparameters.json")["official_test_used_for_selection"])
            rows[0]["pairing"] = {"different": True}
            with self.assertRaises(AssertionError):
                report.verify_pairing(rows)
