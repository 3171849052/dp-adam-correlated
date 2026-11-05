import unittest

import numpy as np
import torch
from opacus import GradSampleModule

from exp18.model import CNN
from exp18.noise import OnlineNoise
from exp18.optimizer import Update, adaptive
from exp18.matrices import workload
from exp18.privacy import epsilon_from_mu, fixed_epoch_sensitivity
from exp18.scheduler import admission_limit
from exp18c import BASE
from exp18c.common import GROUPS, GRID, SEEDS, load_matrix, plan, read, sha, trial_id, validate_config
from exp18c.report import stats


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(SEEDS[0])

    def test_exact_90_trial_grid(self):
        configs = plan()
        self.assertEqual(len(configs), 90)
        self.assertEqual(len({trial_id(cfg) for cfg in configs}), 90)
        for cfg in configs:
            validate_config(cfg)
        for group, methods in GROUPS.items():
            self.assertEqual(sum(c["group"] == group for c in configs), 30)
            for lr in GRID[group]:
                for seed in SEEDS:
                    pair = [c for c in configs if c["group"] == group and c["lr"] == lr and c["seed"] == seed]
                    self.assertEqual({c["method"] for c in pair}, set(methods))
                    self.assertEqual({(c["clip"], c["eps"], c["steps"]) for c in pair},
                                     {(3 if group == "adam" else 4 if group == "momentum" else 100,
                                       .005 if group == "stpex2" else 1e-8, 162)})

    def test_exact_exp18_cnn(self):
        model = CNN()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 11274)
        self.assertEqual(tuple(model(torch.zeros(2, 1, 28, 28)).shape), (2, 10))

    def test_saved_initializations_are_exp18_initializations(self):
        from exp18c.common import tensor_hash
        for seed in SEEDS:
            torch.manual_seed(seed)
            expected = CNN().state_dict()
            actual = torch.load(BASE / "initializations" / f"{seed}.pt", weights_only=True)
            self.assertEqual(tensor_hash(actual), tensor_hash(expected))

    def test_gdp_whole_trajectory_all_clips(self):
        audit = read(BASE / "privacy_audit.json")
        for method, entry in audit["methods"].items():
            self.assertAlmostEqual(epsilon_from_mu(entry["clip"] * entry["sensitivity"] / entry["sigma_sum"], 1e-5), 3, places=9)
            self.assertAlmostEqual(entry["sigma_sum"] / 1000, entry["sigma_average"])
            self.assertFalse(entry["sampling_amplification"])
        self.assertAlmostEqual(fixed_epoch_sensitivity(np.eye(162), 3, 54), np.sqrt(3))

    def test_b2lt_stream_matches_full_matrix_through_band_boundary(self):
        for name in ("sgd", "momentum", "momentum-bias"):
            matrix = load_matrix(name)
            noise = OnlineNoise([torch.zeros(19)], matrix, "b2lt", .7, 162, SEEDS[0])
            actual = torch.stack([noise.next()[0] for _ in range(162)])
            rng = torch.Generator().manual_seed(SEEDS[0])
            innovations = torch.stack([torch.randn(19, generator=rng) * .7 for _ in range(162)])
            expected = torch.tensor(matrix["D"], dtype=torch.float32) @ innovations
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            self.assertEqual(noise.step_count, 162)

    def test_pdf_alpha_endpoints_and_bisection(self):
        d, v = torch.tensor([.25]), torch.tensor([1 / 16])
        alpha, restored = adaptive(torch.tensor([.5]), d, v)
        self.assertAlmostEqual(alpha, .5, places=5)
        self.assertGreaterEqual(float(restored.square().sum()), float(v.sum()))
        self.assertEqual(adaptive(torch.tensor([1.]), d, v)[0], 1)
        self.assertEqual(adaptive(torch.tensor([.125]), d, v)[0], 0)

    def test_stpex2_no_second_parameter_denominator(self):
        model = torch.nn.Linear(2, 1, bias=False)
        opt = Update(model, "stpex2", .03, .005)
        start = model.weight.detach().flatten().clone()
        q = torch.tensor([.2, .4])
        self.assertTrue(torch.equal(opt.d, torch.ones(2)))
        self.assertEqual(opt.step(q), 1)
        torch.testing.assert_close(model.weight.flatten(), start - .03 * q)
        torch.testing.assert_close(opt.v, .001 * q * q)
        torch.testing.assert_close(opt.d, q.abs() + .005)

    def test_adam_matches_torch_reference(self):
        model = torch.nn.Linear(2, 1, bias=False)
        ref = torch.nn.Linear(2, 1, bias=False)
        ref.load_state_dict(model.state_dict())
        opt = Update(model, "adam", .03, 1e-8)
        reference = torch.optim.Adam(ref.parameters(), lr=.03, betas=(.9, .999), eps=1e-8, weight_decay=0)
        for q in (torch.tensor([.2, -.4]), torch.tensor([.1, .3]), torch.tensor([-.5, .2])):
            opt.step(q)
            ref.weight.grad = q.reshape_as(ref.weight)
            reference.step()
            torch.testing.assert_close(model.weight, ref.weight, atol=1e-7, rtol=1e-6)

    def test_momentum_workload_includes_one_minus_beta(self):
        rng = np.random.default_rng(SEEDS[0])
        q = rng.normal(size=162)
        for name in ("momentum", "momentum-bias"):
            u, path, expected = 0., 0., []
            for step, value in enumerate(q, 1):
                u = .9 * u + .1 * value
                path += u / (1 - .9 ** step) if name == "momentum-bias" else u
                expected.append(path)
            np.testing.assert_allclose(workload(name) @ q, expected, atol=1e-12)

    def test_per_sample_gradient_and_scaled_global_clip(self):
        base = CNN()
        model = GradSampleModule(base, loss_reduction="sum")
        x, y = torch.randn(3, 1, 28, 28), torch.tensor([1, 3, 5])
        torch.nn.functional.cross_entropy(model(x), y, reduction="sum").backward()
        actual = torch.cat([p.grad_sample.reshape(3, -1) for p in base.parameters()], 1).clone()
        model.remove_hooks()
        expected = []
        for i in range(3):
            base.zero_grad()
            torch.nn.functional.cross_entropy(base(x[i:i + 1]), y[i:i + 1]).backward()
            expected.append(torch.cat([p.grad.flatten() for p in base.parameters()]))
        torch.testing.assert_close(actual, torch.stack(expected), atol=1e-6, rtol=1e-4)
        scaled = actual / (torch.rand(actual.shape[1]) + .005)
        clipped = scaled * (100 / scaled.norm(dim=1)).clamp_max(1)[:, None]
        self.assertTrue(bool((clipped.norm(dim=1) <= 100.0001).all()))

    def test_dynamic_admission_accounts_for_external_processes(self):
        policy = dict(worker_budget_mib=768, reserve_mib=1536, target_per_gpu=12)
        self.assertEqual(admission_limit(12288, 0, 0, policy), 12)
        self.assertEqual(admission_limit(12288, 8000, 0, policy), 3)
        self.assertEqual(admission_limit(12288, 10000, 2000, policy), 3)
        self.assertEqual(admission_limit(12288, 8500, 8500, policy), 12)

    def test_student_t_five_seed_statistics(self):
        result = stats([1, 2, 3, 4, 5])
        self.assertEqual(result["mean"], 3)
        self.assertAlmostEqual(result["sd"], np.sqrt(2.5))
        self.assertAlmostEqual(result["ci95_high"] - 3, 2.7764451051977987 / np.sqrt(2))


if __name__ == "__main__":
    unittest.main()
