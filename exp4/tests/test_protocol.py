import json
import socket
import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from torchvision import datasets

from exp4.config import METHODS, load_config
from exp4.runtime import ROOT, EXP, output_path
from exp4.model import pretrained_vit, checkpoint_path, StableLayerNorm, Attention, ViTTiny
from exp4.search import radius_points, refinement_points, winner
from exp4.final_runner import make_trials as final_trials
from exp4.launch_batch import trial_command, launch
from exp4.report import write_final_report


def test_protocol_and_only_two_search_dimensions():
    cfg = load_config()
    assert cfg['total_steps'] == cfg['epochs'] * cfg['dataset_size'] // cfg['logical_batch_size'] == 250
    assert cfg['gradient_accumulation'] == 4
    assert cfg['bandinvmf_num_bands'] == 4 and cfg['bandinvmf_workload'] == 'sgd'
    assert radius_points(256., .001) == [(.001, 2. ** i) for i in range(9)]
    assert len(refinement_points(dict(R=2., lr=.001))) == 9
    frozen = json.loads((EXP / 'specs/frozen_baseline.json').read_text())
    final = final_trials(frozen, EXP / 'results/final')
    assert len(final) == 6
    assert {t['seed'] for t in final} == {20261011, 20261012, 20261013}
    assert all(trial_command(t)[0][:4] == [trial_command(t)[0][0], '-B', '-m', 'exp4.train'] for t in final)


def test_outputs_cannot_escape_exp4():
    for path in (ROOT / 'exp2/forbidden', ROOT / 'results', EXP / '../exp3/nope'):
        with pytest.raises(ValueError):
            output_path(path)


def test_layernorm_large_backward_matches_double_reference_without_changing_parameters():
    torch.manual_seed(10)
    layer = StableLayerNorm(192, eps=1e-6)
    with torch.no_grad():
        layer.weight.fill_(1e10)
    x = (torch.randn(4, 192) * 1e10).requires_grad_()
    upstream = torch.randn(4, 192) * 1e30
    output = layer(x)
    output.backward(upstream)
    rx = x.detach().double().requires_grad_()
    rw = layer.weight.detach().double().requires_grad_()
    rb = layer.bias.detach().double().requires_grad_()
    reference = torch.nn.functional.layer_norm(rx, (192,), rw, rb, 1e-6)
    reference.backward(upstream.double())
    torch.testing.assert_close(output, reference.float())
    for actual, expected in ((x.grad, rx.grad), (layer.weight.grad, rw.grad), (layer.bias.grad, rb.grad)):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected.float())
    assert layer.weight.dtype == layer.bias.dtype == output.dtype == torch.float32


def test_math_attention_large_logits_has_finite_backward(monkeypatch):
    original = torch.nn.functional.scaled_dot_product_attention
    observed = []
    def checked_attention(q, k, v, **kwargs):
        observed.append((q.dtype, k.dtype, v.dtype))
        return original(q, k, v, **kwargs)
    monkeypatch.setattr(torch.nn.functional, 'scaled_dot_product_attention', checked_attention)
    torch.manual_seed(16)
    layer = Attention(6, 3)
    with torch.no_grad():
        layer.qkv.weight.mul_(1e4)
    x = (torch.randn(2, 4, 6) * 1e4).requires_grad_()
    output = layer(x)
    output.square().mean().backward()
    assert torch.isfinite(output).all() and torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters())
    assert observed == [(torch.float64, torch.float64, torch.float64)]
    assert output.dtype == x.dtype == layer.qkv.weight.dtype == torch.float32


def test_block_checkpointing_preserves_forward_and_raw_gradients():
    torch.manual_seed(25)
    model = ViTTiny(patch_size=4, embed_dim=6, depth=2, heads=3,
                    mlp_ratio=2, num_classes=3, image_size=8)
    reference = copy.deepcopy(model)
    model.train()  # checkpoint every block
    reference.eval()  # same operators/parameters, with checkpointing disabled
    x = torch.randn(2, 3, 8, 8, requires_grad=True)
    rx = x.detach().clone().requires_grad_()
    actual, expected = model(x), reference(rx)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, rx.grad)
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_local_data_and_checkpoint_with_network_connections_forbidden(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Network access forbidden during local resource test')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    train = datasets.CIFAR100(ROOT / 'data', train=True, download=False)
    test = datasets.CIFAR100(ROOT / 'data', train=False, download=False)
    assert len(train) == 50000 and len(test) == 10000
    assert Path(train.root).resolve() == (ROOT / 'data').resolve()
    path = checkpoint_path()
    assert path.resolve().is_relative_to(ROOT / 'cache')
    model, metadata = pretrained_vit()
    assert model.head.out_features == 100
    assert len(metadata['checkpoint_sha256']) == 64
    assert metadata['checkpoint_path'].startswith('cache/')
    assert all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters())


def test_launcher_schedules_only_0_2_3_and_at_most_three(monkeypatch, tmp_path):
    import exp4.launch_batch as module
    running, gpu_history = {}, []

    class FakeProcess:
        counter = 0
        def __init__(self, command, cwd, env, stdout, stderr):
            gpu = int(env['CUDA_VISIBLE_DEVICES'])
            assert gpu in (0, 2, 3) and gpu not in running
            assert Path(cwd) == ROOT
            self.gpu = gpu
            self.pid = FakeProcess.counter
            FakeProcess.counter += 1
            running[gpu] = self
            gpu_history.append(gpu)
            assert len(running) <= 3
            self.destination = Path(command[command.index('--result-dir') + 1])
            self.polls = 0

        def poll(self):
            self.polls += 1
            if self.polls < 2:
                return None
            running.pop(self.gpu)
            (self.destination / 'summary.json').write_text('{"status": "completed"}')
            return 0

    monkeypatch.setattr(module.subprocess, 'Popen', FakeProcess)
    monkeypatch.setattr(module.time, 'sleep', lambda duration: None)
    trials = [dict(method=METHODS[i % 2], seed=20261001, lr=1e-5,
                   update_clip_norm=1., result_dir=str(tmp_path / f'trial_{i}')) for i in range(7)]
    launch(trials)
    assert gpu_history == [0, 2, 3, 0, 2, 3, 0] and running == {}


def test_selection_and_final_report_use_full_trials_only(tmp_path):
    trials = []
    for method in METHODS:
        for i, seed in enumerate((20261011, 20261012, 20261013)):
            destination = tmp_path / method / str(seed)
            destination.mkdir(parents=True)
            summary = dict(method=method, status='completed', smoke=False, optimizer_steps=250,
                           noise_steps=250, seed=seed, lr=1e-5, update_clip_norm=1.,
                           final=dict(test_top1=.1 + .01 * i, test_loss=4.5, train_loss=4.4,
                                      epoch_clip_fraction=1., gdp_epsilon=8.))
            (destination / 'summary.json').write_text(json.dumps(summary))
            trials.append(dict(method=method, seed=seed, lr=1e-5, update_clip_norm=1.,
                               result_dir=str(destination)))
    write_final_report(trials, tmp_path)
    assert (tmp_path / 'aggregate.csv').exists()
    assert (tmp_path / 'final_multiseed.csv').exists()
    final_summary = json.loads((tmp_path / 'final_summary.json').read_text())
    assert final_summary['std_ddof'] == 1
    assert np.isclose(final_summary['methods'][METHODS[0]]['mean_test_top1'], .11)
    assert np.isclose(final_summary['methods'][METHODS[0]]['sample_std_test_top1'], .01)
    assert '0.1100 ± 0.0100' in (tmp_path / 'report.md').read_text()
    candidates = [dict(stage=1, status='completed', final_test_top1=.12, R=r, lr=lr,
                       test_loss=loss) for r, lr, loss in [(2., .001, .1), (1., .002, 9.), (1., .001, 99.)]]
    assert winner(candidates) is candidates[-1]
