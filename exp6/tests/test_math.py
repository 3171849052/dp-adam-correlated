import copy
import numpy as np
import pytest
import torch
from torch import nn
from exp6.lora import LoRALinear
from exp6.geometry import Geometry, root_pair
from exp6.clipping import per_example, global_norm, clipped_sum
from exp6.mechanism import TemporalNoise, matrices, materialize
from exp6.privacy import calibration, spent
from exp6.config import Trial, METHODS, FIXED


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = LoRALinear(nn.Linear(5, 4), rank=2, alpha=2)
        self.head = nn.Linear(4, 3)

    def forward(self, x):
        return self.head(self.layer(x).tanh())


def test_initial_weight_and_output_exact():
    torch.manual_seed(2)
    base = nn.Linear(5,4)
    lora = LoRALinear(copy.deepcopy(base))
    x = torch.randn(6,5)
    assert torch.equal(base.weight, lora.effective_weight())
    assert torch.equal(base(x), lora(x))
    assert lora.A.count_nonzero() and not lora.B.count_nonzero()
    assert not lora.base.weight.requires_grad


@pytest.mark.parametrize('zero_B', [True, False])
def test_geometry_inverse_and_snapshot(zero_B):
    model = Tiny()
    if not zero_B:
        with torch.no_grad(): model.layer.B.normal_()
    geometry = Geometry(model, .13)
    g = {n: torch.randn(7,*p.shape) for n,p in model.named_parameters() if p.requires_grad}
    recovered = geometry.transform(geometry.transform(g), inverse=True)
    for n in g:
        torch.testing.assert_close(g[n], recovered[n], rtol=2e-5, atol=2e-6)
    assert recovered['head.weight'] is g['head.weight']
    snapshot = geometry.pairs['layer.A'][0].clone()
    with torch.no_grad(): model.layer.B.add_(1)
    assert torch.equal(snapshot, geometry.pairs['layer.A'][0])
    # Check the defining square root, independent of inverse transform.
    root, inv, _ = root_pair(model.layer.B.T @ model.layer.B, .13)
    target = (model.layer.B.T @ model.layer.B).double() + .13**2 * torch.eye(2,dtype=torch.float64)
    torch.testing.assert_close(root @ root, target)
    torch.testing.assert_close(root @ inv, torch.eye(2,dtype=torch.float64))


def test_per_example_gradients_match_autograd_and_no_frozen_grads():
    model = Tiny()
    x, y = torch.randn(4,5), torch.tensor([0,1,2,0])
    g, losses, _ = per_example(model, x, y)
    assert set(g) == {n for n,p in model.named_parameters() if p.requires_grad}
    for i in range(4):
        loss = torch.nn.functional.cross_entropy(model(x[i:i+1]), y[i:i+1])
        reference = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad])
        for actual, expected in zip(g.values(), reference):
            torch.testing.assert_close(actual[i], expected)
        torch.testing.assert_close(loss, losses[i])
    assert model.layer.base.weight.grad is None


def test_global_clipping():
    g = {'A': torch.tensor([[[3.,4.]],[[0.,0.]]]), 'head': torch.tensor([[12.],[0.]])}
    q, norms, factors = clipped_sum(g, 2.)
    torch.testing.assert_close(norms, torch.tensor([13.,0.]))
    assert factors[1] == 1
    assert global_norm(q, per_sample=False) <= 2.00001
    scaled = Geometry(Tiny(), .1)
    model = Tiny()
    raw,_,_ = per_example(model, torch.randn(5,5), torch.tensor([0,1,2,0,1]))
    scaled = Geometry(model, .1).transform(raw)
    _, scaled_norm, scaled_factor = clipped_sum(scaled, .01)
    assert torch.all(scaled_norm * scaled_factor <= .010001)


def test_raw_scale_same_adam_inputs_without_clipping_noise():
    model = Tiny()
    with torch.no_grad(): model.layer.B.normal_()
    g,_,_ = per_example(model, torch.randn(6,5), torch.tensor([0,1,2,0,1,2]))
    geometry = Geometry(model, .1)
    raw, _, _ = clipped_sum(g, None)
    scaled, _, _ = clipped_sum(geometry.transform(g), None)
    recovered = geometry.transform(scaled, inverse=True)
    a, b = copy.deepcopy(model), copy.deepcopy(model)
    oa, ob = torch.optim.Adam([p for p in a.parameters() if p.requires_grad], lr=.001), torch.optim.Adam([p for p in b.parameters() if p.requires_grad], lr=.001)
    for n, p in a.named_parameters():
        if p.requires_grad: p.grad = raw[n]/6
    for n, p in b.named_parameters():
        if p.requires_grad:
            p.grad = recovered[n]/6
            torch.testing.assert_close(p.grad, raw[n]/6, rtol=2e-5, atol=2e-6)
    oa.step(); ob.step()
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=2e-5, atol=2e-6)
    for sa,sb in zip(oa.state.values(),ob.state.values()):
        for key in ('exp_avg','exp_avg_sq'):
            torch.testing.assert_close(sa[key],sb[key], rtol=2e-5,atol=2e-7)


def test_noise_iid_exact_identity_draw_count_and_shared_innovations():
    parameters = {'a': nn.Parameter(torch.zeros(2,3)), 'b': nn.Parameter(torch.zeros(4))}
    iid = TemporalNoise(parameters, [1.], 2., 31)
    band = TemporalNoise(parameters, [1., -.3, .2, .1], 3., 31)
    generator = torch.Generator().manual_seed(31)
    for step in range(7):
        output = iid.draw(); band.draw()
        assert iid.step_count == band.step_count == step+1
        for n,p in parameters.items():
            torch.testing.assert_close(output[n], torch.randn(p.shape, generator=generator)*2., rtol=0,atol=0)
        assert iid.innovation_digests[-1] == band.innovation_digests[-1]


def test_factorial_matrices_and_privacy():
    results = [matrices(Trial(m)) for m in METHODS]
    np.testing.assert_array_equal(results[0][0], [1.])
    np.testing.assert_array_equal(results[0][1], np.eye(250))
    np.testing.assert_array_equal(materialize(results[0][0], 250), np.eye(250))
    for a,b in zip(results[1],results[3]): np.testing.assert_array_equal(a,b)
    # Momentum workload is prefix convolved with beta1^t.
    expected = np.cumsum(.9**np.arange(250))
    np.testing.assert_allclose(results[1][2], expected, rtol=1e-12)
    np.testing.assert_allclose(results[1][1] @ materialize(results[1][0],250), np.eye(250),atol=2e-14)
    for method, (_,strategy,_) in zip(METHODS,results):
        trial = Trial(method)
        privacy = calibration(strategy,trial)
        assert privacy['sampling_amplification'] is False
        assert privacy['adjacency'] == 'add_remove_zero_out'
        assert abs(spent(strategy,250,trial,privacy)['epsilon']-8) < 1e-8
        assert spent(strategy,3,trial,privacy)['epsilon'] < 8
