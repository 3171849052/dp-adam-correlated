"""Raw-gradient Adam -> global direction clipping -> noise -> learning rate.

No torch.optim.Adam.step, per-example gradients, or noisy moment updates.
"""
import math
import torch


class UCAdam:
    def __init__(self, parameters, lr, update_clip_norm, noise,
                 beta1=.9, beta2=.999, eps=1e-8):
        self.parameters = list(parameters)
        assert all(p.requires_grad and p.dtype == torch.float32 for p in self.parameters)
        assert lr > 0 and (update_clip_norm is None or update_clip_norm > 0)
        self.lr, self.update_clip_norm = lr, update_clip_norm
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.noise = noise
        # Fixed FP64 raw moments avoid overflow in the mandated large-R search.
        # Parameters, accumulated gradients, clipped directions and noise are FP32.
        self.m = [torch.zeros_like(p, dtype=torch.float64) for p in self.parameters]
        self.v = [torch.zeros_like(p, dtype=torch.float64) for p in self.parameters]
        self.step_count = 0
        self.num_parameters = sum(p.numel() for p in self.parameters)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.step_count += 1
        t = self.step_count
        directions, vhats = [], []
        # Ordinary logical-minibatch mean gradients are the ONLY input to m/v.
        for p, m, v in zip(self.parameters, self.m, self.v):
            assert p.grad is not None
            gradient = p.grad.double()
            m.mul_(self.beta1).add_(gradient, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(gradient, gradient, value=1 - self.beta2)
            vhat = v / (1 - self.beta2 ** t)
            directions.append(((m / (1 - self.beta1 ** t)) / (vhat.sqrt() + self.eps)).float())
            vhats.append(vhat)
        raw_norm = float(torch.stack([u.square().sum() for u in directions]).sum().sqrt())
        scale = 1.0 if self.update_clip_norm is None or raw_norm == 0 else min(1.0, self.update_clip_norm / raw_norm)
        for u in directions:
            u.mul_(scale)
        clipped_norm = float(torch.stack([q.square().sum() for q in directions]).sum().sqrt())
        noise_std = self.noise.marginal_std(t - 1)
        noises = self.noise.next()
        noise_norm = float(torch.stack([n.square().sum() for n in noises]).sum().sqrt())
        # LR and negative sign appear ONLY here, after clip and noise.
        for p, q, n in zip(self.parameters, directions, noises):
            p.add_(q + n, alpha=-self.lr)
        root_dim = math.sqrt(self.num_parameters)
        update_rms = raw_norm / root_dim
        signal_rms = clipped_norm / root_dim
        return dict(step=t, raw_update_norm=raw_norm, clipped_update_norm=clipped_norm,
                    clip_scale=scale, clip_indicator=int(scale < 1),
                    update_rms=update_rms, clipped_update_rms=signal_rms,
                    noise_marginal_std=noise_std,
                    parameter_noise_std=self.lr * noise_std,
                    signal_parameter_norm=self.lr * clipped_norm,
                    noise_signal_rms_ratio=noise_std / signal_rms if signal_rms > 0 else None,
                    realized_noise_rms=noise_norm / root_dim,
                    realized_noise_signal_rms_ratio=noise_norm / clipped_norm if clipped_norm > 0 else None,
                    adam_m_norm=float(torch.stack([m.double().square().sum() for m in self.m]).sum().sqrt()),
                    vhat_min=float(torch.stack([v.min() for v in vhats]).min()),
                    vhat_max=float(torch.stack([v.max() for v in vhats]).max()),
                    vhat_mean=float(torch.stack([v.double().sum() for v in vhats]).sum()) / self.num_parameters,
                    vhat_rms=float(torch.stack([v.double().square().sum() for v in vhats]).sum().sqrt()) / root_dim)

    def state_dict(self):
        return dict(m=self.m, v=self.v, step_count=self.step_count,
                    lr=self.lr, update_clip_norm=self.update_clip_norm,
                    beta1=self.beta1, beta2=self.beta2, eps=self.eps)
