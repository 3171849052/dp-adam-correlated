"""Frozen, invertible Muon block geometry; Adam blocks always use identity."""
import math
import torch


class Identity:
    def transform(self, g):
        return g

    def inverse(self, e):
        return e


class NormScale:
    def __init__(self, h, lambda_parallel):
        assert lambda_parallel > 0
        self.q = h / h.norm()
        self.gain = math.sqrt(lambda_parallel)

    def _apply(self, g, gain):
        parallel = (g * self.q).sum(dim=(-2, -1), keepdim=True) * self.q
        return g + (gain - 1) * parallel

    def transform(self, g):
        return self._apply(g, self.gain)

    def inverse(self, e):
        return self._apply(e, 1 / self.gain)


class SpectralScale:
    def __init__(self, h, kappa, rho):
        assert kappa >= 1 and rho > 0
        hbar = (h / h.norm()).double()
        ql, qr = hbar @ hbar.mT, hbar.mT @ hbar
        vl, ul = torch.linalg.eigh(ql)
        vr, ur = torch.linalg.eigh(qr)
        # PSD roundoff is clamped; the rank threshold defines nonzero s^2.
        vl, vr = vl.clamp_min(0), vr.clamp_min(0)
        small = vl if len(vl) <= len(vr) else vr
        threshold = torch.finfo(hbar.dtype).eps * max(h.shape) * small.max()
        self.delta = float(rho * small[small > threshold].mean())
        limit = .5 * math.log(kappa)
        gains = []
        for eigenvalues in (vl, vr):
            logs = -.25 * torch.log(eigenvalues + self.delta)
            # Median log-gain on each side: the typical pair has scale 1.
            logs = logs - torch.quantile(logs, .5)
            gains.append(logs.clamp(-limit, limit).exp())
        self.left_gain, self.right_gain = (v.to(h.dtype) for v in gains)
        self.pairwise_min = float(self.left_gain.min() * self.right_gain.min())
        self.pairwise_max = float(self.left_gain.max() * self.right_gain.max())
        self.left = ((ul * gains[0]) @ ul.mT).to(h.dtype)
        self.right = ((ur * gains[1]) @ ur.mT).to(h.dtype)
        self.left_inv = ((ul / gains[0]) @ ul.mT).to(h.dtype)
        self.right_inv = ((ur / gains[1]) @ ur.mT).to(h.dtype)

    def transform(self, g):
        return self.left @ g @ self.right

    def inverse(self, e):
        return self.left_inv @ e @ self.right_inv


@torch.no_grad()
def freeze_geometry(optimizer, method, lambda_parallel, kappa, rho):
    geometries = {p: Identity() for group in optimizer.param_groups for p in group['params']}
    for p in optimizer.muon.values():
        state = optimizer.state[p]
        if 'pre_ns' not in state or float(state['pre_ns'].norm()) == 0:
            continue
        h = state['pre_ns']
        if method == 'mf_muon_normscale' and lambda_parallel != 1:
            geometries[p] = NormScale(h, lambda_parallel)
        if method == 'mf_muon_spectralscale' and kappa != 1:
            geometries[p] = SpectralScale(h, kappa, rho)
    return geometries
