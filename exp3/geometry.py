"""Frozen invertible Muon geometry; spectral centers use only active singular values."""
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
        h64 = h.double()
        hbar = h64 / h64.norm()
        u, sigma, vh = torch.linalg.svd(hbar, full_matrices=False)
        threshold = torch.finfo(h64.dtype).eps * max(h.shape) * sigma.max()
        active = sigma > threshold
        active_squared = sigma[active].square()
        self.delta = float(rho * active_squared.mean())
        self.normalization_center = float(torch.quantile(-.25 * torch.log(active_squared + self.delta), .5))
        limit = .5 * math.log(kappa)
        gains = (-.25 * torch.log(sigma.square() + self.delta) - self.normalization_center).clamp(-limit, limit).exp()
        null_gain = math.exp(max(-limit, min(limit, -.25 * math.log(self.delta) - self.normalization_center)))
        self.active_gain = gains[active].to(h.dtype)
        self.left_gain = torch.cat((gains, gains.new_full((h.shape[0] - len(sigma),), null_gain))).to(h.dtype)
        self.right_gain = torch.cat((gains, gains.new_full((h.shape[1] - len(sigma),), null_gain))).to(h.dtype)
        self.pairwise_min = float(self.left_gain.min() * self.right_gain.min())
        self.pairwise_max = float(self.left_gain.max() * self.right_gain.max())
        self.exact_identity = kappa == 1
        factors, inverses = [], []
        for basis in (u, vh.mT):
            eye = torch.eye(basis.shape[0], dtype=h64.dtype, device=h.device)
            factors.append((null_gain * eye + (basis * (gains - null_gain)) @ basis.mT).to(h.dtype))
            inverses.append((eye / null_gain + (basis * (gains.reciprocal() - 1 / null_gain)) @ basis.mT).to(h.dtype))
        self.left, self.right = factors
        self.left_inv, self.right_inv = inverses

    def transform(self, g):
        return g if self.exact_identity else self.left @ g @ self.right

    def inverse(self, e):
        return e if self.exact_identity else self.left_inv @ e @ self.right_inv


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


def snapshot_geometry(geometry):
    """Inverse factors completely specify S_t and avoid redundant matrix storage."""
    if isinstance(geometry, Identity):
        return dict(kind='identity')
    if isinstance(geometry, NormScale):
        return dict(kind='normscale', q=geometry.q.detach().cpu().clone(), gain=geometry.gain)
    assert isinstance(geometry, SpectralScale)
    return dict(kind='spectralscale', left_inv=geometry.left_inv.detach().cpu().clone(),
                right_inv=geometry.right_inv.detach().cpu().clone())


def snapshot_inverse(snapshot, e):
    if snapshot['kind'] == 'identity':
        return e
    if snapshot['kind'] == 'normscale':
        q = snapshot['q'].to(e)
        parallel = (e * q).sum(dim=(-2, -1), keepdim=True) * q
        return e + (1 / snapshot['gain'] - 1) * parallel
    assert snapshot['kind'] == 'spectralscale'
    return snapshot['left_inv'].to(e) @ e @ snapshot['right_inv'].to(e)
