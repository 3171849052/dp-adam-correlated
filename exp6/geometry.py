"""Detached logical-step actual-weight geometry, computed in FP64."""
import torch
from exp6.lora import LoRALinear


@torch.no_grad()
def root_pair(gram, geom_eps):
    assert geom_eps > 0
    gram = gram.double()
    vals, vecs = torch.linalg.eigh(gram + geom_eps ** 2 * torch.eye(
        gram.shape[0], device=gram.device, dtype=torch.float64))
    assert torch.all(vals > 0)
    root = (vecs * vals.sqrt()) @ vecs.T
    inverse = (vecs * vals.rsqrt()) @ vecs.T
    return root, inverse, vals.sqrt()


class Geometry:
    @torch.no_grad()
    def __init__(self, model, geom_eps):
        self.pairs = {}
        self.layers = []
        for name, layer in model.named_modules():
            if not isinstance(layer, LoRALinear):
                continue
            A, B = layer.A.detach().double(), layer.B.detach().double()
            PA, IA, eigA = root_pair(B.T @ B, geom_eps)
            PB, IB, eigB = root_pair(A @ A.T, geom_eps)
            self.pairs[name + '.A'] = (PA.float(), IA.float(), 'left')
            self.pairs[name + '.B'] = (PB.float(), IB.float(), 'right')
            wa, wb = B @ IA, IB @ A
            identity = torch.eye(A.shape[0], device=A.device, dtype=torch.float64)
            self.layers.append(dict(layer=name, A_fro=float(A.norm()), B_fro=float(B.norm()),
                actual_weight_energy=float(B.shape[0] * B.square().sum() + A.shape[1] * A.square().sum()),
                PA_eig_min=float(eigA.min()), PA_eig_max=float(eigA.max()),
                PB_eig_min=float(eigB.min()), PB_eig_max=float(eigB.max()),
                PA_condition=float(eigA.max()/eigA.min()), PB_condition=float(eigB.max()/eigB.min()),
                B_PA_inverse_gram_error=float((wa.T @ wa - identity).norm()),
                PB_inverse_A_gram_error=float((wb @ wb.T - identity).norm())))

    def transform(self, gradients, inverse=False):
        result = {}
        for name, g in gradients.items():
            if name not in self.pairs:
                result[name] = g
            else:
                p, inv, side = self.pairs[name]
                matrix = inv if inverse else p
                result[name] = matrix @ g if side == 'left' else g @ matrix
        return result

    def diagnostics(self):
        keys = [k for k in self.layers[0] if k != 'layer']
        result = {}
        for key in keys:
            values = [r[key] for r in self.layers]
            result[key + '_mean'] = sum(values) / len(values)
            result[key + '_min'] = min(values)
            result[key + '_max'] = max(values)
        result['actual_weight_energy_sum'] = sum(r['actual_weight_energy'] for r in self.layers)
        result['A_fro_global'] = sum(r['A_fro'] ** 2 for r in self.layers) ** .5
        result['B_fro_global'] = sum(r['B_fro'] ** 2 for r in self.layers) ** .5
        return result
