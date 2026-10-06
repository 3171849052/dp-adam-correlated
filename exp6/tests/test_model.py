import torch
from exp2.model import pretrained_vit, initialization_digest
from exp6.runtime import configure
configure()
from exp6.lora import inject
from exp6.config import METHODS


def test_full_pretrained_initialization_and_four_cell_digests():
    torch.set_num_threads(2)
    digests, orders = [], []
    for _ in METHODS:
        torch.manual_seed(91)
        model, _ = pretrained_vit()
        x = torch.randn(1,3,224,224)
        with torch.no_grad(): baseline = model(x)
        layers = inject(model)
        with torch.no_grad(): injected = model(x)
        assert torch.equal(injected, baseline)
        assert len(layers) == 48
        assert all(torch.equal(layer.effective_weight(),layer.base.weight) for layer in layers.values())
        assert len([p for p in model.parameters() if p.requires_grad]) == 98
        digests.append(initialization_digest(model))
        orders.append(torch.randperm(50000,generator=torch.Generator().manual_seed(91)))
    assert len(set(digests)) == 1
    assert all(torch.equal(o,orders[0]) for o in orders)
