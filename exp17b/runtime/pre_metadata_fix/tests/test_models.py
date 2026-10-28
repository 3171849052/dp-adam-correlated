import copy
import unittest
import torch
from exp17b.cv_model import ViTTiny,pretrained_vit,checkpoint_path
from exp17b.clipping import StandardGhostModule,clipped_microbatch
def test_pretrained_vit_matches_timm_logits():
    import timm
    from safetensors.torch import load_file
    torch.manual_seed(45);model,_=pretrained_vit();model.eval()
    reference=timm.create_model('vit_tiny_patch16_224',pretrained=False,num_classes=100).eval()
    weights=load_file(str(checkpoint_path()))
    weights={k:v for k,v in weights.items() if not k.startswith('head.')}
    weights.update({'head.weight':model.head.weight,'head.bias':model.head.bias})
    reference.load_state_dict(weights,strict=True)
    images=torch.randn(2,3,224,224)
    with torch.no_grad():
        # Patch convolution vs unfolded Linear have different FP32 accumulation order.
        torch.testing.assert_close(model(images),reference(images),rtol=3e-5,atol=1e-5)
        torch.testing.assert_close(model.double()(images.double()),reference.double()(images.double()),rtol=1e-9,atol=1e-10)

def test_vit_clipping():
    torch.manual_seed(42)
    base=ViTTiny(patch_size=4,image_size=16,embed_dim=12,depth=1,heads=3,mlp_ratio=2,num_classes=3)
    model=StandardGhostModule(copy.deepcopy(base),1.3)
    x=torch.randn(3,3,16,16);y=torch.tensor([0,1,2])
    sums=[torch.zeros_like(p) for p in base.parameters()];norms=[]
    for xx,yy in zip(x,y):
        base.zero_grad();torch.nn.functional.cross_entropy(base(xx[None]),yy[None]).backward()
        norm=sum(p.grad.square().sum() for p in base.parameters()).sqrt();norms.append(norm)
        for s,p in zip(sums,base.parameters()):s.add_(p.grad*min(1.,1.3/norm.item()))
    clipped_microbatch(model,x,y,1.3)
    torch.testing.assert_close(model.last_norms,torch.stack(norms),rtol=5e-5,atol=3e-6)
    for p,s in zip(model.parameters(),sums):torch.testing.assert_close(p.grad,s,rtol=5e-5,atol=3e-6)
    model.remove_hooks()

class ModelTests(unittest.TestCase):
    def test_mapping(self):test_pretrained_vit_matches_timm_logits()
    def test_clipping(self):test_vit_clipping()
