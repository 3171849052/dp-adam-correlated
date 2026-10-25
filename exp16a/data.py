"""Offline assets and immutable internal splits; official evaluation is formal-only."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset
from exp16a import ROOT
RESULTS = ROOT / 'exp9/results'
from torchvision.datasets import CIFAR100
from torchvision.datasets.utils import check_integrity

class OfflineCIFAR100(CIFAR100):
    """Check only the requested split; torchvision otherwise hashes test during search."""
    def _check_integrity(self):
        files=self.train_list if self.train else self.test_list
        return all(check_integrity(str(Path(self.root)/self.base_folder/name),md5) for name,md5 in files)

def cv_datasets(cfg, pretrained_cfg):
    from torchvision import transforms
    from torch.utils.data import Subset
    normalize = transforms.Normalize(pretrained_cfg['mean'], pretrained_cfg['std'])
    bicubic = transforms.InterpolationMode.BICUBIC
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=bicubic),
                                         transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    eval_transform = transforms.Compose([transforms.Resize(int(224/pretrained_cfg['crop_pct']), interpolation=bicubic),
                                        transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    train = OfflineCIFAR100(ROOT/'data', train=True, transform=train_transform, download=False)
    with np.load(RESULTS/'cv_split.npz') as split:
        indices = np.arange(50000) if cfg.formal else split['train'].copy()
        internal = split['search_validation'].copy()
    order = indices[torch.randperm(len(indices), generator=torch.Generator().manual_seed(cfg.seed)).numpy()]
    # No official test access in any smoke, including the formal-horizon smoke.
    if cfg.stage=='final':
        valid = OfflineCIFAR100(ROOT/'data', train=False, transform=eval_transform, download=False)
    else:
        valid = Subset(OfflineCIFAR100(ROOT/'data', train=True, transform=eval_transform, download=False), internal.tolist())
    return Subset(train, order.tolist()), valid, order

def nlp_datasets(cfg):
    inputs = torch.load(RESULTS/'nlp_tokens_128.pt', weights_only=True)
    keys = ('input_ids','attention_mask','token_type_ids','labels')
    with np.load(RESULTS/'nlp_split.npz') as split:
        train = split['train'].copy(); valid = split['search_validation'].copy()
    order = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(cfg.seed)).numpy()]
    def subset(indices): return TensorDataset(*(inputs[k][torch.from_numpy(indices)] for k in keys))
    training, validation = subset(order), subset(valid)
    if cfg.stage=='final':
        # Stage2 prepares this once, serially, after freezing; never downloads.
        official = torch.load(RESULTS/'nlp_official_tokens_128.pt', weights_only=True)
        validation = TensorDataset(*(official[k] for k in keys))
        if len(validation)!=872:raise ValueError('Official NLP example count changed')
    return training, validation, order

