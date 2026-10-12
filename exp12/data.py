"""Offline CIFAR-100; exact Exp9 CV split and loader."""
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from exp12 import ROOT, RESULTS
from exp12.config import SPLIT_SEED, file_hash, array_hash, save_json

def cv_datasets(cfg, pretrained_cfg):
    from torchvision import datasets, transforms
    from torch.utils.data import Subset
    normalize = transforms.Normalize(pretrained_cfg['mean'], pretrained_cfg['std'])
    bicubic = transforms.InterpolationMode.BICUBIC
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224, interpolation=bicubic),
                                         transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    eval_transform = transforms.Compose([transforms.Resize(int(224/pretrained_cfg['crop_pct']), interpolation=bicubic),
                                        transforms.CenterCrop(224), transforms.ToTensor(), normalize])
    train = datasets.CIFAR100(ROOT/'data', train=True, transform=train_transform, download=False)
    with np.load(RESULTS/'cv_split.npz') as split:
        indices = np.arange(50000) if cfg.formal else split['train'].copy()
        internal = split['search_validation'].copy()
    order = indices[torch.randperm(len(indices), generator=torch.Generator().manual_seed(cfg.seed)).numpy()]
    # No official test access in any smoke, including the formal-horizon smoke.
    if cfg.stage in ('final','sweep'):
        valid = datasets.CIFAR100(ROOT/'data', train=False, transform=eval_transform, download=False)
    else:
        valid = Subset(datasets.CIFAR100(ROOT/'data', train=True, transform=eval_transform, download=False), internal.tolist())
    return Subset(train, order.tolist()), valid, order


def prepare():
    from exp12.cv_model import checkpoint_path
    with (ROOT/'data/cifar-100-python/train').open('rb') as f:
        labels=np.asarray(pickle.load(f,encoding='bytes')[b'fine_labels'])
    train,valid=train_test_split(np.arange(50000),train_size=45000,random_state=SPLIT_SEED,stratify=labels)
    split=dict(train=np.sort(train).astype(np.int64),search_validation=np.sort(valid).astype(np.int64))
    with np.load(ROOT/'exp9/results/cv_split.npz') as reference:
        for key,value in split.items(): np.testing.assert_array_equal(value,reference[key])
    np.savez(RESULTS/'cv_split.npz',**split)
    assets=dict(cv_pretrained=file_hash(checkpoint_path()),cifar_train=file_hash(ROOT/'data/cifar-100-python/train'),
                split_seed=SPLIT_SEED,splits={'cv':dict(file_sha256=file_hash(RESULTS/'cv_split.npz'),
                indices_sha256={k:array_hash(v) for k,v in split.items()},examples={k:len(v) for k,v in split.items()})},
                official_evaluation_loaded=False,offline=True)
    save_json(RESULTS/'assets_manifest.json',assets)
    return assets


def prepare_official():
    save_json(RESULTS/'official_assets.json',dict(cifar_test_sha256=file_hash(ROOT/'data/cifar-100-python/test')))
