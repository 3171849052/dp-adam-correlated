"""Offline assets and immutable internal splits; official evaluation is formal-only."""
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from exp14a import ROOT
RESULTS = ROOT / 'exp9/results'
from exp14a.config import MODEL_PATH, SPLIT_SEED, file_hash, array_hash, save_json

def tokenize(raw):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    inputs = tokenizer(raw['sentence'].tolist(), padding='max_length', max_length=128,
                       truncation=True, return_tensors='pt')
    return dict(inputs, labels=torch.tensor(raw['label'].tolist()))

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

def nlp_datasets(cfg):
    inputs = torch.load(RESULTS/'nlp_tokens_128.pt', weights_only=True)
    keys = ('input_ids','attention_mask','token_type_ids','labels')
    with np.load(RESULTS/'nlp_split.npz') as split:
        train = split['train'].copy(); valid = split['search_validation'].copy()
    order = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(cfg.seed)).numpy()]
    def subset(indices): return TensorDataset(*(inputs[k][torch.from_numpy(indices)] for k in keys))
    training, validation = subset(order), subset(valid)
    if cfg.stage in ('final','sweep'):
        # Stage2 prepares this once, serially, after freezing; never downloads.
        official = torch.load(RESULTS/'nlp_official_tokens_128.pt', weights_only=True)
        validation = TensorDataset(*(official[k] for k in keys))
        assert len(validation) == 872
    return training, validation, order

