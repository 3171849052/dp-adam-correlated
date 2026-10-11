"""Offline assets and immutable internal splits; official evaluation is formal-only."""
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from exp10 import ROOT, RESULTS
from exp10.config import MODEL_PATH, SPLIT_SEED, file_hash, array_hash, save_json

def tokenize(raw):
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    inputs = tokenizer(raw['sentence'].tolist(), padding='max_length', max_length=128,
                       truncation=True, return_tensors='pt')
    return dict(inputs, labels=torch.tensor(raw['label'].tolist()))

def prepare():
    import pandas as pd
    from exp10.cv_model import checkpoint_path
    RESULTS.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(ROOT/'data/sst2/train.parquet')
    assert len(raw) == 67349
    assert file_hash(ROOT/'data/sst2/train.parquet') == '66a253e67968acfabcbe49dbe9da964b42ac1c851c40ab760e8c8942efdb3229'
    assert file_hash(MODEL_PATH/'pytorch_model.bin') == 'dab2c2bddcfb48ea430ef63fd76d46d67d704487844d967256a50dd7d7fd0a66'
    train, valid = train_test_split(np.arange(len(raw)), train_size=62000,
                                   random_state=SPLIT_SEED, stratify=raw['label'])
    train = np.random.default_rng(SPLIT_SEED+1).permutation(train).astype(np.int64)
    splits = {'nlp': dict(train=train, search_validation=np.sort(valid).astype(np.int64))}
    # Match Exp8b's published fixed split exactly, reading it without modification.
    with np.load(ROOT/'exp8b/results/split.npz') as historical:
        for key, value in splits['nlp'].items(): np.testing.assert_array_equal(value, historical[key])
    with open(ROOT/'data/cifar-100-python/train','rb') as f: labels = np.asarray(pickle.load(f,encoding='bytes')[b'fine_labels'])
    train, valid = train_test_split(np.arange(50000), train_size=45000,
                                   random_state=SPLIT_SEED, stratify=labels)
    splits['cv'] = dict(train=np.sort(train).astype(np.int64), search_validation=np.sort(valid).astype(np.int64))
    for task in ('cv','nlp'):
        with np.load(ROOT/f'exp9/results/{task}_split.npz') as reference:
            for key,value in splits[task].items(): np.testing.assert_array_equal(value,reference[key])
    split_meta = {}
    for task, values in splits.items():
        path = RESULTS/f'{task}_split.npz'
        if path.exists():
            with np.load(path) as existing:
                for key, value in values.items(): np.testing.assert_array_equal(value, existing[key])
        else: np.savez(path, **values)
        split_meta[task] = dict(file_sha256=file_hash(path), indices_sha256={k:array_hash(v) for k,v in values.items()},
                                examples={k:len(v) for k,v in values.items()})
    tokens = RESULTS/'nlp_tokens_128.pt'
    if not tokens.exists(): torch.save(tokenize(raw), tokens)
    assets = dict(cv_pretrained=file_hash(checkpoint_path()),
                  nlp_pretrained={n:file_hash(MODEL_PATH/n) for n in ('pytorch_model.bin','config.json','vocab.txt')},
                  cifar_train=file_hash(ROOT/'data/cifar-100-python/train'),
                  sst2_train=file_hash(ROOT/'data/sst2/train.parquet'),
                  tokens_sha256=file_hash(tokens), split_seed=SPLIT_SEED, splits=split_meta,
                  official_evaluation_loaded=False, offline=True)
    save_json(RESULTS/'assets_manifest.json', assets)
    return assets

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

def prepare_official():
    import pandas as pd
    path = RESULTS/'nlp_official_tokens_128.pt'
    raw = pd.read_parquet(ROOT/'data/sst2/validation.parquet')
    assert len(raw) == 872
    if not path.exists(): torch.save(tokenize(raw), path)
    save_json(RESULTS/'official_assets.json', dict(sst2_validation_sha256=file_hash(ROOT/'data/sst2/validation.parquet'),
             tokens_sha256=file_hash(path), cifar_test_sha256=file_hash(ROOT/'data/cifar-100-python/test')))
