"""Download train only. Tokenized inputs and split artifacts belong to Exp8a."""
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_from_disk
from transformers import BertTokenizer
import json
import subprocess
from exp8a import BASE
from exp8a import ROOT, RESULTS
from exp8a.config import DATA_PATH, MODEL_PATH, MODEL_ID, MODEL_REVISION, DATA_REVISION, CHECKPOINT_SHA256, TRAIN_SHA256, SPLIT_SEED, LENGTHS, save_json, file_hash


def fetch(url, path):
    path.parent.mkdir(parents=True,exist_ok=True)
    partial = path.with_suffix(path.suffix+'.part')
    subprocess.run(['curl','--location','--fail','--retry','2','--max-time','300',
                    '--output',str(partial),url],check=True)
    partial.replace(path)


def download():
    # curl uses the host proxy correctly; all assets come from the official Hub.
    revision = MODEL_REVISION
    files = ('config.json','pytorch_model.bin','vocab.txt')
    for name in files:
        path = MODEL_PATH/name
        if not path.exists():
            fetch(f'https://huggingface.co/{MODEL_ID}/resolve/{revision}/{name}?download=true',path)
    source = ROOT/'data/sst2/train.parquet'
    dataset_revision = DATA_REVISION
    if not source.exists():
        fetch(f'https://huggingface.co/datasets/nyu-mll/glue/resolve/{dataset_revision}/sst2/train-00000-of-00001.parquet?download=true',source)
    assert file_hash(MODEL_PATH/'pytorch_model.bin') == CHECKPOINT_SHA256
    assert file_hash(source) == TRAIN_SHA256
    if not DATA_PATH.exists():
        ds = Dataset.from_parquet(str(source),cache_dir=str(ROOT/'data/huggingface'))
        assert len(ds) == 67349
        ds.save_to_disk(str(DATA_PATH))
    save_json(RESULTS/'download_manifest.json',dict(model=MODEL_ID,model_revision=revision,
              dataset='nyu-mll/glue/sst2',dataset_revision=dataset_revision,
              model_files={name:file_hash(MODEL_PATH/name) for name in files},
              dataset_train_sha256=file_hash(source),official_validation_downloaded=False))


def split_indices(labels):
    train, valid = train_test_split(np.arange(len(labels)),train_size=62000,
                                    random_state=SPLIT_SEED,stratify=labels)
    # One fixed permutation, independent of run seed and physical partition.
    train = np.random.default_rng(SPLIT_SEED+1).permutation(train)
    return train.astype(np.int64), np.sort(valid).astype(np.int64)


def prepare():
    raw = load_from_disk(str(DATA_PATH))
    assert len(raw) == 67349
    labels = np.asarray(raw['label'])
    train, valid = split_indices(labels)
    np.savez(RESULTS/'split.npz',train=train,search_validation=valid)
    tok = BertTokenizer.from_pretrained(str(MODEL_PATH),local_files_only=True)
    tok.save_pretrained(str(MODEL_PATH))
    texts = list(raw['sentence'])
    lengths = np.array([len(ids) for ids in tok(texts,truncation=False,padding=False)['input_ids']])
    stats = dict(count=len(raw), includes_special_tokens=True,
                 percentiles={f'P{p}':float(np.percentile(lengths,p)) for p in (50,90,95,99)},
                 truncation_ratio={str(n):float((lengths>n).mean()) for n in LENGTHS},
                 split_sha256=file_hash(RESULTS/'split.npz'),
                 train_label_counts=np.bincount(labels[train]).tolist(),
                 search_validation_label_counts=np.bincount(labels[valid]).tolist(),
                 official_validation_used=False)
    save_json(RESULTS/'token_lengths.json',stats)
    for n in LENGTHS:
        path = RESULTS/f'tokens_{n}.pt'
        if not path.exists():
            inputs = tok(texts,padding='max_length',max_length=n,truncation=True,return_tensors='pt')
            torch.save(dict(inputs,labels=torch.tensor(labels)),path)
    return stats


def datasets(max_length):
    inputs = torch.load(RESULTS/f'tokens_{max_length}.pt',weights_only=True)
    with np.load(RESULTS/'split.npz') as split:
        def subset(key):
            idx = torch.from_numpy(split[key].copy())
            return TensorDataset(*(inputs[k][idx] for k in ('input_ids','attention_mask','token_type_ids','labels')))
        return subset('train'), subset('search_validation')
