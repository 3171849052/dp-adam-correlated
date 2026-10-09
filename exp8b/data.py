"""Shared raw assets; experiment-local split/tokens. Official validation is final-only."""
import numpy as np
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_from_disk
from transformers import BertTokenizer
import subprocess
from exp8b import ROOT, RESULTS
from exp8b.config import *


def fetch(url,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    partial=path.with_suffix(path.suffix+'.part')
    subprocess.run(['curl','--location','--fail','--retry','2','--max-time','300','--output',str(partial),url],check=True)
    partial.replace(path)


def download():
    for name in ('config.json','pytorch_model.bin','vocab.txt'):
        path=MODEL_PATH/name
        if not path.exists(): fetch(f'https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}?download=true',path)
    source=ROOT/'data/sst2/train.parquet'
    if not source.exists():
        fetch(f'https://huggingface.co/datasets/nyu-mll/glue/resolve/{DATA_REVISION}/sst2/train-00000-of-00001.parquet?download=true',source)
    assert file_hash(MODEL_PATH/'pytorch_model.bin')==CHECKPOINT_SHA256
    assert file_hash(source)==TRAIN_SHA256
    if not DATA_PATH.exists():
        ds=Dataset.from_parquet(str(source),cache_dir=str(ROOT/'data/huggingface'))
        assert len(ds)==67349
        ds.save_to_disk(str(DATA_PATH))
    save_json(RESULTS/'download_manifest.json',dict(model=MODEL_ID,model_revision=MODEL_REVISION,
              dataset_revision=DATA_REVISION,train_sha256=file_hash(source),
              model_files={n:file_hash(MODEL_PATH/n) for n in ('config.json','pytorch_model.bin','vocab.txt')},
              official_validation_used=False))


def split_indices(labels):
    train,valid=train_test_split(np.arange(len(labels)),train_size=62000,random_state=SPLIT_SEED,stratify=labels)
    train=np.random.default_rng(SPLIT_SEED+1).permutation(train)
    return train.astype(np.int64),np.sort(valid).astype(np.int64)


def tokenize(raw):
    tok=BertTokenizer.from_pretrained(str(MODEL_PATH),local_files_only=True)
    inputs=tok(list(raw['sentence']),padding='max_length',max_length=128,truncation=True,return_tensors='pt')
    return dict(inputs,labels=torch.tensor(list(raw['label'])))


def prepare():
    RESULTS.mkdir(parents=True,exist_ok=True)
    raw=load_from_disk(str(DATA_PATH))
    # Verify that a reused Arrow dataset is the pinned parquet's content.
    pinned=Dataset.from_parquet(str(ROOT/'data/sst2/train.parquet'),cache_dir=str(ROOT/'data/huggingface'))
    assert len(raw)==67349 and raw['sentence']==pinned['sentence'] and raw['label']==pinned['label']
    labels=np.asarray(raw['label'])
    train,valid=split_indices(labels)
    path=RESULTS/'split.npz'
    if path.exists():
        with np.load(path) as s:
            np.testing.assert_array_equal(s['train'],train)
            np.testing.assert_array_equal(s['search_validation'],valid)
    else: np.savez(path,train=train,search_validation=valid)
    tokens=RESULTS/'tokens_128.pt'
    if not tokens.exists(): torch.save(tokenize(raw),tokens)
    save_json(RESULTS/'data_manifest.json',dict(split_seed=SPLIT_SEED,split_sha256=file_hash(path),
              tokens_sha256=file_hash(tokens),train_examples=62000,search_validation_examples=5349,
              train_label_counts=np.bincount(labels[train]).tolist(),search_validation_label_counts=np.bincount(labels[valid]).tolist(),
              tokenizer_sha256=file_hash(MODEL_PATH/'vocab.txt'),official_validation_used=False))


def datasets():
    inputs=torch.load(RESULTS/'tokens_128.pt',weights_only=True)
    with np.load(RESULTS/'split.npz') as split:
        def subset(key):
            idx=torch.from_numpy(split[key].copy())
            return TensorDataset(*(inputs[k][idx] for k in ('input_ids','attention_mask','token_type_ids','labels')))
        return subset('train'),subset('search_validation')


def official_validation(phase):
    assert phase=='final', 'Official validation is accessible only from final trials'
    source=ROOT/'data/sst2/validation.parquet'
    if not source.exists():
        fetch(f'https://huggingface.co/datasets/nyu-mll/glue/resolve/{DATA_REVISION}/sst2/validation-00000-of-00001.parquet?download=true',source)
    raw=Dataset.from_parquet(str(source),cache_dir=str(ROOT/'data/huggingface'))
    assert len(raw)==872
    inputs=tokenize(raw)
    save_json(RESULTS/'official_validation_manifest.json',dict(phase='final',examples=872,sha256=file_hash(source),revision=DATA_REVISION))
    return TensorDataset(*(inputs[k] for k in ('input_ids','attention_mask','token_type_ids','labels')))
