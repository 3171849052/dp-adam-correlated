"""Derive local tensors from data/ with cache/ tokenizer; compare historical inputs."""
import numpy as np
import torch
from exp16f import ROOT,BASE,RESULTS
from exp16f.config import file_hash,array_hash,save_json,MODEL_PATH,SPLIT_SEED
from exp16f.support import read
ASSETS=BASE/'runtime/assets'

def tokenize(raw):
    from transformers import BertTokenizer
    tokenizer=BertTokenizer.from_pretrained(str(MODEL_PATH),local_files_only=True)
    return dict(tokenizer(raw['sentence'].tolist(),padding='max_length',max_length=128,truncation=True,return_tensors='pt'),labels=torch.tensor(raw['label'].tolist()))

def prepare(official=False):
    import pandas as pd
    import pickle
    from sklearn.model_selection import train_test_split
    ASSETS.mkdir(parents=True,exist_ok=True)
    filename='nlp_official_tokens_128.pt' if official else 'nlp_tokens_128.pt'
    marker=RESULTS/('official_local_assets.json' if official else 'local_assets.json')
    if marker.exists():
        for p,h in read(marker)['files'].items():assert file_hash(ROOT/p)==h
        return
    raw=pd.read_parquet(ROOT/('data/sst2/validation.parquet' if official else 'data/sst2/train.parquet'))
    assert len(raw)==(872 if official else 67349)
    tokens=tokenize(raw)
    historical=torch.load(ROOT/'exp9/results'/filename,weights_only=True)
    assert set(tokens)==set(historical)
    for k in tokens:assert torch.equal(tokens[k],historical[k]),k
    torch.save(tokens,ASSETS/filename)
    files=[ASSETS/filename]
    if not official:
        train,valid=train_test_split(np.arange(len(raw)),train_size=62000,random_state=SPLIT_SEED,stratify=raw['label'])
        splits={'nlp':dict(train=np.random.default_rng(SPLIT_SEED+1).permutation(train).astype(np.int64),search_validation=np.sort(valid).astype(np.int64))}
        with open(ROOT/'data/cifar-100-python/train','rb') as f:labels=np.asarray(pickle.load(f,encoding='bytes')[b'fine_labels'])
        train,valid=train_test_split(np.arange(50000),train_size=45000,random_state=SPLIT_SEED,stratify=labels)
        splits['cv']=dict(train=np.sort(train).astype(np.int64),search_validation=np.sort(valid).astype(np.int64))
        for task,arrays in splits.items():
            with np.load(ROOT/f'exp9/results/{task}_split.npz') as old:
                for k,v in arrays.items():np.testing.assert_array_equal(v,old[k])
            p=ASSETS/f'{task}_split.npz';np.savez(p,**arrays);files.append(p)
    save_json(marker,dict(files={str(p.relative_to(ROOT)):file_hash(p) for p in files},tensor_sha256={k:array_hash(v.numpy()) for k,v in tokens.items()},historical_tensor_equal=True,raw_source='data/',tokenizer_source='cache/bert-tiny'))
