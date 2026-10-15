"""Read local task assets and assign simulated columns to real parameter coordinates.

Only checkpoint structure is inspected. No model is constructed or executed.
"""
import json
import pickle
import numpy as np
import pyarrow.parquet as pq
from safetensors import safe_open
import torch
from exp13 import ROOT
from exp13.io import file_hash, save_json


def coordinate_map(parameters, k):
    names = sorted(parameters)
    sizes = [int(np.prod(parameters[n])) for n in names]
    ends = np.cumsum(sizes, dtype=np.int64)
    total = int(ends[-1])
    assert 0 < k <= total
    # Evenly spaced, deterministic, distinct real flattened parameter indices.
    flat = ((2*np.arange(k, dtype=np.int64)+1) * total) // (2*k)
    owners = np.searchsorted(ends, flat, side='right')
    starts = np.r_[0, ends[:-1]]
    return dict(parameter_count=total, parameters=parameters,
                coordinates=[dict(column=c, parameter=names[i],
                                  offset=int(g-starts[i]), global_offset=int(g))
                             for c, (i, g) in enumerate(zip(owners, flat))])


def prepare(output, k):
    cv_data = ROOT / 'data/cifar-100-python/train'
    nlp_data = ROOT / 'data/sst2/train.parquet'
    bert_path = ROOT / 'cache/bert-tiny/pytorch_model.bin'
    bert_config = ROOT / 'cache/bert-tiny/config.json'
    hub = ROOT / 'cache/huggingface/hub/models--timm--vit_tiny_patch16_224.augreg_in21k_ft_in1k'
    revision = (hub / 'refs/main').read_text().strip()
    vit_path = hub / 'snapshots' / revision / 'model.safetensors'
    assert vit_path.resolve().is_relative_to(ROOT / 'cache')
    with cv_data.open('rb') as f:
        cv = pickle.load(f, encoding='bytes')
    labels = np.asarray(cv[b'fine_labels'])
    assert cv[b'data'].shape == (50000, 3072) and len(np.unique(labels)) == 100
    nlp = pq.read_table(nlp_data, columns=['label'])
    assert nlp.num_rows == 67349 and set(nlp['label'].to_pylist()) == {0, 1}
    cv_params = {}
    with safe_open(vit_path, framework='pt', device='cpu') as f:
        for name in f.keys():
            if name.startswith('head.'):
                continue
            shape = list(f.get_slice(name).get_shape())
            if name == 'cls_token':
                name, shape = 'cls.weight', [1, shape[-1]]
            elif name == 'pos_embed':
                name, shape = 'position.weight', shape[1:]
            elif name == 'patch_embed.proj.weight':
                name, shape = 'patch.weight', [shape[0], int(np.prod(shape[1:]))]
            elif name == 'patch_embed.proj.bias':
                name = 'patch.bias'
            name = name.replace('.mlp.fc1.', '.mlp.0.').replace('.mlp.fc2.', '.mlp.2.')
            cv_params[name] = shape
    assert cv_params['patch.weight'] == [192, 768]
    assert cv_params['position.weight'] == [197, 192]
    cv_params.update({'head.weight': [100, 192], 'head.bias': [100]})
    weights = torch.load(bert_path, map_location='cpu', weights_only=True)
    nlp_params = {n: list(v.shape) for n, v in weights.items()
                  if n.startswith('bert.') and v.is_floating_point()}
    cfg = json.loads(bert_config.read_text())
    assert cfg['hidden_size'] == 128 and cfg['num_hidden_layers'] == 2
    nlp_params.update({'classifier.weight': [2, cfg['hidden_size']], 'classifier.bias': [2]})
    assert nlp_params['bert.embeddings.word_embeddings.weight'] == [30522, 128]
    result = dict(offline=True, model_constructed=False, model_forward=False,
                  backward=False, parameter_update=False, official_evaluation_loaded=False,
                  input_sha256={str(p.relative_to(ROOT)): file_hash(p)
                                for p in (cv_data, nlp_data, bert_path, bert_config, vit_path)},
                  cv=dict(dataset='CIFAR-100', examples=50000, classes=100, image_size=224,
                          model='ViT-Tiny / patch16', replacement_head='100 classes, structure only',
                          search_examples=45000, search_validation_examples=5000,
                          split_seed=20261008, **coordinate_map(cv_params, k)),
                  nlp=dict(dataset='SST-2', examples=67349, classes=2, max_length=128,
                           model='BERT-Tiny', replacement_head='2 classes, structure only',
                           train_examples=62000, internal_validation_examples=5349,
                           split_seed=20261008, **coordinate_map(nlp_params, k)),
                  horizons={'225': dict(task='cv', examples=45000, spacing=45),
                            '250': dict(task='cv', examples=50000, spacing=50),
                            '310': dict(task='nlp', examples=62000, spacing=62)},
                  coordinate_semantics='K equal-weight noise coordinates; no parameter values or data enter objectives')
    save_json(output / 'assets.json', result)
    return result
