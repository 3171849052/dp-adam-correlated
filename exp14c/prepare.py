"""Use raw local data and the historical tokenizer; never download."""
import pandas as pd
import torch
from exp14c import ROOT, RESULTS
from exp14c.data import tokenize
from exp14c.config import file_hash, save_json

def prepare(official=False):
    if official:
        assert (RESULTS/'frozen_manifest.json').exists()
    raw=ROOT/'data/sst2'/('validation.parquet' if official else 'train.parquet')
    target=RESULTS/('nlp_official_tokens_128.pt' if official else 'nlp_tokens_128.pt')
    tokens=tokenize(pd.read_parquet(raw))
    original=torch.load(ROOT/'exp9/results'/target.name,weights_only=True)
    assert tokens.keys()==original.keys()
    for k in tokens: torch.testing.assert_close(tokens[k],original[k],rtol=0,atol=0)
    torch.save(tokens,target)
    save_json(RESULTS/('official_assets.json' if official else 'local_tokens.json'),
              dict(raw=str(raw.relative_to(ROOT)),raw_sha256=file_hash(raw),tokens_sha256=file_hash(target),
                   token_values_exact_exp9=True,examples=len(tokens['labels'])))
if __name__=='__main__': prepare()
