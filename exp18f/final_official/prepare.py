"""Audit local inputs and inherit the Exp18e formal protocol without tuning."""
import shutil
import subprocess
import sys
import numpy as np
import torch
from exp18f.final_official.common import (BASE, ROOT, SEEDS, METHODS, BASELINES, PARAMS,
    read, save, sha, configs, calibration, load_matrix, now)
from exp18f.final_official.matrices import audit, workload


def prepare():
    assert '/envs/curve/' in sys.executable
    assert not (BASE/'protocol.json').exists()
    assert read(ROOT/'exp18f/rounded_freeze.json')['parameters'] == PARAMS
    (BASE/'logs').mkdir(exist_ok=True)
    (BASE/'matrices').mkdir(exist_ok=True)
    for suffix in ('npz', 'json'):
        shutil.copyfile(ROOT/f'exp18e/matrices/momentum-bias.{suffix}', BASE/f'matrices/momentum-bias.{suffix}')
    matrix = load_matrix('momentum-bias')
    np.testing.assert_allclose(matrix['A'], workload('momentum-bias'), atol=1e-12)
    a = audit(matrix['A'], matrix['C'], matrix['D'])
    save(BASE/'matrix_audit.json', dict(passed=True, sha256=sha(BASE/'matrices/momentum-bias.npz'), audit=a,
        source='exp18e/matrices/momentum-bias.npz', K=62, P=62, Q=5, T=310))
    formal = [dict(stage='formal', method=m, group='stpex2', seed=s, steps=310, **PARAMS)
              for s in SEEDS for m in METHODS]
    smoke = [dict(stage='smoke', method=m, group='stpex2', seed=20261110, steps=3, gpu_pair=g, **PARAMS)
             for m in METHODS for g in range(3)]
    save(BASE/'plans/formal.json', formal); save(BASE/'plans/smoke.json', smoke)
    hp = read(ROOT/'exp18e/hyperparameters.json')['methods']
    methods = {m: dict(**(hp[m] if m in BASELINES else PARAMS), beta1=.9, beta2=.999) for m in BASELINES+METHODS}
    save(BASE/'hyperparameters.json', dict(frozen=True, methods=methods,
        stpex2_freeze='exp18f/rounded_freeze.json', stpex2_freeze_sha256=sha(ROOT/'exp18f/rounded_freeze.json'),
        baseline_source='exp18e/hyperparameters.json'))
    protocol = read(ROOT/'exp18e/protocol.json')
    protocol.update(experiment='Exp18f final official', formal_trainings=10,
        methods=methods, stpex2_parameters=PARAMS,
        selection='No search, model selection, or early stopping; use the frozen parameters and final step 310.',
        historical_results='Reuse only the four Exp18e Adam/Momentum official five-seed results; train both STP methods anew.',
        gpu='Three GPUs (0,1,2), UUID isolation, <=2 processes per GPU, no DDP',
        statistics='Five paired seeds; ddof=1 and Student-t(df=4) 95% CI; all 15 algorithm pairs.',
        offline=True, training_tokens='exp18e/assets/nlp_tokens_128.pt',
        split='exp18e/assets/nlp_split.npz', official_tokens='exp18e/assets/nlp_official_tokens_128.pt',
        source_protocol_sha256=sha(ROOT/'exp18e/protocol.json'))
    save(BASE/'protocol.json', protocol)
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used',
        '--format=csv,noheader,nounits'], text=True)
    records = [r.split(',') for r in inventory.splitlines()]
    uuids = {r[0].strip(): r[1].strip() for r in records if int(r[0]) in (0,1,2)}
    assert len(uuids) == 3
    save(BASE/'scheduling.json', dict(physical_gpus=[0,1,2], gpu_uuids=uuids, inventory=inventory,
        max_per_gpu=2, max_processes=6, worker_reservation_mib=5200, headroom_mib=700))
    # Retokenize the existing local SST-2 files to check that cached tensors match.
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(str(ROOT/'cache/bert-tiny'), local_files_only=True)
    keys = ('input_ids','attention_mask','token_type_ids')
    for split, file, n in [('train','nlp_tokens_128.pt',67349), ('validation','nlp_official_tokens_128.pt',872)]:
        df = pd.read_parquet(ROOT/f'data/sst2/{split}.parquet')
        assert len(df) == n
        tokens = tokenizer(df['sentence'].tolist(), padding='max_length', truncation=True, max_length=128, return_tensors='pt')
        cached = torch.load(ROOT/f'exp18e/assets/{file}', weights_only=True)
        for key in keys: assert torch.equal(tokens[key], cached[key])
        assert torch.equal(torch.tensor(df['label'].to_numpy()), cached['labels'])
    tr, va = train_test_split(np.arange(67349), train_size=62000, random_state=20261008,
        stratify=pd.read_parquet(ROOT/'data/sst2/train.parquet')['label'].to_numpy())
    tr = np.random.default_rng(20261009).permutation(tr).astype(np.int64)
    va = np.sort(va).astype(np.int64)
    with np.load(ROOT/'exp18e/assets/nlp_split.npz') as split:
        assert np.array_equal(tr, split['train']) and np.array_equal(va, split['search_validation'])
        assert not np.intersect1d(tr, va).size
    save(BASE/'data_audit.json', dict(passed=True, train=62000, internal_validation=5349,
        official_validation=872, local_raw_retokenization_equal=True, split_equal=True, downloads=False,
        official_used_for_selection=False, python=sys.executable, time=now()))
    audits = {m: calibration(next(c for c in formal if c['method']==m)) for m in METHODS}
    for m in METHODS: assert audits[m] == read(ROOT/'exp18e/formal_privacy_audit.json')[m]
    save(BASE/'privacy_audit.json', audits)
    paths = []
    for folder in ('exp18e', 'exp18f'):
        paths += [p for p in (ROOT/folder).rglob('*') if p.is_file()
                  and BASE not in p.parents and 'runtime' not in p.parts
                  and p.suffix in ('.py','.json','.csv','.md','.sh') and 'logs' not in p.parts]
    paths += list((ROOT/'exp18e/assets').glob('*')) + list((ROOT/'cache/bert-tiny').glob('*'))
    paths += [ROOT/f'data/sst2/{s}.parquet' for s in ('train','validation')]
    save(BASE/'input_manifest.json', {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(paths)) if p.is_file()})
    save(BASE/'preflight.json', dict(passed=True, time=now(), formal_trainings=10, smoke_trainings=6,
        privacy_matches_exp18e=True, inherited_protocol=True))
    print('PASS: local data/model, frozen parameters, matrix, privacy calibration, and three GPU UUIDs', flush=True)


if __name__ == '__main__': prepare()
