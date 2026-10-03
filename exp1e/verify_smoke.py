"""Validate actual four-method GPU smoke artifacts."""
import csv
import json
import numpy as np
import torch
import yaml
from exp1e.sweep import ROOT, METHODS, FINAL_SEEDS, SCALE


def main():
    summaries,orders=[],[]
    for method in METHODS:
        directory=ROOT/'exp1e/results/smoke'/method/f'seed_{FINAL_SEEDS[0]}'
        for name in ('config.yaml','train.log','metrics.csv','summary.json','train_order.npy','final.pt'):
            assert (directory/name).is_file()
        s=json.loads((directory/'summary.json').read_text())
        c=yaml.safe_load((directory/'config.yaml').read_text())
        assert s['status']=='completed' and s['smoke'] and s['optimizer_steps']==1
        assert s['noise_steps']==(0 if method=='adam' else 1)
        assert s['bandinvmf_steps']==int('bandinvmf' in method)
        assert s['final']['train_examples']==1000
        assert (c['physical_batch_size'],c['logical_batch_size'],c['gradient_accumulation'])==(250,1000,4)
        assert (c['privacy']['k'],c['privacy']['b_participation'],c['total_steps'])==(5,50,250)
        assert c['bandinvmf']['num_bands']==4 and not c['download']
        assert not c['privacy']['sampling_amplification']
        checkpoint=torch.load(directory/'final.pt',map_location='cpu',weights_only=True)
        assert checkpoint['logical_steps']==1
        assert all(v.dtype==torch.float32 for v in checkpoint['model'].values())
        assert all(int(v['step'])==1 for v in checkpoint['optimizer']['state'].values())
        if method==SCALE:
            for filename,key in [('norm_stats.csv','final_norm_stats'),('scale_stats.csv','final_scale_stats')]:
                rows=list(csv.DictReader((directory/filename).open())); assert len(rows)==2
                for row in rows:
                    for stat,value in s[key][row['group']].items(): assert float(row[stat])==value
            assert s['final_scale_stats']['sqrt_vhat']['p99']==0
            assert s['final_scale_stats']['scale']['p50']==10
            assert s['final_norm_stats']['scaled_norm']['clip_fraction']==s['final']['clip_fraction']
        summaries.append(s); orders.append(np.load(directory/'train_order.npy'))
    assert len({s['initialization_sha256'] for s in summaries})==1
    for order in orders:
        np.testing.assert_array_equal(order,orders[0])
        np.testing.assert_array_equal(np.sort(order),np.arange(50000))
    print('4 GPU smoke trials passed: identical initialization/permutation, FP32 full finetuning, 4 physical batches per Adam update, one noise output for private methods, frozen previous-vhat diagnostics.')


if __name__=='__main__': main()
