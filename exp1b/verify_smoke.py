"""Verify only isolated smoke artifacts; never inspect formal sweep trials."""
import json
import csv
from exp1b.privacy import epsilon_from_mu, gdp_delta
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent

def main():
    summaries, orders = [], []
    for method in ('adam', 'dp_adam'):
        directory = ROOT / 'results/smoke' / method
        for name in ('config.yaml','train.log','metrics.csv','summary.json','train_order.npy','final.pt'):
            assert (directory/name).is_file()
        summary = json.loads((directory/'summary.json').read_text())
        config = yaml.safe_load((directory/'config.yaml').read_text())
        assert summary['smoke'] and summary['status']=='completed'
        assert summary['optimizer_steps']==1
        assert config['privacy']['max_grad_norm']==1.0
        metrics=list(csv.DictReader((directory/'metrics.csv').open()))
        assert len(metrics)==1 and int(metrics[0]['logical_steps'])==1
        assert summary['final']['train_examples']==1000
        assert (config['privacy']['k'],config['privacy']['b_participation'],config['total_steps'])==(5,50,250)
        assert config['physical_batch_size'] * config['gradient_accumulation'] == config['logical_batch_size'] == 1000
        checkpoint = torch.load(directory/'final.pt',map_location='cpu',weights_only=True)
        assert checkpoint['logical_steps']==1
        assert all(value.dtype==torch.float32 for value in checkpoint['model'].values())
        assert all(int(state['step'])==1 for state in checkpoint['optimizer']['state'].values())
        if method=='dp_adam':
            assert summary['noise_steps']==1 and summary['calibration']['target_mu']>0
            assert summary['final']['noise_std']>0
            calibration=summary['calibration']
            assert abs(epsilon_from_mu(calibration['target_mu'],1e-5)-8)<1e-8
            assert abs(gdp_delta(calibration['target_mu'],8)-1e-5)<1e-12
        else:
            assert not (directory/'norm_stats.csv').exists()
            assert summary['noise_steps']==0 and summary['calibration'] is None
            assert summary['final']['noise_std']==0
        if method!='adam':
            assert (directory/'norm_stats.csv').is_file()
            rows=list(csv.DictReader((directory/'norm_stats.csv').open()))
            assert len(rows)==3
            for row in rows:
                assert int(row['epoch'])==1
                for stat in ('p10','p25','p50','p75','p90','p99','mean','clip_fraction'):
                    assert float(row[stat])==summary['final_norm_stats'][row['norm_group']][stat]
            assert set(summary['final_norm_stats'])=={'full_model_norm','head_norm','backbone_norm'}
            for stats in summary['final_norm_stats'].values():
                assert 0<=stats['clip_fraction']<=1
                assert [stats[q] for q in ('p10','p25','p50','p75','p90','p99')]==sorted(stats[q] for q in ('p10','p25','p50','p75','p90','p99'))
        summaries.append(summary)
        orders.append(np.load(directory/'train_order.npy'))
    assert len({s['initialization_sha256'] for s in summaries})==1
    for order in orders:
        assert len(order)==50000
        np.testing.assert_array_equal(np.sort(order),np.arange(50000))
        np.testing.assert_array_equal(order,orders[0])
    print('2 GPU smoke trials passed: shared initialization/order, 1 logical update, DP noise once, artifacts verified.')

if __name__=='__main__':
    main()
