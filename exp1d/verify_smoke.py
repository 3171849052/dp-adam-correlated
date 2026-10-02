"""Validate the two isolated, single-logical-step GPU smoke runs."""
import csv
import json
import numpy as np
import torch
import yaml
from exp1d.train import ROOT
from exp1d.privacy import gdp_delta


def main():
    summaries, orders = [], []
    for method in ('dp_adam_bandinvmf_momentum','dp_adam_bandinvmf_scale'):
        directory=ROOT/'exp1d/results/smoke'/method
        for name in ('config.yaml','train.log','metrics.csv','summary.json','train_order.npy','final.pt'):
            assert (directory/name).is_file()
        summary=json.loads((directory/'summary.json').read_text())
        config=yaml.safe_load((directory/'config.yaml').read_text())
        assert summary['status']=='completed' and summary['smoke']
        assert summary['optimizer_steps']==summary['noise_steps']==summary['bandinvmf_steps']==1
        assert summary['final']['train_examples']==1000
        assert (config['privacy']['k'],config['privacy']['b_participation'],config['total_steps'])==(5,50,250)
        assert config['physical_batch_size']==250 and config['gradient_accumulation']==4
        assert config['bandinvmf']['num_bands']==4
        assert config['privacy']['epsilon']==8 and config['privacy']['delta']==1e-5
        assert not config['download'] and not config['privacy']['sampling_amplification']
        assert abs(gdp_delta(summary['calibration']['target_mu'],8)-1e-5)<1e-12
        calibration=summary['calibration']
        assert abs(config['privacy']['max_grad_norm']*calibration['sensitivity']/calibration['innovation_std_sum']-calibration['target_mu'])<1e-12
        metrics=list(csv.DictReader((directory/'metrics.csv').open()))
        assert len(metrics)==1 and float(metrics[0]['strategy_sensitivity'])>0
        checkpoint=torch.load(directory/'final.pt',map_location='cpu',weights_only=True)
        assert checkpoint['logical_steps']==1
        assert all(v.dtype==torch.float32 for v in checkpoint['model'].values())
        assert all(int(s['step'])==1 for s in checkpoint['optimizer']['state'].values())
        if method.endswith('_scale'):
            for filename,key in [('norm_stats.csv','final_norm_stats'),('scale_stats.csv','final_scale_stats')]:
                rows=list(csv.DictReader((directory/filename).open())); assert len(rows)==2
                for row in rows:
                    for stat,value in summary[key][row['group']].items(): assert float(row[stat])==value
            assert summary['final_scale_stats']['sqrt_vhat']['p99']==0
            assert summary['final_scale_stats']['scale']['p50']==float(torch.tensor(config['scale']['eps_scale'], dtype=torch.float32).reciprocal())
            assert summary['final_norm_stats']['scaled_norm']['p50']>summary['final_norm_stats']['unscaled_norm']['p50']
            assert summary['final_norm_stats']['scaled_norm']['clip_fraction']==summary['final']['clip_fraction']
        summaries.append(summary); orders.append(np.load(directory/'train_order.npy'))
    assert len({s['initialization_sha256'] for s in summaries})==1
    np.testing.assert_array_equal(orders[0],orders[1])
    np.testing.assert_array_equal(np.sort(orders[0]),np.arange(50000))
    print('2 GPU smoke trials passed: shared initialization/order, 4 microbatches, one Adam update/noise output, FP32, GDP and exact scale diagnostics.')


if __name__=='__main__': main()
