"""Check recorded artifacts and actual FIFO transitions without running training."""
import csv
import json
from pathlib import Path
import yaml
from exp7 import BASE, ROOT
from exp7.config import SEED, save_json, validate
from exp7.report import rows

def audit():
    candidates = rows()
    reference = next(r for r in candidates if r['source'] == 'historical' and r.get('augmentation_trace_sha256'))
    for row in candidates:
        directory = ROOT / row['result_dir']
        assert directory.resolve().is_relative_to(BASE)
        assert row['seed'] == SEED and not row['smoke']
        for name in ('config.yaml','metrics.csv','summary.json','train_order.npy','matrices.npz','final.pt','train.log'):
            assert (directory / name).is_file(), (directory, name)
        cfg = validate(yaml.safe_load((directory / 'config.yaml').read_text()))
        assert cfg['method'] == row['method']
        assert cfg['optimizer']['lr'] == row['lr'] and cfg['privacy']['max_grad_norm'] == row['C']
        if row['eps_scale'] is not None:
            assert cfg['scale']['eps_scale'] == row['eps_scale']
        if row['status'] == 'completed':
            assert row['initialization_sha256'] == reference['initialization_sha256']
            if row['source'] == 'new':
                assert row['augmentation_trace_sha256'] == reference['augmentation_trace_sha256']
            metrics = list(csv.DictReader((directory / 'metrics.csv').open()))
            assert len(metrics) == 5 and int(metrics[-1]['epoch']) == 5
            assert int(metrics[-1]['logical_steps']) == row['optimizer_steps'] == row['noise_steps'] == 250
            assert float(metrics[-1]['test_top1']) == row['final_test_top1']
    live = {}
    starts = []
    for line in (BASE / 'results/scheduler.jsonl').read_text().splitlines():
        event = json.loads(line)
        assert event['seed'] == SEED
        if event['event'] == 'start':
            assert event['gpu'] not in live
            live[event['gpu']] = event['trial_id']
            starts.append(event)
        else:
            assert live.pop(event['gpu']) == event['trial_id']
        assert len(live) <= 3
    identities = [(r['trial_id'], r['smoke']) for r in starts]
    assert len(identities) == len(set(identities)), 'duplicate actual training'
    result = dict(status='passed', audited_trials=len(candidates), live_gpu_slots=live,
                  new_search_trials_started=sum(not r['smoke'] for r in starts),
                  smoke_trials_started=sum(r['smoke'] for r in starts),
                  historical_reused=sum(r['source'] == 'historical' for r in candidates),
                  only_search_seed=True, final_seeds_run=False)
    save_json(BASE / 'results/artifact_audit.json', result)
    return result

if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
