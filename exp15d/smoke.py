"""Real CV/NLP acceptance at original eps and both wide-grid endpoints."""
from dataclasses import replace
import json
from exp15d import RESULTS
from exp15d.config import Trial, METHODS, inherited_eps, save_json
from exp15d.scheduler import run_batch
from exp15d.support import verify_assets, code_hashes
from exp15d.audit import check_protected


def jobs():
    trials=[Trial(t,m,.5 if t=='cv' else .20 if m=='stpex2' else .15,clip=3. if t=='cv' else 20.,eps_scale=eps)
            for t in ('cv','nlp') for m in METHODS for eps in (inherited_eps(t),1e-8,1.)]
    trials += [replace(c,smoke_phase='final') for c in trials if c.task=='cv' and c.eps_scale==inherited_eps('cv')]
    return trials


def main():
    assets=verify_assets();rows=run_batch('smoke',jobs())
    for r in rows:
        if not r['finite'] or r['noise_draws']!=2 or r['optimizer_steps']!=2 or r['official_evaluation_used']: raise ValueError('Smoke failed')
        from exp15d.search import cfg_of
        path=cfg_of(r).output
        first=json.loads((path/'diagnostics.jsonl').read_text().splitlines()[0])
        if abs(first['denominator_mean']/r['eps_scale']-1)>1e-6: raise ValueError('eps not propagated')
    for task in ('cv','nlp'):
        group=[r for r in rows if r['task']==task and r['smoke_phase']=='search']
        for key in ('initialization_sha256','classifier_initialization_sha256','train_order_sha256','rng_trace','noise_rng_start','noise_rng_end'):
            if any(r[key]!=group[0][key] for r in group): raise ValueError(f'Unpaired smoke RNG: {task}/{key}')
    check_protected()
    save_json(RESULTS/'phase1_acceptance.json',dict(status='passed',runs=len(rows),assets=assets,
        code_sha256=code_hashes(),official_evaluation_used=False,historical_files_unchanged=True))
    print(f'{len(rows)} real-model smoke trials passed',flush=True)


if __name__=='__main__':main()
