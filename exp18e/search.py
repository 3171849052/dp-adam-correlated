"""Paired STP-Ex2 search; official validation is absent from every ranking."""
import itertools
import math
from exp18e.common import (BASE,GROUPS,SEARCH_STAGES,read,save,write_csv,candidate_id,
    make_config,outdir,completed,sha,now,configs,calibration)
PARAMS=('lr','clip','eps')


def coarse_candidates():
    spec=read(BASE/'search_space.json')
    candidates=[dict(zip(PARAMS,x)) for x in itertools.product(*(spec['coarse'][p] for p in PARAMS))]
    candidates.append(spec['historical'])
    assert len({candidate_id(p) for p in candidates})==121
    return candidates

def stage_plan(stage,candidates,seeds):
    jobs=[make_config(stage,m,s,p) for p in candidates for s in seeds for m in GROUPS['stpex2']]
    save(BASE/'plans'/f'{stage}.json',jobs)
    save(BASE/'search'/f'{stage}_candidates.json',candidates)
    save(BASE/'search'/f'{stage}_calibration.json', {candidate_id(p):{m:calibration(make_config(stage,m,seeds[0],p))
        for m in GROUPS['stpex2']} for p in candidates})
    return jobs

def scores(stages):
    records={};pairing={}
    for stage in stages:
        for cfg in configs(stage):
            result=completed(cfg); assert result,f'Unfinished search: {outdir(cfg)}'
            if cfg['seed'] in pairing: assert result['pairing']==pairing[cfg['seed']],f'Unpaired search seed {cfg["seed"]}'
            pairing[cfg['seed']]=result['pairing']
            assert result['evaluation']=='internal_validation' and result['metrics']['examples']==5349
            params={p:cfg[p] for p in PARAMS}; cid=candidate_id(params)
            row=records.setdefault(cid,dict(candidate=cid,parameters=params,accuracy=[],loss=[],trials=[]))
            row['accuracy'].append(result['metrics']['accuracy']);row['loss'].append(result['metrics']['loss'])
            row['trials'].append(str(outdir(cfg).relative_to(BASE)))
    ranked=[dict(candidate=r['candidate'],parameters=r['parameters'],
                 mean_accuracy=sum(r['accuracy'])/len(r['accuracy']),
                 mean_loss=sum(r['loss'])/len(r['loss']),trials=r['trials']) for r in records.values()]
    ranked.sort(key=lambda r:(-r['mean_accuracy'],r['mean_loss'],r['candidate']))
    return ranked

def fine_candidates(ranked,previous):
    spec=read(BASE/'search_space.json');seen={candidate_id(p) for p in previous};result=[]
    grids={p:sorted({c[p] for c in previous}) for p in PARAMS}
    for winner in ranked[:2]:
        for p in PARAMS:
            params=winner['parameters'];values=grids[p];index=values.index(params[p])
            for j in (index-1,index+1):
                if not 0<=j<len(values):continue
                candidate=dict(params);candidate[p]=math.sqrt(params[p]*values[j]);cid=candidate_id(candidate)
                if cid not in seen: result.append(candidate);seen.add(cid)
    return result[:spec['fine_max']]

def boundary_candidates(ranked,previous):
    spec=read(BASE/'search_space.json');result=[];best=ranked[0]['parameters']
    seen={candidate_id(c) for c in previous}
    for p in PARAMS:
        values=[c[p] for c in previous]
        for edge,factor in ((min(values),.5),(max(values),2.)):
            if best[p]!=edge:continue
            c=dict(best);c[p]*=factor
            if candidate_id(c) not in seen:result.append(c);seen.add(candidate_id(c))
    return result[:spec['boundary_max']]

def search_records():
    rows=[]
    for stage in SEARCH_STAGES:
        if not (BASE/'plans'/f'{stage}.json').exists():continue
        for cfg in configs(stage):
            result=read(outdir(cfg)/'summary.json') if (outdir(cfg)/'complete.json').exists() else None
            cid=candidate_id({p:cfg[p] for p in PARAMS})
            noise=read(BASE/'search'/f'{stage}_calibration.json')[cid][cfg['method']]
            rows.append(dict(stage=stage,candidate=candidate_id({p:cfg[p] for p in PARAMS}),
                method=cfg['method'],seed=cfg['seed'],**{p:cfg[p] for p in PARAMS},
                status='complete' if result else 'pending',
                accuracy=result['metrics']['accuracy'] if result else None,
                loss=result['metrics']['loss'] if result else None,
                sigma_sum=noise['sigma_sum'],
                directory=str(outdir(cfg).relative_to(BASE))))
    write_csv(BASE/'search/records.csv',rows,list(rows[0]))
    save(BASE/'search/counts.json',dict(planned_trainings=len(rows),completed_trainings=sum(r['status']=='complete' for r in rows),
        per_stage={s:sum(r['stage']==s for r in rows) for s in SEARCH_STAGES}))

def run_search():
    from exp18e.scheduler import run
    coarse=coarse_candidates();run('coarse');ranked=scores(['coarse'])
    save(BASE/'search/coarse_ranking.json',ranked)
    fine=fine_candidates(ranked,coarse);stage_plan('fine',fine,[20261102]);run('fine')
    ranked=scores(['coarse','fine']);save(BASE/'search/fine_ranking.json',ranked)
    boundary=boundary_candidates(ranked,coarse+fine);stage_plan('boundary',boundary,[20261102]);run('boundary')
    ranked=scores(['coarse','fine','boundary']);save(BASE/'search/boundary_ranking.json',ranked)
    candidates=[r['parameters'] for r in ranked[:2]]
    stage_plan('review',candidates,[20261103,20261104,20261105]);run('review')
    reviewed=scores(['review']);save(BASE/'search/review_ranking.json',reviewed)
    frozen=dict(frozen=True,time=now(),parameters=reviewed[0]['parameters'],ranking=reviewed,
        primary='review-only mean final internal accuracy across two methods and three seeds',
        secondary='review-only mean final internal loss',official_validation_used=False,
        plans={s:sha(BASE/'plans'/f'{s}.json') for s in SEARCH_STAGES})
    save(BASE/'selection_freeze.json',frozen)
    methods=dict(read(BASE/'protocol.json')['methods'])
    methods.update({m:frozen['parameters'] for m in GROUPS['stpex2']})
    save(BASE/'hyperparameters.json',dict(frozen=True,methods=methods,selection_sha256=sha(BASE/'selection_freeze.json')))
    formal=configs();assert len(formal)==30
    save(BASE/'plans/formal.json',formal)
    write_csv(BASE/'trial_plan.csv',formal,list(formal[0]))
    save(BASE/'formal_privacy_audit.json',{m:calibration(next(c for c in formal if c['method']==m)) for m in methods})
    frozen_paths=['selection_freeze.json','hyperparameters.json','plans/formal.json','formal_privacy_audit.json','trial_plan.csv']
    frozen_paths += [f'plans/{s}.json' for s in SEARCH_STAGES]+[f'search/{s}_ranking.json' for s in SEARCH_STAGES]
    save(BASE/'freeze_manifest.json',{rel:sha(BASE/rel) for rel in frozen_paths})
    search_records()
    print(f'FROZEN shared STP-Ex2 parameters: {frozen["parameters"]}; 30 new formal trainings',flush=True)
