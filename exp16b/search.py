"""Only the user-specified 108 coarse trials are authorized."""
from itertools import product
from exp16b import RESULTS
from exp16b.config import Trial,METHODS,save_json
from exp16b.support import read
GRIDS=dict(cv=dict(lr=[.15,.5,1.5],clip=[1.,3.,10.],eps_scale=[.1,1.,3.]),
           nlp=dict(lr=[.05,.15,.3],clip=[5.,20.,40.],eps_scale=[.0001,.001,.01]))
def coarse_jobs():
    return [Trial(task,m,**dict(zip(GRIDS[task],v))) for task in ('cv','nlp') for v in product(*GRIDS[task].values()) for m in METHODS]
def freeze():
    value=dict(trials=[c.values() for c in coarse_jobs()],grids=GRIDS)
    p=RESULTS/'coarse_grid.json'
    if p.exists():assert read(p)==value
    else:save_json(p,value)
def authorize(cfg):
    from exp16b.stage1 import verify_acceptance
    verify_acceptance()
    assert cfg.stage=='search' and cfg.values() in read(RESULTS/'coarse_grid.json')['trials']
