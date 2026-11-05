import itertools,math
from exp18.common import GROUPS,digest
GRIDS={'adam':dict(lr=[1e-4,3e-4,1e-3,3e-3,1e-2],clip=[.25,.5,1,2,4]),'momentum':dict(lr=[1e-3,3e-3,1e-2,3e-2,1e-1],clip=[.25,.5,1,2,4]),'stpex2':dict(lr=[1e-4,3e-4,1e-3,3e-3,1e-2],clip=[3,10,30,100],eps=[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2])}
def coarse():return {g:[dict(zip(grid,v)) for v in itertools.product(*grid.values())] for g,grid in GRIDS.items()}
def trials(candidates,stage,seeds):
    return [dict(group=g,method=m,stage=stage,seed=s,**hp) for g,configs in candidates.items() for hp in configs for s in seeds for m in GROUPS[g]]
def key(cfg):return digest(cfg)[:20]
def hp(cfg):return {k:cfg[k] for k in GRIDS[cfg['group']]}
def rank(rows,g):
    grouped={}
    for row in rows:
        c=row['config']
        if c['group']!=g:continue
        h=hp(c);entry=grouped.setdefault(digest(h),[h,[]]);entry[1].append(row['metrics'])
    return [a[0] for a in sorted(grouped.values(),key=lambda a:(-sum(x['accuracy'] for x in a[1])/len(a[1]),sum(x['loss'] for x in a[1])/len(a[1]),digest(a[0])))]
def fine(rows):
    result={}
    for g,budget in [('adam',10),('momentum',10),('stpex2',14)]:
        top=rank(rows,g)[:2];configs=[]
        # Coordinate neighbors around two leaders; shared pair average determines leaders.
        for center in top:
            for dim,grid in GRIDS[g].items():
                value=center[dim];i=grid.index(value)
                for j in (i-1,i+1):
                    if 0<=j<len(grid):configs.append(dict(center,**{dim:math.sqrt(value*grid[j])}))
        unique={digest(c):c for c in configs};result[g]=list(unique.values())[:budget]
    return result

def boundary(rows):
    result={}
    for g in GROUPS:
        center=rank(rows,g)[0];configs=[]
        for dim,grid in GRIDS[g].items():
            v=center[dim]
            if v==min(grid):configs.append(dict(center,**{dim:v/2}))
            elif v==max(grid):configs.append(dict(center,**{dim:v*2}))
        result[g]=configs[:2]
    return result
