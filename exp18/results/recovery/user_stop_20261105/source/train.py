import os,time,sys,traceback,json
import numpy as np,torch
from opacus import GradSampleModule
from exp18 import BASE,RESULTS
from exp18.common import read,save,sha,verify_seal
from exp18.model import CNN,tensors
from exp18.optimizer import Update
from exp18.noise import OnlineNoise
from exp18.matrices import load
from exp18.privacy import target_mu

@torch.no_grad()
def evaluate(model,x,y):
    total=correct=0;loss=0.
    for start in range(0,len(y),1000):
        xx=x[start:start+1000].cuda();yy=y[start:start+1000].cuda();logits=model(xx)
        loss+=float(torch.nn.functional.cross_entropy(logits,yy,reduction='sum'));correct+=int((logits.argmax(1)==yy).sum());total+=len(yy)
    return dict(loss=loss/total,accuracy=correct/total,n=total)

def run(config_path):
    cfg=read(config_path);out=config_path.parent
    if cfg['stage']!='smoke':verify_seal()
    assets=read(RESULTS/'assets.json')
    # Search validates only training assets, never opens official test tensors.
    for path,h in assets.items():
        if path.endswith('test.pt') and cfg['stage']!='formal':continue
        assert sha(BASE/path)==h
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=CNN().cuda();initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    group=cfg['group'];opt=Update(model,group,cfg['lr'],cfg.get('eps',1e-8));model=GradSampleModule(model,loss_reduction='sum')
    x,y=tensors();split=np.load(BASE/'data/split.npz');train=split['train'];valid=split['valid']
    order=train[np.random.default_rng(cfg['seed']).permutation(len(train))];np.save(out/'order.npy',order)
    if cfg['stage']=='formal':
        assert (RESULTS/'hyperparameters.json').exists();vx,vy=tensors(True)
    else:vx,vy=x[valid],y[valid]
    iid=cfg['method'].endswith('-iid');matrix={'metadata':{}} if iid else load({'adam':'sgd','momentum':'momentum','stpex2':'momentum-bias'}[group])
    sens=np.sqrt(3) if iid else matrix['metadata']['audit']['sensitivity'];sigma=cfg['clip']*sens/target_mu(3,1e-5)
    mech='iid' if iid or matrix['metadata'].get('selected')=='iid' else 'b2lt'
    noise=OnlineNoise(opt.ps,matrix,mech,sigma,162,cfg['seed']+1000000)
    privacy=dict(epsilon=3,delta=1e-5,calibration='GDP',sampling_amplification=False,participations=3,interval=54,sensitivity=float(sens),sigma_sum=float(sigma),scope='one full training only')
    save(out/'privacy.json',privacy);save(out/'process.json',dict(pid=os.getpid(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],batch=1000,precision='FP32',accumulation=1))
    if cfg['stage']=='smoke':
        ready=RESULTS/'smoke_ready'/str(cfg['seed'])
        save(ready/(cfg['method']+'.json'),dict(pid=os.getpid(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],cuda_memory=torch.cuda.memory_allocated(),time=time.time()))
        deadline=time.time()+120
        while len(list(ready.glob('*.json')))<6:
            if time.time()>deadline:raise RuntimeError('Six-worker CUDA readiness timeout')
            time.sleep(.2)
        time.sleep(3)
    start=time.time();curve=[];limit=2 if cfg['stage']=='smoke' else 162
    with (out/'steps.jsonl').open('w',buffering=1) as log:
        for step in range(limit):
            ids=order[(step%54)*1000:(step%54+1)*1000];xx=x[ids].cuda();yy=y[ids].cuda()
            model.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(model(xx),yy,reduction='sum');loss.backward()
            gs=torch.cat([p.grad_sample.reshape(1000,-1) for p in opt.ps],1)
            if group=='stpex2':gs.div_(opt.d)
            norms=gs.norm(dim=1);weights=(cfg['clip']/norms.clamp_min(1e-30)).clamp_max(1)
            q=(gs*weights[:,None]).sum(0);q.add_(torch.cat([z.flatten() for z in noise.next()]));q.div_(1000)
            alpha=opt.step(q)
            if not torch.isfinite(loss) or not all(torch.isfinite(p).all() for p in opt.ps):raise FloatingPointError('nonfinite training')
            log.write(json.dumps(dict(step=step+1,loss=float(loss.detach())/1000,alpha=alpha,clipped_fraction=float((weights<1).float().mean()),seconds=time.time()-start))+'\n')
            del gs,q,xx,yy
            for p in opt.ps:p.grad_sample=None
            if (step+1)%54==0 or step+1==limit:
                metrics=evaluate(model,vx[:1000] if cfg['stage']=='smoke' else vx,vy[:1000] if cfg['stage']=='smoke' else vy);curve.append(dict(step=step+1,**metrics));save(out/'curve.json',curve)
    torch.save(dict(model=model._module.state_dict(),optimizer=opt.state_dict(),initial=initial,config=cfg,privacy=privacy),out/'checkpoint.pt')
    summary=dict(status='complete',config=cfg,steps=limit,metrics=curve[-1],seconds=time.time()-start,peak_memory=torch.cuda.max_memory_allocated(),evaluation='official_test' if cfg['stage']=='formal' else 'internal_validation')
    save(out/'summary.json',summary)
    save(out/'complete.json',{p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name not in ('complete.json','worker.log')})

if __name__=='__main__':
    from pathlib import Path
    p=Path(sys.argv[1])
    try:run(p)
    except Exception as e:
        save(p.parent/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));raise
