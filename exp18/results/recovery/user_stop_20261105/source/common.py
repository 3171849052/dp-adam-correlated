import hashlib,json,os
from exp18 import BASE,ROOT,RESULTS
PYTHON='/home/longt29/miniconda3/envs/curve/bin/python'
GROUPS={'adam':['adam-iid','adam-b2lt-sgd'],'momentum':['momentum-iid','momentum-b2lt-momentum'],'stpex2':['stpex2-iid','stpex2-b2lt-momentum-bias']}
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def codefiles():return sorted(BASE.glob('*.py'))+sorted((BASE/'tests').glob('*.py'))+[BASE/'protocol.json',ROOT/'stp-ex2.pdf',RESULTS/'assets.json']+sorted((RESULTS/'matrices').glob('*.json'))+sorted((RESULTS/'matrices').glob('*.npz'))
def seal():return {str(p.relative_to(ROOT)):sha(p) for p in codefiles()}
def verify_seal():
    expected=read(RESULTS/'acceptance.json')['code'];assert seal()==expected,'Frozen code changed'
def env(gpu):
    e=os.environ.copy();e.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',XDG_CACHE_HOME=str(BASE/'runtime/cache'),TORCH_HOME=str(BASE/'runtime/torch'));return e
