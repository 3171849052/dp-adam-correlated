import torch,numpy as np
from torchvision.datasets import MNIST
from sklearn.model_selection import train_test_split
from exp18 import DATA,ROOT,RESULTS
from exp18.common import save,sha

def prepare():
    paths=[]
    for train,name in [(True,'train'),(False,'test')]:
        ds=MNIST(str(DATA),train=train,download=True)
        p=DATA/f'{name}.pt'
        torch.save({'images':ds.data.unsqueeze(1).float().div(255).sub(.1307).div(.3081),'labels':ds.targets},p);paths.append(p)
        assert len(ds)==(60000 if train else 10000)
        if train:
            a,b=train_test_split(np.arange(60000),train_size=54000,stratify=ds.targets.numpy(),random_state=20261101)
            p=DATA/'split.npz';np.savez(p,train=np.sort(a),valid=np.sort(b));paths.append(p)
    save(RESULTS/'assets.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
if __name__=='__main__':prepare()
