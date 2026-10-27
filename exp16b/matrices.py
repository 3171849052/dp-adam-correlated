"""Read-only reuse of precisely four accepted exp16a designs; no solver."""
import json
import numpy as np
from exp16b import ROOT, RESULTS
from exp16b.config import file_hash, save_json
from exp16b.privacy import fixed_epoch_sensitivity, target_mu, epsilon_from_mu

def load(work,mechanism,n):
    assert work=='momentum-bias' and mechanism in ('bandinvmf','b2lt') and n in (225,310)
    source=ROOT/'exp16a/results'
    p=source/'matrices'/f'{work}_{mechanism}_T{n}.npz'
    acceptance=json.loads((source/'phase1_acceptance.json').read_text())
    assert acceptance['status']=='passed'
    for f in (p,p.with_suffix('.json')):
        assert file_hash(f)==acceptance['matrix_sha256'][str(f.relative_to(ROOT))]
    meta=json.loads(p.with_suffix('.json').read_text())
    assert file_hash(p)==meta['sha256']
    with np.load(p) as f: arrays={k:f[k].copy() for k in f.files}
    return dict(arrays,path=p,metadata=meta)

def audit(a):
    A,C,D=a['A'],a['C'],a['D'];n=len(A)
    expected=np.zeros_like(A)
    for t in range(1,n+1):
        for j in range(1,t+1):expected[t-1,j-1]=sum(.9**(s-j)/(1-.9**s) for s in range(j,t+1))
    np.testing.assert_allclose(A,expected,rtol=1e-12)
    np.testing.assert_allclose(C@D,np.eye(n),atol=1e-8)
    np.testing.assert_allclose((A@D)@C,A,atol=1e-8)
    sens=fixed_epoch_sensitivity(C,5,n//5)
    objective=sens**2*np.sum((A@D)**2)/n
    assert np.isclose(objective,a['metadata']['audit']['objective'],rtol=1e-10)
    sigma=sens/target_mu(8,1e-5)
    np.testing.assert_allclose(C@a['covariance_clip1']@C.T,sigma**2*np.eye(n),atol=1e-6)
    return dict(source=str(a['path'].relative_to(ROOT)),sha256=file_hash(a['path']),sensitivity=sens,objective=objective,epsilon=epsilon_from_mu(sens/sigma,1e-5))

def prepare():
    rows=[audit(load('momentum-bias',m,n)) for n in (225,310) for m in ('bandinvmf','b2lt')]
    save_json(RESULTS/'matrix_audit.json',rows)
    return rows
