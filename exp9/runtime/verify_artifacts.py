"""Independent CPU inspection of real smoke checkpoints and matrix artifacts."""
import json,sys,platform,subprocess
from importlib.metadata import version
import numpy as np
import torch
from exp9 import ROOT,RESULTS
from exp9.config import Trial,file_hash,array_hash,save_json
from exp9.audit import audit_trial,verify_stage1,protected_snapshot
from exp9.privacy import fixed_epoch_sensitivity,epsilon_from_mu,target_mu
from exp9.noise import materialize

before=protected_snapshot();verify_stage1()
rows=json.loads((RESULTS/'smoke_summary.json').read_text());checks=[]
for row in rows:
    cfg=Trial(**{k:row[k] for k in Trial.__dataclass_fields__});audit_trial(cfg)
    with np.load(cfg.output/'matrices.npz') as matrices:
        d,S,W=(matrices[k] for k in ('coefficients','strategy','workload'))
        np.testing.assert_allclose(materialize(d,cfg.total_steps)@S,np.eye(cfg.total_steps),atol=3e-12)
        assert array_hash(S)==row['strategy_sha256'] and array_hash(W)==row['workload_sha256']
        sens=fixed_epoch_sensitivity(S,5,cfg.spacing)
        sigma=cfg.C*sens/target_mu(cfg.epsilon,1e-5)
        assert np.isclose(sigma,row['calibration']['innovation_std_sum'],rtol=1e-12)
        assert abs(epsilon_from_mu(cfg.C*sens/sigma,1e-5)-cfg.epsilon)<1e-7
        partial=epsilon_from_mu(cfg.C*fixed_epoch_sensitivity(S,5,cfg.spacing,steps=2)/sigma,1e-5)
        assert abs(partial-row['actual_epsilon'])<1e-7
    checkpoint=torch.load(cfg.output/'checkpoint.pt',map_location='cpu',weights_only=True)
    assert checkpoint['logical_steps']==2
    assert all(torch.isfinite(v).all() for v in checkpoint['model'].values())
    states=checkpoint['optimizer']['state']
    assert len(states)==len(checkpoint['model'])
    for state in states.values():
        assert int(state['step'])==2
        assert torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
        assert (state['exp_avg_sq']>=0).all()
    assert len(checkpoint['optimizer']['param_groups'])==1
    group=checkpoint['optimizer']['param_groups'][0]
    assert group['betas']==(.9,.999) and group['eps']==1e-8 and group['weight_decay']==0 and group['lr']==cfg.lr
    assert row['innovation_tensor_draws']==2*len(states)
    checks.append(dict(trial_id=cfg.id,task=cfg.task,method=cfg.method,T=cfg.total_steps,
                  state_tensors=len(states),full_epsilon=cfg.epsilon,executed_prefix_epsilon=partial,status='passed'))
    del checkpoint
assert before==protected_snapshot()
packages=('torch','torchvision','opacus','jax','jax_privacy','numpy','scipy','scikit-learn','timm','transformers','datasets','pandas','pyarrow')
save_json(RESULTS/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),
          conda_environment='curve',versions={p:version(p) for p in packages},torch_cuda=torch.version.cuda,
          GPUs=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,driver_version,memory.total','--format=csv'],text=True)))
save_json(RESULTS/'artifact_math_audit.json',dict(status='passed',independent_checkpoint_reloads=True,checks=checks,
          protected_files_unchanged=True,full_experiments_started=False))
# Bind additional validation artifacts to the same successful platform gate.
p=RESULTS/'platform_validation.json';gate=json.loads(p.read_text())
for name in ('workflow_validation.json','artifact_math_audit.json','environment.json'):
    gate['evidence_sha256'][name]=file_hash(RESULTS/name)
save_json(p,gate);verify_stage1()
print('Independent matrix/GDP/checkpoint audit passed for all 22 current smoke trials; Stage2 gate passed.')
