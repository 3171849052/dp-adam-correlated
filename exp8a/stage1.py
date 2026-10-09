"""Prepare inputs, run every unit test and two REAL 1000-example GPU updates."""
import argparse
import json
import subprocess
import sys
from importlib.metadata import version
from exp8a import ROOT, RESULTS
from exp8a.config import Config, save_json, file_hash, MODEL_PATH
from exp8a.data import download, prepare
from exp8a.train import mechanism
from exp8a.benchmark_batch import isolated, write_csv
from exp8a.benchmark_length import ACCURACY_FIELDS
from exp8a.audit import source_hashes, report


def execute(gpu=0):
    download()
    stats=prepare()
    command=[sys.executable,'-B','-m','pytest','exp8a/tests','-q','-o','cache_dir=exp8a/runtime/pytest_cache',
             '--basetemp=exp8a/runtime/tests','--junitxml=exp8a/results/tests.xml']
    with open(RESULTS/'tests.log','w') as log:
        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    correctness=[json.loads((RESULTS/f'correctness_{a}_{g}_{d}.json').read_text())
                 for a in ('small','pretrained') for g in ('standard','scale') for d in ('cpu','cuda_0')]
    assert len(correctness)==8 and all(r['status']=='passed' for r in correctness)
    save_json(RESULTS/'clipping_correctness.json',dict(status='passed',tests=correctness,
              tolerances=dict(norm_rtol=3e-5,norm_atol=3e-6,gradient_rtol=5e-5,gradient_atol=3e-6),
              embedding_identity='g_i[v]=sum_{j:x_ij=v} backprop_ij; scale after duplicate aggregation; padding zero'))
    cfg=Config()
    rows=[isolated(cfg,g,gpu,warmup=0,measured=1,category='smoke') for g in ('standard','scale')]
    assert all(r['status']=='completed' and r['noise_draws']==r['optimizer_steps']==1 for r in rows)
    for key in ('initialization_sha256','classifier_initialization_sha256','checkpoint_sha256','split_sha256'):
        assert rows[0][key]==rows[1][key]
    write_csv(RESULTS/'stage1_gpu_smoke.csv',rows)
    for name in ('physical_batch_benchmark.csv','max_length_benchmark.csv'):
        if not (RESULTS/name).exists(): write_csv(RESULTS/name,[])
    if not (RESULTS/'max_length_accuracy.csv').exists(): write_csv(RESULTS/'max_length_accuracy.csv',[],ACCURACY_FIELDS)
    d,S,W,meta,privacy=mechanism(cfg)
    save_json(RESULTS/'mechanism.json',dict(coefficients=d.tolist(),optimization=meta,privacy=privacy))
    selected=dict(status='stage1_only',physical_batch_size=None,max_length=None,logical_batch_size=1000,
                  training_protocol=cfg.protocol(),split_sha256=stats['split_sha256'],
                  checkpoint_sha256=file_hash(MODEL_PATH/'pytorch_model.bin'),
                  bandinvmf=dict(coefficients=d.tolist(),optimization=meta),privacy_calibration=privacy,
                  selection_basis='Stage 2 not run. Null parameters must not be consumed by Exp8b.')
    # Preserve a completed Stage 2 selection if Stage 1 is rerun later.
    if not (RESULTS/'selected_config.json').exists() or json.loads((RESULTS/'selected_config.json').read_text())['status']=='stage1_only':
        save_json(RESULTS/'selected_config.json',selected)
        report(selected,rows)
    stage1=dict(status='passed',source_sha256=source_hashes(),split_sha256=stats['split_sha256'],
                checkpoint_sha256=selected['checkpoint_sha256'],gpu=gpu,gpu_smoke=rows,
                versions={p:version(p) for p in ('torch','transformers','opacus','datasets','numpy','scipy','pytest','jax_privacy')},
                tests_xml='exp8a/results/tests.xml',stage2_started=False)
    save_json(RESULTS/'stage1.json',stage1)
    print(json.dumps(stage1,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',type=int,default=0)
    a=p.parse_args(); execute(a.gpu)

if __name__=='__main__': main()
