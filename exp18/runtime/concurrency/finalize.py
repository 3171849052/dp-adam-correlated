import shutil,subprocess,json,time
from exp18 import BASE,RESULTS,ROOT
from exp18.common import read,save,sha,PYTHON
bench=RESULTS/'concurrency';rows=[json.loads(s) for s in (bench/'results.jsonl').read_text().splitlines()]
assert read(bench/'capacity.json')['max_success']==16
assert all(any(r['group']==g and r['n']==16 and r['success'] for r in rows) for g in ('adam','momentum','stpex2'))
assert any(r['group']=='stpex2' and r['n']==12 and r['repeat']==1 and r['success'] for r in rows)
old=read(RESULTS/'recovery/user_stop_20261105/acceptance.json')
for name in ('train.py','optimizer.py','model.py','noise.py','privacy.py','matrices.py','search.py','assets.py','pipeline.py','report.py'):
 assert sha(BASE/name)==old['code']['exp18/'+name],name
shutil.copy2(BASE/'runtime/concurrency/scheduler_candidate.py',BASE/'scheduler.py')
test=(BASE/'runtime/concurrency/test_scheduler.py').read_text();start=test.index('spec=');end=test.index('class Admission')
test=test[:start]+'from exp18 import scheduler as module\n'+test[end:];(BASE/'tests/test_scheduler.py').write_text(test)
policy=dict(target_per_gpu=12,worker_budget_mib=768,reserve_mib=1536,measured_max_success=16,first_failure=17,physical_gpus=[1,2,3],evidence='results/concurrency/results.jsonl',recovery='memory limit decreases immediately; restore one slot when GPU utilization <90%; never exceed 12',selection='12 workers leaves memory margin; STP throughput varies 14.35-18.72 trials/min versus 16.62 at 16; no statistically established throughput optimum')
save(BASE/'scheduling.json',policy)
p=BASE/'common.py';s=p.read_text().replace("+[BASE/'protocol.json',ROOT/'stp-ex2.pdf'", "+[BASE/'protocol.json',BASE/'scheduling.json',ROOT/'stp-ex2.pdf'");p.write_text(s)
p=BASE/'start_background.py';s=p.read_text().replace('RESULTS,ROOT','RESULTS,ROOT,BASE').replace('verify_seal,env','verify_seal,env,read').replace('initial_processes_per_gpu=2',"initial_processes_per_gpu=read(BASE/'scheduling.json')['target_per_gpu']");p.write_text(s)
p=BASE/'protocol.json';proto=read(p);proto['initial_gpu_capacity']={str(g):12 for g in (1,2,3)};proto['dynamic_capacity']='Measured target 12 per GPU; 768 MiB reservation per worker plus 1536 MiB GPU reserve; external process memory accounted; reduce under pressure, restore slots only below 90% utilization; max observed 16, 17 OOM';save(p,proto)
with (bench/'revision_tests.log').open('w') as f:subprocess.run([PYTHON,'-B','-m','unittest','discover','-s','exp18/tests','-v'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
# Reload the changed manifest definition, without importing historical packages.
import exp18.common,importlib
importlib.reload(exp18.common)
new=dict(old);new['code']=exp18.common.seal();new['scheduling_revision']=dict(time=time.time(),prior_acceptance='results/recovery/user_stop_20261105/acceptance.json',training_core_unchanged=True,policy=policy,tests='results/concurrency/revision_tests.log',preserved_full_trials=151)
save(RESULTS/'acceptance.json',new)
exp18.common.verify_seal()
background=read(RESULTS/'background.json');background['status']='stopped_by_user';background['pid_active']=False;save(RESULTS/'background.json',background)
save(RESULTS/'status.json',dict(status='stopped_after_concurrency_revision',active=[],completed_full_trials=151,proposed_total_concurrency=36,scheduling=policy,requires_user_resume=True))
lines=['# Exp18 单卡并发实测','', 'GPU 1: RTX 3080 Ti, 12GB。固定 CNN、物理 batch=1000、FP32、162 步。各进程在第一次参数更新前同步；仅内部验证，无官方测试集访问。', '', '| 优化器 | 并发 | 重复 | 成功 | 峰值 MiB | 整批秒数 | trials/min（含启动） |','|---|---:|---:|---|---:|---:|---:|']
for r in rows:lines.append(f"| {r['group']} | {r['n']} | {r['repeat']} | {r['success']} | {r['peak_gpu_mib']} | {r['wall_seconds']:.2f} | {r['trials_per_minute']:.2f} |")
lines+=['','结论：三种优化器均通过16并发；STP-Ex2在17并发OOM，20也OOM。当前配置的可运行上限为16，而非所有未来模型/硬件的保证。12并发两次吞吐为14.35–18.72次/分钟，16并发为16.62次/分钟；存在波动，不能断言12严格最快。为保留显存余量，方案采用每卡12（总36），每进程预留768 MiB、每卡额外预留1536 MiB。根据其他进程显存占用缩减容量；利用率低于90%时逐步恢复。实测训练峰值约714 MiB/进程。', '', 'NVML瞬时采样可能漏掉短时峰值，尤其1/2进程；推荐容量以高并发同步测试为依据。43份跨并发STP模型逐位一致。三卡同时36进程的系统级吞吐尚未测量，不把单卡吞吐简单乘三作为保证。', '', '所有压力测试与调参结果隔离；不参与候选排名或正式统计。OOM现场保留，没有修改批次/精度。151个既有训练产物已重新校验，中断目录归档在 results/recovery/user_stop_20261105/。训练核心不变，仅调度及其冻结清单修订。完整实验保持停止，尚未恢复。']
(bench/'report.md').write_text('\n'.join(lines)+'\n')
p=BASE/'README.md';s=p.read_text();s+='\n## 2026-11-05 并发修订\n\n用户要求暂停后完成单卡压力测试。新方案覆盖上文的旧并发配置：物理 GPU 1/2/3 各12进程，总36；单卡测试上限16，17进程OOM。每进程预留768 MiB、每卡1536 MiB余量；按其他进程占用缩减，利用率低于90%再逐步恢复。详情见 `results/concurrency/report.md`。当前流水线保持停止。恢复沿用 `-m exp18.start_background`，151个完成任务经过校验后复用，中断现场已归档。\n';p.write_text(s)
print('Scheduling revised and sealed; experiment remains stopped.')
