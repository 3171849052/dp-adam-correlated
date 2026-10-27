"""Queue exp16b behind the entire exp17 pipeline, then use physical GPUs 2, 3."""
import argparse
import ast
from collections import Counter
import fcntl
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from exp16b import RESULTS
from exp16b.config import file_hash, save_json
from exp16b.search import coarse_jobs, freeze
from exp16b.stage1 import verify_acceptance
from exp16b.support import read
import exp16b.scheduler as scheduler

GPUS = (2, 3)
WORKER = ROOT / 'exp16b/runtime/train_gpus_2_3.py'
SELF = Path(__file__).resolve()
DEPENDENCY_STATUS = ROOT / 'exp17/results/pipeline_status.json'
DEPENDENCY_LOCK = ROOT / 'exp17/runtime/scheduler.lock'


def dependency_state():
    """A between-trial GPU gap is insufficient: require terminal pipeline success."""
    try:
        state = read(DEPENDENCY_STATUS)
    except (FileNotFoundError, ValueError) as error:
        return False, dict(reason=str(error))
    if state.get('phase') != 'completed':
        return False, state
    try:
        with DEPENDENCY_LOCK.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False, dict(state, reason='exp17 scheduler still holds its lock')
    # Also handle the small startup interval before a new exp17 scheduler locks.
    background = ROOT / 'exp17/results/background.json'
    if background.exists():
        pid = read(background).get('pid')
        if pid:
            try:
                proc = Path('/proc') / str(pid)
                stat = (proc / 'stat').read_text().split(') ', 1)[1].split()
                command = (proc / 'cmdline').read_text()
                if stat[0] != 'Z' and 'exp17' in command:
                    return False, dict(state, reason='exp17 scheduler is still alive')
            except FileNotFoundError:
                pass
    return True, state


def gated_gpu_processes():
    occupied = scheduler.gpu_processes()
    if not dependency_state()[0]:
        for gpu in GPUS:
            occupied[gpu].add(-1)
    return occupied


def load_batch():
    """Keep result verification, scheduling capacity and failure handling intact."""
    tree = ast.parse((ROOT / 'exp16b/scheduler.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_batch')
    loops = [n for n in ast.walk(function) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == 'gpu'
             and ast.dump(n.iter) == ast.dump(ast.parse('[0]', mode='eval').body)]
    launches = [n for n in ast.walk(function) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'argv' for t in n.targets)]
    expected_argv = ast.parse(
        "[sys.executable,'-B','-m','exp16b.train','--trial-json',json.dumps(c.values(),sort_keys=True)]",
        mode='eval',
    ).body
    if len(loops) != 1 or len(launches) != 1 or ast.dump(launches[0].value) != ast.dump(expected_argv):
        raise RuntimeError('Accepted scheduler changed; inspect GPU adapter')
    loops[0].iter = ast.parse('(2, 3)', mode='eval').body
    launches[0].value = ast.parse(
        f"[sys.executable,'-B',{str(WORKER)!r},'--trial-json',json.dumps(c.values(),sort_keys=True)]",
        mode='eval',
    ).body
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(vars(scheduler), gpu_processes=gated_gpu_processes)
    exec(compile(module, str(ROOT / 'exp16b/scheduler.py'), 'exec'), namespace)
    return namespace['run_batch'], ast.unparse(module)


def preflight():
    verify_acceptance()
    freeze()
    batch, source = load_batch()
    from exp16b.runtime.train_gpus_2_3 import load_run
    load_run()
    counts = Counter()
    for cfg in coarse_jobs():
        if (cfg.output / 'summary.json').exists():
            counts[scheduler.verify(cfg)['status']] += 1
        elif cfg.output.exists():
            raise RuntimeError(f'Partial trial requires inspection: {cfg.output}')
        else:
            counts['pending'] += 1
    print(f'Preflight passed: {dict(counts)}; assigned physical GPUs {GPUS}', flush=True)
    return counts, batch, source


def queue_status(counts, dependency, status='waiting_for_exp17', **extra):
    value = dict(status=status, stage='coarse', pid=os.getpid(), gpus=list(GPUS),
                 completed=counts['completed'], pending=counts['pending'],
                 dependency='exp17 entire pipeline, including ablation',
                 dependency_state=dependency, updated=time.time(),
                 log=str(RESULTS / 'pipeline.log'), **extra)
    save_json(RESULTS / 'pipeline_status.json', value)
    save_json(RESULTS / 'queue_gpus_2_3.json', value)
    background_path = RESULTS / 'background.json'
    if background_path.exists():
        background = read(background_path)
        if background.get('pid') == os.getpid():
            background.update(status=status, updated=value['updated'])
            save_json(background_path, background)


def run(lock_fd):
    os.fstat(lock_fd)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    counts, batch, source = preflight()
    import hashlib
    transition = dict(gpus=list(GPUS), counts=dict(counts), queued=time.time(),
                      scheduler_adapter_sha256=file_hash(SELF), worker_adapter_sha256=file_hash(WORKER),
                      executed_scheduler_source=source,
                      executed_scheduler_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      dependency_status=str(DEPENDENCY_STATUS),
                      interrupted_trial='359aaae1240908c02cb6', interrupted_trial_restart_step=0)
    save_json(RESULTS / 'gpu_transition.json', transition)
    try:
        status = {}
        for cfg in coarse_jobs():
            if (cfg.output / 'summary.json').exists():
                status[cfg.id] = dict(**cfg.values(), status=read(cfg.output / 'summary.json')['status'], reused=True)
            else:
                status[cfg.id] = dict(**cfg.values(), status='pending')
        save_json(RESULTS / 'coarse_status.json', status)
        scheduler.update_report(coarse_jobs(), status)
        scheduler.event(batch='coarse', event='queued_behind_exp17', pid=os.getpid(),
                        gpus=list(GPUS), counts=dict(counts))
        previous = None
        while True:
            ready, dependency = dependency_state()
            if ready:
                break
            queue_status(counts, dependency)
            signature = (dependency.get('phase'), dependency.get('completed'), dependency.get('pending'), dependency.get('reason'))
            if signature != previous:
                print(f'Waiting for exp17 to finish: {dependency}', flush=True)
                previous = signature
            time.sleep(15)
        queue_status(counts, dependency, status='running', started=time.time())
        print('exp17 completed; starting exp16b on physical GPUs 2 and 3', flush=True)
        rows = batch('coarse', coarse_jobs())
        save_json(RESULTS / 'coarse_results.json', rows)
        counts = Counter(row['status'] for row in rows)
        queue_status(counts, dependency, status='completed', ended=time.time())
    except Exception:
        queue_status(counts, {}, status='blocked', traceback=traceback.format_exc(), ended=time.time())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--start', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--lock-fd', type=int)
    args = parser.parse_args()
    if args.run:
        if args.lock_fd is None:
            parser.error('--run requires --lock-fd')
        run(args.lock_fd)
        return
    with (RESULTS / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        counts, _, _ = preflight()
        ready, dependency = dependency_state()
        print(f'exp17 dependency ready: {ready}; phase: {dependency.get("phase")}', flush=True)
        if args.check:
            return
        # Snapshot operational state only; completed trials and the prior interruption archive remain intact.
        archive = ROOT / 'exp16b/runtime' / time.strftime('before_queue_2_3_%Y%m%d_%H%M%S', time.localtime())
        archive.mkdir()
        import shutil
        for name in ('background.json', 'pipeline_status.json', 'progress.json', 'coarse_status.json'):
            if (RESULTS / name).exists():
                shutil.copy2(RESULTS / name, archive / name)
        command = [sys.executable, '-B', str(SELF), '--run', '--lock-fd', str(lock.fileno())]
        with (RESULTS / 'pipeline.log').open('a') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     pass_fds=(lock.fileno(),))
        save_json(RESULTS / 'background.json', dict(
            pid=child.pid, session_id=child.pid, gpus=list(GPUS),
            status='queued_behind_exp17', command=shlex.join(command),
            launch_command=f'{sys.executable} -B {SELF} --start',
            log=str(RESULTS / 'pipeline.log'), started=time.time(), previous_state_archive=str(archive),
        ))
        print(f'Queued exp16b behind exp17; detached PID {child.pid}; GPUs {GPUS}', flush=True)


if __name__ == '__main__':
    main()
