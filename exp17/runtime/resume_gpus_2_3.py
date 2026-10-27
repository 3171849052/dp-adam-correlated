"""Operational GPU override; preserve the accepted source and artifact identities."""
import json
import os
from pathlib import Path
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from exp17.common import RESULTS, save
import exp17.scheduler as scheduler

original_popen = scheduler.subprocess.Popen

def compatible_worker(command, *args, **kwargs):
    if isinstance(command, list) and command[1:4] == ['-B', '-m', 'exp17.train']:
        command = [command[0], '-B', str(ROOT / 'exp17/runtime/train_resume_compat.py'), *command[4:]]
    return original_popen(command, *args, **kwargs)

scheduler.subprocess.Popen = compatible_worker
original_occupied = scheduler.occupied
scheduler.occupied = lambda: original_occupied() | {1}
transition = RESULTS / 'gpu_transition.json'
state = json.loads(transition.read_text())

def running(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False

while any(running(pid) for pid in state['draining_pids']):
    time.sleep(5)
state.update(status='running_on_gpus_2_3', resumed_unix=time.time(), pid=os.getpid())
save(transition, state)
runpy.run_module('exp17.pipeline', run_name='__main__')
