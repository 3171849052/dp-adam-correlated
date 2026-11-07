"""Start a detached process with nohup, a saved PID and complete logs."""
import os
import subprocess
from exp18f.final_official.common import BASE, ROOT, PYTHON, read, save, now, sha


def start():
    assert not (BASE/'launch.json').exists()
    assert read(BASE/'preflight.json')['passed']
    paths = list(BASE.glob('*.py')) + [BASE/n for n in (
        'protocol.json','hyperparameters.json','scheduling.json','privacy_audit.json','input_manifest.json')]
    paths += list((BASE/'plans').glob('*.json')) + list((BASE/'matrices').glob('*'))
    save(BASE/'code_manifest.json', {str(p.relative_to(BASE)): sha(p) for p in sorted(paths)})
    command = ['nohup', PYTHON, '-B', '-m', 'exp18f.final_official.pipeline']
    with (BASE/'logs/pipeline.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
    save(BASE/'launch.json', dict(pid=process.pid, command=command, log='exp18f/final_official/logs/pipeline.log', time=now()))
    (BASE/'logs/pipeline.pid').write_text(str(process.pid)+'\n')
    print(f'Started background PID {process.pid}', flush=True)


if __name__ == '__main__': start()
