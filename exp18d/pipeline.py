import fcntl
import os
from exp18d.common import BASE, now, save, verify_seal
from exp18d.scheduler import run
from exp18d.report import report


def main():
    with (BASE / 'runtime/pipeline.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify_seal()
        save(BASE / 'pipeline_process.json',dict(pid=os.getpid(),started=now(),status='running'))
        try:
            run('formal')
            assert report(), 'Formal results incomplete'
        except BaseException:
            report()
            save(BASE / 'pipeline_process.json',dict(pid=os.getpid(),ended=now(),status='failed'))
            raise
        save(BASE / 'pipeline_process.json',dict(pid=os.getpid(),ended=now(),status='complete'))

if __name__ == '__main__': main()
