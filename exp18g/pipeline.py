"""Detached five-run formal controller and automatic seven-algorithm report."""
import os
from exp18g.common import *
from exp18g.scheduler import run

def pipeline():
    verify_seal();assert read(BASE/'acceptance/preflight.json')['passed']
    save(BASE/'pipeline_status.json',dict(status='running',stage='formal',pid=os.getpid(),time=now()))
    run('formal')
    from exp18g.report import report
    report()
    save(BASE/'pipeline_status.json',dict(status='complete',stage='finished',pid=os.getpid(),completed_formal=5,time=now()))
if __name__=='__main__':pipeline()
