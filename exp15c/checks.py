"""Run focused tests with all writable test state inside Exp15c."""
import subprocess
import sys
from exp15c import ROOT,RESULTS,BASE
from exp15c.config import save_json,file_hash
from exp15c.support import code_hashes,protected_snapshot


def main():
    before=RESULTS/'protected_before.json'
    if not before.exists(): save_json(before,protected_snapshot())
    with (RESULTS/'tests.log').open('w') as log:
        result=subprocess.run([sys.executable,'-B','-m','pytest','-q','exp15c/tests','-p','no:cacheprovider',
            '--basetemp',str(BASE/'runtime/pytest')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    save_json(RESULTS/'test_results.json',dict(status='passed' if result.returncode==0 else 'failed',exit_code=result.returncode,
        log_sha256=file_hash(RESULTS/'tests.log'),code_sha256=code_hashes()))
    print((RESULTS/'tests.log').read_text(),flush=True)
    if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
