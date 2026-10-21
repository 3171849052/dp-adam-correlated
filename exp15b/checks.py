"""Run numerical tests and retain the exact completion/log provenance."""
import json
import re
import subprocess
import sys
from exp15b import ROOT, RESULTS
from exp15b.config import save_json, file_hash
from exp15b.support import code_hashes


def main():
    argv=[sys.executable,'-B','-m','pytest','exp15b/tests','-q','-p','no:cacheprovider',
          '--basetemp=exp15b/runtime/tests']
    with (RESULTS/'tests.log').open('w') as log:
        process=subprocess.run(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    log=(RESULTS/'tests.log').read_text()
    match=re.search(r'(\d+) passed',log)
    row=dict(status='passed' if process.returncode==0 and match else 'failed',
             passed=int(match[1]) if match else 0,exit_code=process.returncode,command=argv,
             log_sha256=file_hash(RESULTS/'tests.log'),code_sha256=code_hashes())
    save_json(RESULTS/'test_results.json',row)
    print(json.dumps(row),flush=True)
    if row['status']!='passed':raise RuntimeError('Numerical tests failed; see tests.log')


if __name__=='__main__':main()
