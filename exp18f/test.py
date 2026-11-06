"""Execute numerical and orchestration tests in curve and persist the result."""
import io
import sys
import unittest
from exp18f.common import BASE,save,now

def main():
    assert '/envs/curve/' in sys.executable
    suite=unittest.defaultTestLoader.discover(str(BASE/'tests'))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    log=stream.getvalue();(BASE/'logs/unit_tests.log').write_text(log);print(log,flush=True)
    save(BASE/'unit_tests.json',dict(passed=result.wasSuccessful(),tests=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),python=sys.executable,time=now()))
    if not result.wasSuccessful():raise SystemExit(1)

if __name__=='__main__':main()
