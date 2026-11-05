import math
import subprocess

from exp18c.continuation import BASE, ROOT
from exp18c.continuation.common import PYTHON, code_seal, now, read, save
from exp18c.continuation.setup import prepare
from exp18c.continuation.search import accept_round
from exp18c.continuation.scheduler import execute


def main():
    prepare()
    with (BASE / "unit_tests.log").open("w") as handle:
        subprocess.run([PYTHON,"-B","-m","unittest","discover","-s","exp18c/continuation/tests","-v"],
                       cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,check=True)
    rows = accept_round(1, execute)
    assert len(rows) == 6
    policy = read(BASE / "scheduling.json")
    measured = read(BASE / "smoke_status.json")["peak_worker_nvml_mib"]
    policy["worker_budget_mib"] = max(policy["worker_budget_mib"],math.ceil(measured*1.1/64)*64)
    policy["smoke_peak_worker_nvml_mib"] = measured
    save(BASE / "scheduling.json",policy)
    save(BASE / "acceptance.json",dict(passed=True,time=now(),tests="unit_tests.log",
        smoke_trials=6,smoke_steps=162,physical_gpus=policy["physical_gpus"],
        exact_initializations_and_matrices=True,paired=True,code=code_seal()))
    (BASE / "final_report.md").write_text("# Exp18c 继续搜索\n\n首轮验收通过。后台逐轮扩展至九条验证曲线均出现高学习率回落；完成后自动更新本报告。\n")
    print("Acceptance passed; round-number extension protocol and source hashes frozen",flush=True)


if __name__ == "__main__":
    main()
