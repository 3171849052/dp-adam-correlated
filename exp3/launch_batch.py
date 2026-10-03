"""A supplied batch of 1–3 trials, on GPUs 1,2,3, without a sweep."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from exp3.spec import ROOT, EXP3, TrialSpec, read_completed

GPUS = (1, 2, 3)


def write_summary(output, results):
    summary = dict(status='failed' if any(r['returncode'] != 0 for r in results) else 'completed',
                   gpus=list(GPUS), trials=results)
    (output / 'batch_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def run_queue(specs, output):
    output = Path(output).resolve()
    assert output.is_relative_to(EXP3) and output != EXP3
    assert len({spec.directory for spec in specs}) == len(specs), 'Duplicate result directories'
    output.mkdir(parents=True, exist_ok=True)
    pending, active, results = [], {}, []
    # Validate all pre-existing directories BEFORE starting any new work.
    for spec in specs:
        if spec.directory.exists():
            try:
                completed = read_completed(spec)
            except RuntimeError as error:
                results.append(dict(spec=spec.mapping(), reused=False, returncode=1, error=str(error)))
            else:
                results.append(dict(spec=spec.mapping(), reused=True, returncode=0, summary=completed))
        else:
            pending.append(spec)
    if any(r['returncode'] != 0 for r in results):
        results.extend(dict(spec=s.mapping(), status='not_started', reused=False, returncode=None) for s in pending)
        return write_summary(output, results)
    while pending or active:
        for gpu in GPUS:
            if pending and gpu not in active:
                spec = pending.pop(0)
                spec.directory.mkdir(parents=True)
                # Keep the reservation directory at train.log only. Input files go to batch output.
                spec_path = output / (spec.fingerprint() + '.json')
                spec_path.write_text(json.dumps(spec.mapping(), indent=2))
                log = (spec.directory / 'train.log').open('x', buffering=1)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           EXP3_LAUNCHER_RESERVED=str(spec.directory), PYTHONDONTWRITEBYTECODE='1',
                           TMPDIR=str(EXP3 / 'tmp'))
                (EXP3 / 'tmp').mkdir(exist_ok=True)
                process = subprocess.Popen([sys.executable, '-m', 'exp3.run_trial', '--spec', str(spec_path)],
                                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (process, spec, log)
        finished = []
        for gpu, (process, spec, log) in active.items():
            code = process.poll()
            if code is not None:
                log.close()
                row = dict(spec=spec.mapping(), gpu=gpu, reused=False, returncode=code)
                if code == 0:
                    try:
                        row['summary'] = read_completed(spec)
                    except RuntimeError as error:
                        row.update(returncode=1, error=str(error))
                results.append(row)
                finished.append(gpu)
        for gpu in finished:
            del active[gpu]
        if active and not finished:
            time.sleep(.25)
    return write_summary(output, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--specs', required=True, type=Path)
    parser.add_argument('--result-dir', type=Path)
    args = parser.parse_args()
    supplied = json.loads(args.specs.read_text())
    assert isinstance(supplied, list) and 1 <= len(supplied) <= 3
    specs = [TrialSpec(**values) for values in supplied]
    summary = run_queue(specs, args.result_dir or args.specs.parent)
    print(json.dumps(dict(status=summary['status'], trials=len(summary['trials']))))
    if summary['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
