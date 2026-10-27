"""Run the accepted worker with only its physical GPU admission guard changed.

The original files and their hashes remain intact. Each new trial records the
executed derivative and this adapter alongside the accepted source provenance.
"""
import ast
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def load_run():
    import exp16b.train as train

    tree = ast.parse(Path(train.__file__).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    expected = ast.parse(
        "os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or torch.cuda.device_count()!=1",
        mode='eval',
    ).body
    replacement = ast.parse(
        "os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2', '3') or torch.cuda.device_count()!=1",
        mode='eval',
    ).body
    guards = [n for n in function.body if isinstance(n, ast.If)
              and ast.dump(n.test) == ast.dump(expected)]
    if len(guards) != 1:
        raise RuntimeError('Accepted worker GPU guard changed; inspect before running')
    guards[0].test = replacement
    guards[0].body = ast.parse(
        "raise RuntimeError('Isolate one physical GPU with CUDA_VISIBLE_DEVICES=2 or 3')"
    ).body
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(vars(train))
    exec(compile(module, train.__file__, 'exec'), namespace)
    return namespace['run'], ast.unparse(module)


def main():
    import argparse
    import hashlib
    import json
    from exp16b.config import Trial, file_hash, save_json
    from exp16b.stage1 import verify_acceptance

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial-json', required=True)
    cfg = Trial(**json.loads(parser.parse_args().trial_json))
    verify_acceptance()
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2', '3'):
        raise RuntimeError('This adapter is restricted to physical GPU 2 or 3')
    if (cfg.output / 'config.json').exists():
        raise FileExistsError(f'Run already exists or incomplete: {cfg.output}')
    run, executed_source = load_run()
    save_json(cfg.output / 'gpu_migration.json', dict(
        physical_gpu=int(os.environ['CUDA_VISIBLE_DEVICES']),
        change='Only GPU admission guard: allow one isolated physical GPU 2 or 3',
        accepted_source='exp16b/train.py',
        accepted_source_sha256=file_hash(ROOT / 'exp16b/train.py'),
        adapter_sha256=file_hash(Path(__file__)),
        executed_run_source=executed_source,
        executed_run_sha256=hashlib.sha256(executed_source.encode()).hexdigest(),
    ))
    run(cfg)


if __name__ == '__main__':
    main()
