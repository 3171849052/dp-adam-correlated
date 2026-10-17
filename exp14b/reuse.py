"""Read historical source without importing its write-producing package initializer."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def load(name, namespace, replacements=()):
    path = ROOT / 'exp14a' / (name + '.py')
    source = path.read_text().replace('exp14a', 'exp14b')
    for old, new in replacements:
        assert old in source, old
        source = source.replace(old, new)
    exec(compile(source, str(path), 'exec'), namespace)
