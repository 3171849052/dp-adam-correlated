"""Small experiment-local artifact helpers."""
import hashlib
import json
from pathlib import Path
from exp13 import BASE


def local_path(path):
    path = Path(path).resolve()
    assert path.is_relative_to(BASE), path
    return path


def save_json(path, value):
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
