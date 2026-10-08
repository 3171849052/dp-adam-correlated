"""Import the three fixed Exp7 settings; freeze once before any final run."""
import json
from exp7b import BASE, ROOT
from exp7b.config import METHODS, FROZEN_METHODS, SEARCH_METHODS, trial, save_json
from exp7b.audit import file_hash

def import_fixed():
    path = ROOT / 'exp7/results/selected_configs.json'
    old = json.loads(path.read_text())
    expected = ((.0005, 30., None), (.005, 30., None), (.002, 100., .1))
    rows = {}
    for method, values in zip(FROZEN_METHODS, expected):
        row = old[method]
        assert tuple(row[k] for k in ('lr', 'C', 'eps_scale')) == values
        rows[method] = dict(trial(method, *values), source='exp7_frozen',
                            num_bands=row['num_bands'], provenance=dict(
                                selected_configs=str(path.relative_to(ROOT)),
                                selected_configs_sha256=file_hash(path), original=row))
    save_json(BASE / 'results/exp7_frozen_configs.json', rows)
    return rows

def freeze(selected):
    path = BASE / 'results/frozen_configs.json'
    rows = dict(import_fixed(), **selected)
    assert set(selected) == set(SEARCH_METHODS) and set(rows) == set(METHODS)
    rows = {m: rows[m] for m in METHODS}
    if path.exists():
        assert json.loads(path.read_text()) == rows, 'Frozen configurations are immutable'
    else:
        save_json(path, rows)
        save_json(BASE / 'results/frozen_manifest.json', dict(sha256=file_hash(path)))
    return load_frozen()

def load_frozen():
    path = BASE / 'results/frozen_configs.json'
    assert file_hash(path) == json.loads((BASE / 'results/frozen_manifest.json').read_text())['sha256']
    rows = json.loads(path.read_text())
    assert set(rows) == set(METHODS)
    for method, row in rows.items():
        trial(method, row['lr'], row['C'], row['eps_scale'])
        assert row['source'] == ('exp7_frozen' if method in FROZEN_METHODS else 'search_selected')
    return rows

if __name__ == '__main__':
    import_fixed()
