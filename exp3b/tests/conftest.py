from pathlib import Path
from exp3b import BASE


def pytest_sessionstart(session):
    path = Path(session.config.option.basetemp).resolve()
    assert path.is_relative_to(BASE)
    path.parent.mkdir(parents=True, exist_ok=True)
