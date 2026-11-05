"""Explicit trial-level recovery; never changes seed, batch, precision or completed data."""
import argparse
import fcntl
import shutil
from pathlib import Path

from exp18b import BASE
from exp18b.common import completed, now, read, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-incomplete", action="store_true", required=True)
    parser.parse_args()
    with (BASE / "pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in Path("/proc").glob("[0-9]*/cmdline"):
            if not path.exists():
                continue
            # Processes may exit while /proc is enumerated; disappearing paths are normal.
            try:
                arguments = path.read_bytes().split(b"\0")
            except FileNotFoundError:
                continue
            assert not any(arg in (b"exp18b.train", b"exp18b.evaluate", b"exp18b.pipeline")
                           for arg in arguments), f"Experiment worker is alive: {path.parent.name}"
        incomplete = []
        for directory in sorted((BASE / "trials").glob("*")):
            if completed(read(directory / "config.json")) is None:
                incomplete.append(directory)
        # An official test may already have consumed records before a crash; don't silently repeat it.
        unfinished_tests = [str(p) for p in (BASE / "official_test").glob("*") if not (p / "complete.json").exists()]
        assert not unfinished_tests, f"Official test attempts need manual inspection: {unfinished_tests}"
        destination = BASE / "recovery" / now().replace(":", "-")
        archived = []
        for directory in incomplete:
            target = destination / directory.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(directory, target)
            archived.append(dict(original=str(directory), archived=str(target)))
        save(BASE / "recovery.json", dict(time=now(), archived=archived, completed_trials_preserved=True))
        print(f"Archived {len(archived)} incomplete trials; resume with python -B -m exp18b.start_background")


if __name__ == "__main__":
    main()
