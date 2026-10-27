"""CPU-only operational checks; never start training or acquire CUDA contexts."""
import json
import fcntl
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from exp16b.runtime import resume_gpus_2_3 as queue
from exp16b.runtime.train_gpus_2_3 import load_run


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / 'exp16b/runtime')
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)

    def test_waits_for_entire_pipeline_and_scheduler_exit(self):
        state = self.folder / 'status.json'
        lock_path = self.folder / 'scheduler.lock'
        with patch.object(queue, 'DEPENDENCY_STATUS', state), \
             patch.object(queue, 'DEPENDENCY_LOCK', lock_path), \
             patch.object(queue, 'ROOT', self.folder):
            for phase in ('coarse', 'fine', 'review', 'final', 'ablation', 'failed'):
                state.write_text(json.dumps(dict(phase=phase)))
                self.assertFalse(queue.dependency_state()[0], phase)
            state.write_text(json.dumps(dict(phase='completed')))
            with lock_path.open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertFalse(queue.dependency_state()[0])
            self.assertTrue(queue.dependency_state()[0])

    def test_idle_gpu_stays_blocked_until_dependency_completes(self):
        with patch.object(queue.scheduler, 'gpu_processes', return_value={g: set() for g in range(4)}), \
             patch.object(queue, 'dependency_state', return_value=(False, dict(phase='final'))):
            busy = queue.gated_gpu_processes()
        self.assertEqual(busy[0], set())
        self.assertEqual(busy[1], set())
        self.assertEqual(busy[2], {-1})
        self.assertEqual(busy[3], {-1})

    def test_worker_accepts_only_one_isolated_gpu_2_or_3(self):
        run, _ = load_run()
        class ReachedDeviceSelection(Exception):
            pass
        def reached_device(_):
            raise ReachedDeviceSelection()
        cuda = SimpleNamespace(device_count=lambda: 1, set_device=reached_device)
        torch = SimpleNamespace(cuda=cuda, device=lambda name: name)
        run.__globals__['torch'] = torch
        cfg = SimpleNamespace(stage='smoke', output=self.folder)
        for gpu in ('2', '3'):
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=gpu):
                with self.assertRaises(ReachedDeviceSelection):
                    run(cfg)
        for gpu in ('0', '1', '2,3', ''):
            with patch.dict(os.environ, CUDA_VISIBLE_DEVICES=gpu):
                with self.assertRaises(RuntimeError):
                    run(cfg)
        cuda.device_count = lambda: 2
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='2'):
            with self.assertRaises(RuntimeError):
                run(cfg)

    def simulate(self, tasks, occupied_gpu2=False, fail_first=False):
        batch, _ = queue.load_batch()
        launched = []
        active = {}
        peak = {2: 0, 3: 0}
        jobs = []
        for index, task in enumerate(tasks):
            trial = SimpleNamespace(id=str(index), task=task, method='example', output=self.folder / str(index))
            trial.values = lambda trial=trial: dict(trial=trial.id, task=trial.task)
            jobs.append(trial)
        class Process:
            def __init__(self, command, **kwargs):
                gpu = int(kwargs['env']['CUDA_VISIBLE_DEVICES'])
                self.gpu = gpu
                self.trial = jobs[int(json.loads(command[-1])['trial'])]
                self.pid = 10000 + len(launched)
                self.failed = fail_first and not launched
                if active.get(gpu):
                    assert self.trial.task == 'nlp'
                    assert all(p.trial.task == 'nlp' for p in active[gpu])
                active.setdefault(gpu, []).append(self)
                assert len(active[gpu]) <= (1 if self.trial.task == 'cv' else 2)
                peak[gpu] = max(peak[gpu], len(active[gpu]))
                launched.append((gpu, self.trial.id, command))
                (self.trial.output / 'summary.json').write_text(json.dumps(dict(status='failed' if self.failed else 'completed')))
            def poll(self):
                active[self.gpu].remove(self)
                return 1 if self.failed else 0
        namespace = batch.__globals__
        namespace.update(
            RESULTS=self.folder,
            gpu_processes=lambda: {0: {50}, 1: {51}, 2: {52} if occupied_gpu2 else set(), 3: set()},
            subprocess=SimpleNamespace(Popen=Process, check_output=lambda *a, **k: '', DEVNULL=-3, STDOUT=-2),
            save_json=lambda *a, **k: None,
            update_report=lambda *a, **k: None,
            event=lambda **k: None,
            seal=lambda cfg: None,
            verify=lambda cfg: dict(status='completed'),
            time=SimpleNamespace(time=lambda: 0, sleep=lambda _: None),
        )
        if fail_first:
            with self.assertRaisesRegex(RuntimeError, 'stopped without retry'):
                batch('coarse', jobs)
        else:
            self.assertEqual(len(batch('coarse', jobs)), len(jobs))
        return launched, peak

    def test_cv_uses_both_assigned_gpus(self):
        launched, peak = self.simulate(['cv'] * 5)
        self.assertEqual({gpu for gpu, _, _ in launched}, {2, 3})
        self.assertEqual(peak, {2: 1, 3: 1})
        self.assertTrue(all(command[2] == str(queue.WORKER) for _, _, command in launched))

    def test_nlp_capacity_remains_two_per_gpu(self):
        _, peak = self.simulate(['nlp'] * 7)
        self.assertEqual(peak, {2: 2, 3: 2})

    def test_external_gpu_occupancy_is_respected(self):
        launched, _ = self.simulate(['cv'] * 3, occupied_gpu2=True)
        self.assertEqual({gpu for gpu, _, _ in launched}, {3})

    def test_failure_stops_new_launches_without_retry(self):
        launched, _ = self.simulate(['cv'] * 5, fail_first=True)
        self.assertEqual(len(launched), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
