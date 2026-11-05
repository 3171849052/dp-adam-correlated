import unittest,importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('candidate',Path(__file__).with_name('scheduler_candidate.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class Admission(unittest.TestCase):
    def test_idle_target(self):
        self.assertEqual(module.admission_limit(12288,0,0,dict(worker_budget_mib=768,reserve_mib=1536,target_per_gpu=12)),12)
    def test_external_occupancy(self):
        p=dict(worker_budget_mib=768,reserve_mib=1536,target_per_gpu=12)
        self.assertEqual(module.admission_limit(12288,8000,0,p),3)
        self.assertEqual(module.admission_limit(12288,10000,2000,p),3)
    def test_own_resident_does_not_double_count(self):
        p=dict(worker_budget_mib=768,reserve_mib=1536,target_per_gpu=12)
        self.assertEqual(module.admission_limit(12288,8500,8500,p),12)
if __name__=='__main__':unittest.main()
