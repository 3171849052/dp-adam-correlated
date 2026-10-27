import unittest
from exp17.search import coarse, fine, rank, key
from exp17.common import METHODS, PROTOCOL


class WorkflowTests(unittest.TestCase):
    def test_grid_all_boundaries(self):
        c=coarse();self.assertEqual(len(c),144)
        for m in METHODS:self.assertEqual(sum(x['method']==m for x in c),36)
        for side in (0,1):
            results=[dict(config=x,accuracy=(x['lr'] if side else -x['lr'])+(x.get('r',0) if side else -x.get('r',0)),validation_loss=1.) for x in c]
            f=fine(results);self.assertEqual(len(f),48)
            for m in METHODS:
                self.assertEqual(sum(x['method']==m for x in f),12)
                for x in f:
                    self.assertTrue(1e-5<=x['lr']<=3e-3)
                    if 'r' in x:
                        lo,hi=PROTOCOL['r_ranges'][x['method']];self.assertTrue(lo<=x['r']<=hi)
            self.assertFalse({key(x) for x in c}&{key(x) for x in f})
        # Interior winning r needs a distinct non-central refinement point.
        results=[dict(config=x,accuracy=-abs(__import__('math').log(x['lr']/1e-4))-abs(__import__('math').log(x.get('r',.1)/.1)),validation_loss=1.) for x in c]
        self.assertEqual(len(fine(results)),48)
        self.assertEqual(rank(results,'adam')[0]['n'],3)
