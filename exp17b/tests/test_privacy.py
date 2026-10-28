import unittest
from exp17b.privacy import calibrate
from opacus.accountants import RDPAccountant


class PrivacyTests(unittest.TestCase):
    def test_both_horizons_and_empty_step(self):
        results = []
        for stage in ('search', 'final'):
            calibration = calibrate(stage)
            accountant = RDPAccountant()
            # Every logical step counts, even if its Poisson draw is empty.
            for _ in range(calibration['steps']):
                accountant.step(noise_multiplier=calibration['noise_multiplier'],
                                sample_rate=calibration['sample_rate'])
            self.assertAlmostEqual(accountant.get_epsilon(1e-5),
                                   calibration['actual_epsilon'])
            self.assertLessEqual(calibration['actual_epsilon'], 8)
            results.append(calibration['noise_multiplier'])
        self.assertNotEqual(*results)


if __name__ == '__main__':
    unittest.main()
