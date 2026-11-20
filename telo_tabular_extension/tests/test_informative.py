import unittest
import numpy as np
from telo_tabular.informative import class_rates, informative_mask, RATE


class InformativeMechanism(unittest.TestCase):
    def test_overall_rate_is_kept_and_classes_separate(self):
        y = np.array([0] * 6500 + [1] * 3500); d = 20
        for level in (0.0, 0.5, 1.0, 2.0):
            p0, p1 = class_rates(y, level, d)
            self.assertAlmostEqual(0.65 * p0 + 0.35 * p1, RATE, places=12)
            self.assertAlmostEqual((p1 - p0) * np.sqrt(d / (RATE * (1 - RATE))), level, places=12)
            mask = informative_mask(y, d, (p0, p1), seed=3)
            self.assertAlmostEqual(mask.mean(), RATE, delta=0.01)
            self.assertAlmostEqual(mask[y == 1].mean() - mask[y == 0].mean(), p1 - p0, delta=0.01)

    def test_strength_zero_is_class_independent(self):
        y = np.array([0, 1] * 50); self.assertEqual(*class_rates(y, 0.0, 8))

    def test_infeasible_strength_is_refused(self):
        with self.assertRaises(ValueError): class_rates(np.array([0, 1] * 50), 9.0, 2)


if __name__ == "__main__":
    unittest.main()
