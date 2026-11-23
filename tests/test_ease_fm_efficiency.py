import unittest
from examples.ease_fm.factorial_replication_report import efficiency


class InteractionCostTest(unittest.TestCase):
    def test_probe_cost_and_reuse(self):
        records = [dict(body_seed=1, condition='same', success=True, steps=20),
                   dict(body_seed=1, condition='same', success=False, steps=150)]
        history = efficiency(records, 'correct')
        self.assertEqual(history['single_use_steps'], 234)
        self.assertEqual(history['reused_steps'], 202)
        self.assertEqual(history['success_within_budget']['50'], 0)
        self.assertEqual(history['success_within_budget']['100'], 1)
        empty = efficiency(records, 'empty')
        self.assertEqual(empty['single_use_steps'], 170)
        self.assertEqual(empty['reused_steps'], 170)
        self.assertEqual(empty['success_within_budget']['25'], 1)

    def test_no_cheap_early_failures(self):
        with self.assertRaises(AssertionError):
            efficiency([dict(body_seed=1, condition='same', success=False, steps=2)], 'reactive')


if __name__ == '__main__': unittest.main()
