import unittest
import numpy as np
from examples.ease_fm.broad_embodiment import axis_sequences, spatial_sequence, family_arms, VARIANTS


class BroadSuiteTest(unittest.TestCase):
    def test_structural_exclusions_and_reproducibility(self):
        self.assertFalse(spatial_sequence('YZZZZZ'))
        self.assertTrue(spatial_sequence('ZYYXYX'))
        excluded={'ZYYXYX','ZYXYYX'}
        a=axis_sequences(np.random.default_rng(3),excluded,2)
        b=axis_sequences(np.random.default_rng(3),excluded,2)
        self.assertEqual(a,b)
        self.assertEqual(len(set(a)),4)
        self.assertFalse(set(a)&excluded)
        self.assertEqual([len(s) for s in a],[6,6,7,7])

    def test_crossed_physics_changes_only_assigned_factors(self):
        arms=family_arms('ZYYXYX',0,np.random.default_rng(4))
        self.assertEqual(arms[0].lengths,arms[2].lengths)
        self.assertEqual(arms[1].lengths,arms[3].lengths)
        self.assertEqual(arms[0].masses,arms[2].masses)
        self.assertEqual(arms[0].strength,arms[1].strength)
        self.assertEqual(arms[2].strength,arms[3].strength)
        self.assertEqual(arms[2].damping,arms[3].damping)
        self.assertNotEqual(arms[0].lengths,arms[1].lengths)
