import unittest
from examples.ease_fm.ambiguous_fixture import PAIRS, audit, pair


class AmbiguousFixtureTests(unittest.TestCase):
    def test_split_axes_disjoint(self):
        groups = [{axis for a, b, _ in pairs for axis in (a, b)} for pairs in PAIRS.values()]
        for i, group in enumerate(groups):
            self.assertEqual(len(group), 4)
            for other in groups[i+1:]:
                self.assertFalse(group & other)
        self.assertFalse(set.union(*groups) & {'ZYXYXX', 'ZYYXXYZ'})

    def test_observation_ambiguity_and_response(self):
        for configuration in PAIRS['validation']:
            arms = pair('validation', configuration, 0)
            self.assertEqual(arms[0].lengths, arms[1].lengths)
            self.assertEqual(arms[0].masses, arms[1].masses)
            result = audit(arms)
            self.assertTrue(result['identical_initial_observations'])
            self.assertGreater(result['probe_max_ee_separation_m'], .001)


if __name__ == '__main__':
    unittest.main()
