import unittest
from dataclasses import replace
from examples.ease_fm.ambiguous_fixture import PAIRS, pair
from examples.ease_fm.factorial_data import home, check_pair


class FactorialTests(unittest.TestCase):
    def test_same_joint_homes_preserve_twin_ambiguity_and_rank(self):
        for split in ('train', 'validation'):
            for config in PAIRS[split]:
                original = pair(split, config, 0)
                for shifted in (False, True):
                    arms = [replace(a, home=home(a.dof, shifted)) for a in original]
                    check_pair(arms)
                    self.assertEqual(arms[0].home, arms[1].home)


if __name__ == '__main__': unittest.main()
