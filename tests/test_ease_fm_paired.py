import unittest
import numpy as np
from examples.ease_fm.ambiguous_fixture import pair, PAIRS
from examples.ease_fm.paired_data import calibration
from examples.ease_fm.paired_experiment import gate


class PairedTests(unittest.TestCase):
    def test_twins_receive_same_commands_but_distinct_responses(self):
        for config in PAIRS['train']:
            a, b = pair('train', config, 0)
            ha, ca = calibration(a, 92)
            hb, cb = calibration(b, 92)
            np.testing.assert_array_equal(ca, cb)
            np.testing.assert_array_equal(ha[0, :, [0, 5, 6, 7]], hb[0, :, [0, 5, 6, 7]])
            self.assertGreater(np.linalg.norm(ha-hb), .001)

    def test_gate_requires_each_body(self):
        rows = [{'body_seed': 0, 'success': True} for _ in range(96)]
        rows += [{'body_seed': 1, 'success': False} for _ in range(4)]
        self.assertFalse(gate(rows)['passed'])


if __name__ == '__main__': unittest.main()
