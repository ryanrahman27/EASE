import unittest
import numpy as np
from examples.ease_fm.kinematic_control import IdentifiedController


class IdentificationTest(unittest.TestCase):
    def test_measured_jacobian_and_bias(self):
        rng = np.random.default_rng(8)
        jac = rng.normal(size=(3, 6))
        dq = rng.normal(0, .02, (32, 6))
        bias = np.full(6, .003)
        history = np.zeros((32, 6, 11))
        history[:, :, 3] = dq
        history[:, :, 2] = (dq+bias)/.04
        history[:, :, 8:11] = (dq @ jac.T)[:, None, :]
        controller = IdentifiedController(history, False)
        np.testing.assert_allclose(controller.jac, jac, atol=.001)
        np.testing.assert_allclose(controller.bias, bias, atol=1e-10)
        self.assertTrue(np.isfinite(controller.action({'task': np.array([0,0,0,.1,.1,.1])})).all())

    def test_unexcited_probe_is_finite(self):
        controller = IdentifiedController(np.zeros((32, 6, 11)), True)
        obs = {'joints': np.zeros((6, 2)), 'task': np.zeros(6)}
        controller.update(obs, obs, np.zeros(6))
        np.testing.assert_array_equal(controller.action(obs), np.zeros(6))

    def test_update_ablations_are_isolated(self):
        before = {'joints': np.zeros((6, 2)), 'task': np.zeros(6)}
        after = {'joints': np.column_stack([np.full(6, .01), np.zeros(6)]),
                 'task': np.array([.01, .02, .03, 0, 0, 0])}
        for jacobian, bias in [(True, False), (False, True), (False, False)]:
            controller = IdentifiedController(np.zeros((32, 6, 11)), True, jacobian, bias)
            controller.update(before, after, np.full(6, .02))
            self.assertEqual(bool(np.any(controller.jac)), jacobian)
            self.assertEqual(bool(np.any(controller.bias)), bias)
