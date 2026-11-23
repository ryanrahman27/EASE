import numpy as np
from examples.ease_fm.geometry_transfer import slow_action


def test_slowing_preserves_bias_and_command_limits():
    bias=np.array([.02,-.01,0.])
    np.testing.assert_array_equal(slow_action(bias,bias),bias)
    np.testing.assert_allclose(slow_action(np.array([.04,-.03,.02]),bias),[.03,-.02,.01])
    assert np.max(np.abs(slow_action(np.array([2.,-2.]),np.array([1.,-1.]))))<=.04
