from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from examples.ease_fm.learned_handoff import HandoffPose


def test_handoff_switches_only_jacobian_and_preserves_measured_bias():
    exact=np.eye(6);learned=np.ones((6,6));env=SimpleNamespace(n=6,jacobian=lambda:exact)
    c=HandoffPose(env,None);c.bias[:]=.01
    gram=c.gram.copy();cross=c.cross.copy()
    with patch('examples.ease_fm.learned_handoff.GuardedPose.predict',return_value=(learned,c.bias)) as predict:
        assert c.predict()[0] is learned
        c.use_exact=True
        jac,bias=c.predict()
        assert jac is exact and bias is c.bias and predict.call_count==1
    np.testing.assert_array_equal(c.gram,gram);np.testing.assert_array_equal(c.cross,cross)
    np.testing.assert_array_equal(c.bias,np.full(6,.01))
