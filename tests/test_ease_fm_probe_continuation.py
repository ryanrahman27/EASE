from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import mujoco
import pytest
from examples.ease_fm.probe_continuation import continuation_prediction
from examples.ease_fm.local_history import LocalPose


def test_analytical_switch_requires_active_handoff_and_preserves_bias():
    c=SimpleNamespace(gram=2*np.eye(6),cross=4*np.eye(6),bias=np.arange(6))
    sentinel=(np.zeros((6,6)),c.bias)
    with patch.object(LocalPose,'predict',return_value=sentinel) as learned:
        assert continuation_prediction(c,None,'analytical',False,None) is sentinel
        j,b=continuation_prediction(c,None,'analytical',True,None)
        np.testing.assert_array_equal(j,2*np.eye(6));assert b is c.bias and learned.call_count==1


def test_exact_access_is_scoped_to_active_branch_and_restores_guard():
    c=SimpleNamespace(bias=np.zeros(6));env=SimpleNamespace(jacobian=lambda:mujoco.mj_jacSite())
    with patch.object(LocalPose,'predict',return_value=('learned',c.bias)),patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('blocked')):
        assert continuation_prediction(c,env,'exact',False,None)[0]=='learned'
        j,b=continuation_prediction(c,env,'exact',True,lambda:np.eye(6))
        np.testing.assert_array_equal(j,np.eye(6));assert b is c.bias
        with pytest.raises(RuntimeError,match='blocked'):mujoco.mj_jacSite()
