from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from examples.ease_fm.novelaxis_paths import sample_path


def test_path_check_detects_interior_collision_with_clear_endpoints():
    class Env:
        n=1
        model=None
        data=SimpleNamespace(qpos=np.array([0.,.02,.03]))
        def contacts(self):
            return set(),[['link','table']] if abs(self.data.qpos[0]-.5)<.1 else []
        def clearance(self,q,fingers):
            return min(.002,abs(q[0]-.5)-.1)
    env=Env()
    with patch('examples.ease_fm.novelaxis_paths.mujoco.mj_forward'):
        result=sample_path(env,np.array([0.]),np.array([1.]))
    assert len(result['samples'])==101
    assert not result['samples'][0]['bad'] and not result['samples'][-1]['bad']
    assert result['samples'][50]['bad']
    assert not result['contact_free'] and not result['margin_2mm']
    np.testing.assert_array_equal(env.data.qpos[1:],[.02,.03])
