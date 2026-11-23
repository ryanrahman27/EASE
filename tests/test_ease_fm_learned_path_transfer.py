from types import SimpleNamespace
import numpy as np
import pytest
from examples.ease_fm.learned_path_transfer import route_check


def fixture_env():
    data=SimpleNamespace(qpos=np.array([.2,.3,.01,.02]),qvel=np.zeros(4),ctrl=np.zeros(4),time=1.)
    env=SimpleNamespace(n=2,data=data)
    env.observe=lambda:dict(q=data.qpos[:2].copy())
    env.clearance=lambda q,fingers: .002 if q[0]<.6 else -.001
    return env


def test_recheck_uses_actual_start_and_retains_blocked_samples():
    env=fixture_env();route,clearances=route_check(env,np.array([1.,.5]))
    np.testing.assert_array_equal(route[0],[.2,.3])
    np.testing.assert_array_equal(route[-1],[1.,.5])
    assert len(route)==len(clearances)==101 and min(clearances)<0
    np.testing.assert_array_equal(env.data.qpos,[.2,.3,.01,.02])


def test_recheck_detects_mutated_physical_state():
    env=fixture_env()
    def bad_clearance(q,fingers):
        env.data.qvel[0]=1.
        return .002
    env.clearance=bad_clearance
    with pytest.raises(AssertionError):route_check(env,np.array([1.,.5]))
