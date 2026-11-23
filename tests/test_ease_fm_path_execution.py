from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from examples.ease_fm.novelaxis_path_execution import PathArm


def test_route_waits_for_waypoint_and_clips_command():
    env=object.__new__(PathArm);env.phase='descend';env.waypoint=0
    env.route=np.tile([.2,-.2],(101,1));env.path_log=[]
    env.controller=SimpleNamespace(bias=np.array([.01,-.01]));env.observe=lambda:dict(q=np.zeros(2))
    with patch('examples.ease_fm.novelaxis_path_execution.GeometryArm.step',return_value='result') as step:
        assert env.step(np.zeros(2),0.)=='result'
    np.testing.assert_array_equal(step.call_args.args[0],[.04,-.04])
    assert env.waypoint==0


def test_route_does_not_replace_later_phase_actions():
    env=object.__new__(PathArm);env.phase='lift';env.route=np.zeros((101,2))
    action=np.array([.01,.02])
    with patch('examples.ease_fm.novelaxis_path_execution.GeometryArm.step') as step:
        env.step(action,.035)
    assert step.call_args.args[0] is action
