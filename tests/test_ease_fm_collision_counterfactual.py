import numpy as np
from examples.ease_fm.collision_counterfactual import DiagnosticArm
from examples.ease_fm.mixed_sim import sample_arm


def test_counterfactual_restores_physics_and_flags():
    env=DiagnosticArm(sample_arm(2,'six','train'));env.reset(np.zeros(6));env.prepare_geometry()
    env.data.qvel[:6]=.01
    q=env.data.qpos.copy();v=env.data.qvel.copy();ctrl=env.data.ctrl.copy();time=env.data.time
    flags=(env.steps,env.bad,env.grasped,env.lifted)
    result=env.branch(np.zeros(6),0.,zero_velocity=True)
    np.testing.assert_array_equal(env.data.qpos,q);np.testing.assert_array_equal(env.data.qvel,v);np.testing.assert_array_equal(env.data.ctrl,ctrl)
    assert env.data.time==time and (env.steps,env.bad,env.grasped,env.lifted)==flags
    assert np.isfinite(result['position']).all() and np.isfinite(result['min_actual_path_clearance'])
