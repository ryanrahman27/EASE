import numpy as np
from examples.ease_fm.geometry_filter import GeometryArm,allowed_pair
from examples.ease_fm.mixed_sim import sample_arm


def test_geometry_queries_do_not_modify_physical_state():
    env=GeometryArm(sample_arm(2,'six','train'));env.reset(np.array([.2,.4,-.3,-.6,.2,.4]));env.prepare_geometry()
    before={k:getattr(env.data,k).copy() for k in ('qpos','qvel','ctrl','qacc','site_xpos')};margins=env.model.geom_margin.copy();time=env.data.time
    selected,info=env.select_action(np.ones(env.n)*.03,.035,np.ones(env.n)*.002)
    assert np.isfinite(selected).all() and np.max(np.abs(selected))<=.04 and 3<=info['queries']<=9
    for key,value in before.items():np.testing.assert_array_equal(getattr(env.data,key),value)
    np.testing.assert_array_equal(env.model.geom_margin,margins);assert env.data.time==time


def test_allowed_grasp_contacts_do_not_hide_arm_or_table_contact():
    ids=dict(cube=1,table=2,gripper_left_pad=3,gripper_right_pad=4)
    assert allowed_pair({1,2},ids) and allowed_pair({1,3},ids) and allowed_pair({1,4},ids)
    assert not allowed_pair({2,3},ids) and not allowed_pair({1,5},ids) and not allowed_pair({5,6},ids)


def test_full_scale_passes_action_through_exactly():
    env=GeometryArm(sample_arm(2,'six','train'));env.reset(np.zeros(6))
    env.clearance=lambda q,f:.002
    rng=np.random.default_rng(72);action=rng.uniform(-.04,.04,6);bias=rng.uniform(-.04,.04,6)
    selected,info=env.select_action(action,.035,bias)
    np.testing.assert_array_equal(selected,action)
    assert info['scale']==1. and not info['changed']
