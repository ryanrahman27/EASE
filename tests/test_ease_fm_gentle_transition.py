import numpy as np
from examples.ease_fm.gentle_transition import ramp_action

def test_ramp_preserves_bias_and_eventually_exact_original_command():
    bias=np.array([.003,-.002]);action=np.array([.03,-.04])
    np.testing.assert_array_equal(ramp_action(bias,bias,0),bias)
    for i in (19,20,100):np.testing.assert_array_equal(ramp_action(action,bias,i),action)
    scales=[np.linalg.norm(ramp_action(action,bias,i)-bias) for i in range(20)]
    assert all(a<b for a,b in zip(scales,scales[1:]))

def test_ramp_enforces_command_bounds_with_large_measured_bias():
    result=ramp_action(np.array([.04,-.04]),np.array([.2,-.2]),0)
    assert np.all(np.abs(result)<=.04)
