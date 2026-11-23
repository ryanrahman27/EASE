import numpy as np
from examples.ease_fm.direction_newbody import CommandResponse


def test_command_ablation_preserves_command_and_ignores_velocity():
    command=np.array([-.12,.03,.1]);original=command.copy();model=CommandResponse()
    a,n=model.predict(command,np.zeros(3));b,m=model.predict(command,np.ones(3)*100)
    np.testing.assert_array_equal(a,[-.08,.03,.08])
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(command,original)
    assert n==m==2
