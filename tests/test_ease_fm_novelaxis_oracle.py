import numpy as np
from examples.ease_fm.novelaxis_oracle import ExactPose


def test_exact_jacobian_keeps_measured_bias_and_statistics():
    class Env:
        n=6
        def jacobian(self):return np.arange(36).reshape(6,6)
    c=ExactPose(Env())
    before=dict(q=np.zeros(6),position=np.zeros(3),rotation=np.eye(3))
    after=dict(q=np.full(6,.01),position=np.ones(3)*.002,rotation=np.eye(3))
    c.update(before,after,np.full(6,.02))
    gram=c.gram.copy();cross=c.cross.copy();bias=c.bias.copy()
    jac,returned_bias=c.predict()
    np.testing.assert_array_equal(jac,Env().jacobian())
    np.testing.assert_allclose(returned_bias,np.full(6,.002))
    np.testing.assert_array_equal(c.gram,gram)
    np.testing.assert_array_equal(c.cross,cross)
    np.testing.assert_array_equal(c.bias,bias)
    assert c.calls==1
