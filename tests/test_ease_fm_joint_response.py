import unittest
import numpy as np
from examples.ease_fm.joint_response_audit import features,fit,predict


class JointResponseTest(unittest.TestCase):
    def test_shared_features_have_no_joint_count_or_identity(self):
        a=np.arange(7)*.01
        seven=features(a,a+1,a+2,a+3,a+4,a+5)
        six=features(a[:6],a[:6]+1,a[:6]+2,a[:6]+3,a[:6]+4,a[:6]+5)
        np.testing.assert_array_equal(seven[:6],six)
        np.testing.assert_array_equal(seven[:,0],a)
        np.testing.assert_array_equal(seven[:,7],a+5)

    def test_regression_recovers_response_without_eval_normalization(self):
        rng=np.random.default_rng(59);x=rng.normal(size=(1000,3));x[:,2]=2
        y=.3*x[:,0]-.2*x[:,1]+.1;model=fit(x,y)
        held=rng.normal(size=(100,3));held[:,2]=2
        expected=.3*held[:,0]-.2*held[:,1]+.1
        self.assertLess(np.sqrt(np.mean((predict(model,held)-expected)**2)),.001)
        first=predict(model,held[:1])
        np.testing.assert_allclose(first,predict(model,np.vstack([held[:1],np.full((10,3),1e6)]))[:1],atol=1e-14,rtol=1e-14)


if __name__=='__main__':unittest.main()
