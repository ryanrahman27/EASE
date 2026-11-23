import unittest
import numpy as np
from examples.ease_fm.motion_gap_audit import decomposition


class MotionGapTest(unittest.TestCase):
    def test_terms_isolate_known_errors_and_cancel(self):
        truth=np.eye(2);jac=2*truth
        terms=decomposition(jac,truth,np.array([1.,0.]),np.array([2.,0.]),np.array([2.,0.]))
        np.testing.assert_array_equal(terms['estimator'],[1,0])
        np.testing.assert_array_equal(terms['actuator'],[-1,0])
        np.testing.assert_array_equal(terms['nonlinear'],[0,0])
        np.testing.assert_array_equal(terms['total'],[0,0])

    def test_rectangular_motion_identity(self):
        rng=np.random.default_rng(58)
        for n in [6,7]:
            truth=rng.normal(size=(6,n));jac=rng.normal(size=(6,n))
            predicted=rng.normal(size=n);actual=rng.normal(size=n);dx=rng.normal(size=6)
            terms=decomposition(jac,truth,predicted,actual,dx)
            np.testing.assert_allclose(np.sum([terms[k] for k in ['estimator','actuator','nonlinear']],axis=0),jac@predicted-dx)


if __name__=='__main__':unittest.main()
