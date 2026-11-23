import unittest
import numpy as np
from examples.ease_fm.native_reliability import bounded_error


class TrajectoryReferenceTest(unittest.TestCase):
    def test_reference_reaches_goal_without_overshoot(self):
        start=np.array([.2,-.03,.32]);goal=np.array([.25,.04,.24]);position=start.copy()
        previous=np.linalg.norm(goal-position)
        for _ in range(200):
            step=bounded_error(goal,position,.002)
            self.assertLessEqual(np.linalg.norm(step),.0020000001)
            position+=step
            distance=np.linalg.norm(goal-position)
            self.assertLessEqual(distance,previous+1e-12);previous=distance
        np.testing.assert_allclose(position,goal,atol=1e-12)
        np.testing.assert_array_equal(bounded_error(goal,goal),np.zeros(3))


if __name__=='__main__':unittest.main()
