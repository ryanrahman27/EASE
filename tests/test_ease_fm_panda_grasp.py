import unittest
from pathlib import Path
import numpy as np
from examples.ease_fm.panda_grasp import GraspArm,OrientationController,nearest_cube_axis


@unittest.skipUnless(Path('outputs/ease_real_arm_assets_v1/MANIFEST.json').exists(),'Native assets absent')
class PandaOrientationTest(unittest.TestCase):
    def test_full_jacobian_matches_measured_axes(self):
        env=GraspArm('outputs/ease_real_arm_assets_v1','panda',(.31,0),(.31,.04),.2,True)
        before=env.observe();jac=env.full_jacobian()
        for j in range(env.n):
            q=before['q'].copy();q[j]+=1e-6;after=env.reset(q)
            dx=np.r_[after['position']-before['position'],.15*(after['rotation'][:,2]-before['rotation'][:,2]),
                .15*(after['rotation'][:,0]-before['rotation'][:,0])]/1e-6
            np.testing.assert_allclose(dx,jac[:,j],atol=2e-6)

    def test_measured_estimator_recovers_full_motion(self):
        rng=np.random.default_rng(42);jac=rng.normal(size=(9,7))*.1
        controller=OrientationController(7)
        for _ in range(300):
            dq=rng.normal(size=7)*.01;dx=jac@dq
            before=dict(q=np.zeros(7),position=np.zeros(3),rotation=np.eye(3))
            rotation=np.eye(3);rotation[:,2]+=dx[3:6]/.15;rotation[:,0]+=dx[6:]/.15
            controller.update(before,dict(q=dq,position=dx[:3],rotation=rotation),dq)
        estimated,_=controller.predict();np.testing.assert_allclose(estimated,jac,atol=2e-4)
        np.testing.assert_array_equal(nearest_cube_axis(np.eye(3)),[1,0,0])


if __name__=='__main__':unittest.main()
