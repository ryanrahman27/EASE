import unittest
from pathlib import Path
import numpy as np
from examples.ease_fm.pose_geometry import rotation_vector,upright_target
from examples.ease_fm.angular_control import AngularArm,AngularPose


class RotationGeometryTest(unittest.TestCase):
    def test_log_small_and_half_turn(self):
        for angle in [0.,1e-9,.3,np.pi-1e-8]:
            c,s=np.cos(angle),np.sin(angle);matrix=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            np.testing.assert_allclose(rotation_vector(matrix),[0,0,angle],atol=1e-8)
        r=upright_target(np.eye(3),True);np.testing.assert_allclose(r.T@r,np.eye(3))
        self.assertAlmostEqual(np.linalg.det(r),1.)

    @unittest.skipUnless(Path('outputs/ease_real_arm_assets_v1/MANIFEST.json').exists(),'Native assets absent')
    def test_world_angular_increment_matches_native_jacobian(self):
        for name,xy in [('piper',(.21,-.02)),('panda',(.32,-.02))]:
            env=AngularArm('outputs/ease_real_arm_assets_v1',name,xy,(xy[0],.02),.2,True)
            before=env.observe();jac=env.jacobian()
            for j in range(env.n):
                q=before['q'].copy();q[j]+=1e-6;after=env.reset(q)
                measured=np.r_[after['position']-before['position'],.15*rotation_vector(after['rotation']@before['rotation'].T)]/1e-6
                np.testing.assert_allclose(measured,jac[:,j],atol=3e-6)
                controller=AngularPose(env.n);controller.update(before,after,np.ones(env.n)*.001)
                np.testing.assert_allclose(controller.cross,np.outer(after['q']-before['q'],measured*1e-6),atol=1e-15)


if __name__=='__main__':unittest.main()
