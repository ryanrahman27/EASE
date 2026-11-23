from pathlib import Path
import unittest
import numpy as np
from examples.ease_fm.real_arms import RealArm

ASSETS=Path('outputs/ease_real_arm_assets_v1')


@unittest.skipUnless((ASSETS/'MANIFEST.json').exists(),'Pinned models not downloaded')
class RealArmTest(unittest.TestCase):
    def test_native_mapping_limits_and_gripper(self):
        for name,n in [('piper',6),('panda',7)]:
            env=RealArm(ASSETS,name);self.assertEqual(env.n,n)
            obs=env.observe();self.assertEqual(len(obs['q']),n)
            before=obs['q'].copy();_,command,_=env.step(np.ones(n)*100,0.)
            self.assertLessEqual(np.max(np.abs(command)),.040000001)
            self.assertTrue(np.all(env.data.ctrl[env.aids]<=env.limits[:,1]))
            self.assertTrue(np.all(env.data.ctrl[env.aids]>=env.limits[:,0]))
            env.reset(openness=0);closed=env.opening();env.reset(openness=1)
            self.assertGreater(env.opening(),closed+.03)

    def test_native_pose_jacobian_matches_measurements(self):
        for name in ['piper','panda']:
            env=RealArm(ASSETS,name);before=env.observe();q=before['q'];jac=env.jacobian()
            for i in range(env.n):
                changed=q.copy();changed[i]+=1e-6
                after=env.reset(changed)
                dx=np.r_[after['position']-before['position'],.15*(after['rotation'][:,2]-before['rotation'][:,2])]/1e-6
                np.testing.assert_allclose(jac[:,i],dx,atol=2e-6)


if __name__=='__main__':unittest.main()
