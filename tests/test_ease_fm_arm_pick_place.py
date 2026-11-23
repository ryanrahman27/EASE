import unittest
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from examples.ease_fm.arm_pick_place import PickPlace,direction_jacobian
from examples.ease_fm.mixed_sim import sample_arm
from examples.ease_fm.pick_place_control import PoseController
from examples.ease_fm.motion_transformer import MotionTransformer


class ArmPickPlaceTest(unittest.TestCase):
    def test_direction_jacobian_matches_finite_difference(self):
        env=PickPlace(sample_arm(2,'seven','train'));q=np.array([.2,.4,-.3,-.6,.2,.4,-.1]);env.reset(q)
        jac=env.jacobian();before=env.observe()
        for i in range(env.n):
            changed=q.copy();changed[i]+=1e-6;env.reset(changed);after=env.observe()
            difference=np.r_[after['position']-before['position'],.15*(after['rotation'][:,2]-before['rotation'][:,2])]/1e-6
            np.testing.assert_allclose(jac[:,i],difference,atol=2e-6)

    def test_neural_pose_predict_uses_no_inverse_and_updates_no_weights(self):
        model=MotionTransformer().eval();controller=PoseController(6,model)
        before=dict(q=np.zeros(6),position=np.zeros(3),rotation=np.eye(3))
        after=dict(q=np.ones(6)*.001,position=np.ones(3)*.002,rotation=np.eye(3))
        weights={k:v.clone() for k,v in model.state_dict().items()}
        controller.update(before,after,np.ones(6)*.002)
        with patch.object(np.linalg,'solve',side_effect=AssertionError('inverse')),patch.object(torch.linalg,'solve',side_effect=AssertionError('inverse')):
            jac,bias=controller.predict()
        self.assertEqual(jac.shape,(6,6));self.assertTrue(np.isfinite(jac).all())
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,weights[k]))

    def test_gripper_fingers_extend_away_from_terminal_link(self):
        env=PickPlace(sample_arm(2,'six','train'));env.reset(np.zeros(6))
        ee=mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_SITE,'ee')
        self.assertGreater(env.data.site_xpos[env.site,2],env.data.site_xpos[ee,2])


if __name__=='__main__':unittest.main()
