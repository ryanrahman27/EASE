import unittest
import mujoco
import numpy as np
import torch
from examples.ease_fm.full_control import recovery_chunk, commands
from examples.ease_fm.mixed_sim import sample_arm, MixedReach
from examples.ease_fm.joint_policy import JointFlowPolicy


class FullControlTests(unittest.TestCase):
    def test_synchronized_cartesian_observation_and_error(self):
        for family in ['six','seven','six_novel']:
            env=MixedReach(sample_arm(321,family,'train'),synchronize=True)
            env.reset(9,[.1,.1,.85])
            for _ in range(5):
                observation,_,info=env.step(np.full(env.arm.dof,.02))
                mujoco.mj_forward(env.model,env.data)
                np.testing.assert_allclose(observation['task'],env.observe()['task'],atol=1e-7)
                self.assertAlmostEqual(info['error'],np.linalg.norm(env.data.site_xpos[0]-env.goal),places=12)

    def test_recovery_clone_preserves_source_and_masks_horizon(self):
        env=MixedReach(sample_arm(123,'six','train'), synchronize=True)
        obs=env.reset(5,[.1,.1,.85])
        for _ in range(3):
            obs,_,_=env.step(np.full(6,.01))
        env.steps=148
        q=env.data.qpos.copy(); dq=env.data.qvel.copy(); ctrl=env.data.ctrl.copy()
        expected=np.clip(env.data.qpos+env.expert(),-2.5,2.5)-env.data.qpos
        target,valid,collision=recovery_chunk(env,obs)
        np.testing.assert_allclose(target[0,:6]*.04,expected,atol=1e-7)
        self.assertEqual(int(valid.sum()),12)
        self.assertFalse(valid[:,6].any())
        self.assertFalse(valid[2:].any())
        np.testing.assert_array_equal(env.data.qpos,q)
        np.testing.assert_array_equal(env.data.qvel,dq)
        np.testing.assert_array_equal(env.data.ctrl,ctrl)
        self.assertEqual(env.steps,148)

    def test_direct_actions_ignore_sampling_noise(self):
        torch.set_num_threads(2)
        model=JointFlowPolicy(history=True).eval()
        q=torch.randn(2,7,2); task=torch.randn(2,6)
        mask=torch.tensor([[True]*6+[False],[True]*7])
        z=torch.randn(2,7,64)
        with torch.no_grad():
            a=commands(model,'direct',q,task,mask,torch.randn(2,16,7),z)
            b=commands(model,'direct',q,task,mask,torch.randn(2,16,7)*100,z)
        torch.testing.assert_close(a,b)
        self.assertEqual(a[0,-1].item(),0)
        self.assertLessEqual(a.abs().max().item(),.040001)


if __name__=='__main__':
    unittest.main()
