import unittest
import mujoco
import numpy as np
from examples.ease_fm.mixed_sim import sample_arm,MixedReach
from examples.ease_fm.procedural_pose_audit import observe,vector,reference_step,targets,branch,execute
from examples.ease_fm.retention_fresh import GuardedPose
from examples.ease_fm.procedural_action_audit import clone
from examples.ease_fm.arm_pick_place import direction_jacobian


class ProceduralPoseTest(unittest.TestCase):
    def test_pose_jacobian_and_goal_generation(self):
        for family in ['six','seven']:
            env=MixedReach(sample_arm(55100,family,'train'),synchronize=True);env.reset(0,[0,0,.9],spread=0)
            before=observe(env);initial=vector(before)
            goals=targets(env,55100);self.assertEqual(goals,targets(env,55100))
            np.testing.assert_array_equal(env.data.qpos,before['q'])
            jp=np.zeros((3,env.arm.dof));jr=np.zeros_like(jp);mujoco.mj_jacSite(env.model,env.data,jp,jr,0)
            jac=np.concatenate([jp,.15*direction_jacobian(before['rotation'],jr)])
            dq=np.linspace(-1,1,env.arm.dof)*1e-7
            with clone(env):
                env.data.qpos[:]+=dq;mujoco.mj_forward(env.model,env.data)
                np.testing.assert_allclose(vector(observe(env))-initial,jac@dq,atol=1e-12)
            ref=reference_step(initial,[g['pose'] for g in goals],0)
            self.assertAlmostEqual(np.linalg.norm(ref[3:]),.15)
            self.assertLessEqual(np.linalg.norm(ref[:3]-initial[:3]),.002000001)

    def test_pose_branch_first_step_and_restoration(self):
        env=MixedReach(sample_arm(55101,'six','train'),synchronize=True);env.reset(0,[0,0,.9],spread=0)
        initial=observe(env);ref=vector(initial);goals=[g['pose'] for g in targets(env,55101)]
        result=branch(env,GuardedPose(6,None),ref,goals,49)
        np.testing.assert_array_equal(env.data.qpos,initial['q']);self.assertEqual(env.steps,0)
        _,actual=execute(env,GuardedPose(6,None),ref,goals,49)
        self.assertEqual(result['1'],actual)


if __name__=='__main__':unittest.main()
