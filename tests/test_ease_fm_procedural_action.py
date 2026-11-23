import unittest
import numpy as np
from examples.ease_fm.procedural_action_audit import clone,reference_step,branch,execute,Estimator
from examples.ease_fm.mixed_sim import sample_arm,MixedReach


class ProceduralActionTest(unittest.TestCase):
    def test_reference_switch_and_ramp(self):
        goals=[[1.,0.,0.],[0.,1.,0.]]
        np.testing.assert_allclose(reference_step(np.zeros(3),goals,49),[.002,0,0])
        np.testing.assert_allclose(reference_step(np.zeros(3),goals,50),[0,.002,0])
        np.testing.assert_allclose(reference_step(np.zeros(3),goals,100),[.002,0,0])

    def test_branch_matches_first_step_and_restores_goal(self):
        env=MixedReach(sample_arm(54111,'six','train'),synchronize=True)
        env.reset(0,[0.,0.,.9],spread=0)
        reference=env.data.site_xpos[0].copy();goals=[reference+[.01,0,0],reference+[0,.01,0]]
        q=env.data.qpos.copy();old_goal=env.goal.copy()
        result=branch(env,Estimator(None,6),reference.copy(),goals,49)
        np.testing.assert_array_equal(env.data.qpos,q);np.testing.assert_array_equal(env.goal,old_goal)
        self.assertEqual(env.steps,0)
        _,actual=execute(env,Estimator(None,6),reference,goals,49)
        self.assertEqual(result['1'],actual)


if __name__=='__main__':unittest.main()
