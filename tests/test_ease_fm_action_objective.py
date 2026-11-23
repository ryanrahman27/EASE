import unittest
from types import SimpleNamespace
import mujoco
import numpy as np
from examples.ease_fm.action_objective_audit import cloned_state,operator
from examples.ease_fm.action_objective_report import agreement


class ActionObjectiveTest(unittest.TestCase):
    def test_agreement_averages_cases_and_excludes_ties(self):
        def row(case,reverse=False,tie=False):
            return dict(case=case,candidates=[dict(method='a',scores={'s':0.},branch={'1':dict(error_squared=2. if reverse else 0.,collision=False)}),dict(method='b',scores={'s':0. if tie else 1.},branch={'1':dict(error_squared=1.,collision=False)})])
        result=agreement([row(0),row(0),row(1,True),row(2,tie=True)],'s',1)
        self.assertEqual(result['case_mean'],.5)
        self.assertEqual(result['non_tied_pairs'],3)
        self.assertEqual(result['clusters'],2)

    def test_clone_restores_physics_and_flags_on_exception(self):
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom type="sphere" size=".1" mass="1"/></body></worldbody></mujoco>')
        data=mujoco.MjData(model);mujoco.mj_step(model,data)
        env=SimpleNamespace(model=model,data=data,steps=9,bad=False,grasped=True,lifted=False)
        initial=data.qpos.copy();velocity=data.qvel.copy()
        with self.assertRaisesRegex(RuntimeError,'intentional'):
            with cloned_state(env):
                self.assertIsNot(env.data,data)
                np.testing.assert_array_equal(env.data.qpos,initial)
                mujoco.mj_step(model,env.data);env.steps=500;env.bad=True;env.grasped=False
                raise RuntimeError('intentional')
        self.assertIs(env.data,data);np.testing.assert_array_equal(data.qpos,initial)
        np.testing.assert_array_equal(data.qvel,velocity)
        self.assertEqual((env.steps,env.bad,env.grasped,env.lifted),(9,False,True,False))

    def test_operator_matches_executed_controller_formula(self):
        rng=np.random.default_rng(53);jac=rng.normal(size=(6,7));error=rng.normal(size=6)
        expected=jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(6),.35*error)
        np.testing.assert_allclose(operator(jac)@(.35*error),expected,atol=1e-12)


if __name__=='__main__':unittest.main()
