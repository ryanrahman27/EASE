import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from examples.ease_fm.local_refresh import refresh
from examples.ease_fm.pick_place_control import PoseController
from examples.ease_fm.split_refresh import SplitPose,CONDITIONS


class MeasuredArm:
    def __init__(self,collision_at=None):
        self.n=6;self.q=np.zeros(6);self.steps=0;self.bad=False
        self.obj=0;self.data=SimpleNamespace(xpos=np.array([[0.,0.,.215]]))
        self.collision_at=collision_at

    def observe(self):
        return dict(q=self.q.copy(),position=self.q[:3].copy(),rotation=np.eye(3))

    def contacts(self):return [],[]

    def step(self,action,aperture):
        assert aperture==0.
        self.q+=.8*action;self.steps+=1;self.bad=self.steps==self.collision_at
        return self.observe(),action,dict(bad=['contact'] if self.bad else [])


class LocalRefreshTest(unittest.TestCase):
    def test_factorial_updates_change_only_selected_state(self):
        controllers={};traces={}
        for condition in CONDITIONS:
            c=SplitPose(6,condition=condition);env=MeasuredArm()
            traces[condition]=c.hook(env,c,47);controllers[condition]=c
            self.assertFalse(c.in_refresh)
        for condition,c in controllers.items():
            self.assertEqual(traces[condition],traces['both'])
            for key in ['gram','cross','bias']:
                enabled=CONDITIONS[condition][1 if key=='bias' else 0]
                expected=controllers['both'].states['after'][key] if enabled else c.states['before'][key]
                self.assertEqual(c.states['after'][key],expected)
            self.assertEqual(len(c.diagnostics),32)
        # Outside refresh, even the neither condition must resume ordinary updates.
        c=controllers['neither'];before=dict(q=np.zeros(6),position=np.zeros(3),rotation=np.eye(3))
        after=dict(q=np.ones(6)*.01,position=np.ones(3)*.001,rotation=np.eye(3))
        c.update(before,after,np.ones(6)*.02)
        self.assertGreater(np.linalg.norm(c.cross),0.);self.assertGreater(np.linalg.norm(c.bias),0.)

    def test_identical_excitation_freezes_all_estimator_state(self):
        a,b=MeasuredArm(),MeasuredArm();ca,cb=PoseController(6),PoseController(6)
        initial=[x.copy() for x in [cb.gram,cb.cross,cb.bias]]
        with patch.object(ca,'predict',side_effect=AssertionError('probe must not use model predictions')):
            active=refresh(a,ca,'probe',47)
        frozen=refresh(b,cb,'frozen_probe',47)
        self.assertEqual(active,frozen);self.assertEqual(a.steps,32)
        for actual,expected in zip([cb.gram,cb.cross,cb.bias],initial):np.testing.assert_array_equal(actual,expected)
        self.assertGreater(np.linalg.norm(ca.cross),0.)
        self.assertGreater(np.linalg.norm(ca.bias),0.)

    def test_wait_budget_and_collision_stop(self):
        env=MeasuredArm();trace=refresh(env,PoseController(6),'wait',47)
        self.assertEqual(len(trace),32);np.testing.assert_array_equal(env.q,np.zeros(6))
        env=MeasuredArm(collision_at=3);trace=refresh(env,PoseController(6),'probe',47)
        self.assertEqual(len(trace),3);self.assertTrue(env.bad)


if __name__=='__main__':unittest.main()
