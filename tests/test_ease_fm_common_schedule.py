import unittest
from types import SimpleNamespace
import numpy as np
from examples.ease_fm.native_common_schedule import make_schedule,summarize_phase


class CommonScheduleTest(unittest.TestCase):
    def test_fixed_schedule_uses_complete_budgets_and_bounded_ramp(self):
        env=SimpleNamespace(name='piper',xy=np.array([.24,0.]),target=np.array([.24,.04]))
        start=np.array([.24,0.,.30]);schedule=make_schedule(env,start)
        self.assertEqual(len(schedule),1700)
        points=np.vstack([start,[s['reference'] for s in schedule]])
        self.assertLessEqual(float(np.max(np.linalg.norm(np.diff(points,axis=0),axis=1))),.002+1e-12)
        np.testing.assert_array_equal(start,[.24,0.,.30])
        self.assertEqual(schedule,make_schedule(env,start))

    def test_requires_all_last_eight_samples(self):
        rows=[dict(goal_error=.001,direction_error=.01,tracking_error=.001,joint_speed=0.,clipped=False) for _ in range(10)]
        self.assertTrue(summarize_phase('lift',rows)['passes'])
        rows[-8]={**rows[-8],'goal_error':.007}
        self.assertFalse(summarize_phase('lift',rows)['passes'])
        self.assertFalse(summarize_phase('lift',rows[-7:])['passes'])


if __name__=='__main__':unittest.main()
