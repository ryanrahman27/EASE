import unittest
from collections import deque
import numpy as np
import torch
from examples.ease_fm.robust_training import synthetic_batch
from examples.ease_fm.robustness import measure, window_statistics
from examples.ease_fm.kinematic_control import IdentifiedController


class RobustnessTest(unittest.TestCase):
    def test_training_batch_has_valid_masked_statistics_and_gradients(self):
        from examples.ease_fm.motion_transformer import MotionTransformer
        torch.manual_seed(9)
        profiles=torch.randn(8,32,7)*.008
        g,c,mask,target=synthetic_batch(profiles,16,'cpu')
        self.assertTrue(torch.all(torch.linalg.eigvalsh(g)>0))
        self.assertTrue(torch.isfinite(target).all())
        self.assertEqual(target[~mask].abs().sum().item(),0)
        model=MotionTransformer();prediction=model(g,c,mask)
        loss=((prediction-target)**2).mean();loss.backward()
        self.assertGreater(model.input.weight.grad.abs().sum().item(),0)

    def test_measurement_is_reproducible_and_does_not_change_truth_or_goal(self):
        obs=dict(joints=np.zeros((6,2)),task=np.arange(6,dtype=float))
        a=measure(obs,np.random.default_rng(3),.005)
        b=measure(obs,np.random.default_rng(3),.005)
        np.testing.assert_array_equal(a['task'],b['task'])
        np.testing.assert_array_equal(a['task'][3:],obs['task'][3:])
        np.testing.assert_array_equal(obs['joints'],np.zeros((6,2)))
        self.assertGreater(np.linalg.norm(a['joints'][:,0]),0)

    def test_window_discards_old_transitions_and_matches_online_update(self):
        rng=np.random.default_rng(2);history=deque(maxlen=4)
        for _ in range(7):history.append((rng.normal(size=6)*.01,rng.normal(size=3)*.01,rng.normal(size=6)*.01))
        controller=IdentifiedController(np.zeros((1,6,11)),True)
        before=dict(joints=np.zeros((6,2)),task=np.zeros(6))
        for dq,dx,command in history:
            after=dict(joints=np.stack([dq,np.zeros(6)],-1),task=np.r_[dx,np.zeros(3)])
            controller.update(before,after,command)
        g,c,b=window_statistics(history,6)
        np.testing.assert_allclose(g,controller.gram)
        np.testing.assert_allclose(c,controller.cross)
        np.testing.assert_allclose(b,controller.bias)


if __name__=='__main__':unittest.main()
