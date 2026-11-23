import unittest
from unittest.mock import patch
import numpy as np
import torch
from examples.ease_fm.motion_transformer import MotionTransformer, MotionController, statistics


class MotionTransformerTest(unittest.TestCase):
    def test_mask_and_no_inverse_in_prediction(self):
        torch.manual_seed(3)
        model=MotionTransformer().eval()
        gram=torch.eye(7)[None]; cross=torch.randn(1,7,3)
        mask=torch.tensor([[True]*6+[False]])
        with torch.no_grad(), patch.object(torch.linalg,'solve',side_effect=AssertionError('No solve in neural prediction')):
            a=model(gram,cross,mask)
            gram[:,6,:]=1000; gram[:,:,6]=1000; cross[:,6]=1000
            b=model(gram,cross,mask)
        torch.testing.assert_close(a,b)
        torch.testing.assert_close(a[:,6],torch.zeros(1,3))

    def test_statistics_follow_joint_reordering(self):
        rng=np.random.default_rng(4)
        h=rng.normal(size=(32,6,11))
        h[:,:,8:11]=h[:,0:1,8:11]
        g,c,b=statistics(h)
        permutation=np.array([3,0,5,1,4,2])
        gp,cp,bp=statistics(h[:,permutation])
        np.testing.assert_allclose(gp,g[np.ix_(permutation,permutation)])
        np.testing.assert_allclose(cp,c[permutation])
        np.testing.assert_allclose(bp,b[permutation])

    def test_online_controller_updates_state_not_weights(self):
        model=MotionTransformer().eval()
        controller=MotionController(model,6)
        weights={k:v.clone() for k,v in model.state_dict().items()}
        before={'joints':np.zeros((6,2)), 'task':np.array([0,0,0,.1,.1,.1])}
        after={'joints':np.column_stack([np.full(6,.01),np.zeros(6)]), 'task':np.array([.01,.02,.03,.1,.1,.1])}
        with patch.object(torch.linalg,'solve',side_effect=AssertionError('No analytical J')):
            action=controller.action(before)
            controller.update(before,after,action)
            next_action=controller.action(after)
        self.assertEqual(action.shape,(6,))
        self.assertTrue(np.isfinite(next_action).all() and (np.abs(next_action)<=.04).all())
        self.assertTrue(np.any(controller.cross))
        for key,value in model.state_dict().items():torch.testing.assert_close(value,weights[key])
        with self.assertRaises(ValueError):MotionController(model,6,np.empty((0,6,11)))
