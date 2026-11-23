import unittest
import numpy as np
import torch
from examples.ease_fm.local_action_training import action_context,local_loss
from examples.ease_fm.control_training import batch


class LocalActionTest(unittest.TestCase):
    def test_local_loss_matches_clipped_numpy_controller(self):
        _,_,m,_,j=batch(8,'cpu',torch.Generator().manual_seed(56))
        j=j.double();context=tuple(x.double() for x in action_context(j.float(),m,torch.Generator().manual_seed(57)))
        prediction=(j*1.1).detach().requires_grad_(True);actual=local_loss(prediction,j,m,context)
        values=[]
        for p,t,mask,e,b,q in zip(prediction.detach().numpy(),j.numpy(),m.numpy(),*[x.numpy() for x in context]):
            motion=p@np.linalg.solve(p.T@p+.015**2*np.eye(6),.35*e)
            command=np.clip(q+np.clip(motion+b,-.04,.04),-2.5,2.5)-q
            residual=e-t.T@np.where(mask,command-b,0)
            values.append((residual@residual)/(e@e+.002**2))
        self.assertAlmostEqual(float(actual.detach()),np.mean(values),places=10)
        actual.backward();self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()),0.)

    def test_action_context_reproducible_and_masked(self):
        _,_,m,_,j=batch(16,'cpu',torch.Generator().manual_seed(58))
        a=action_context(j,m,torch.Generator().manual_seed(59));b=action_context(j,m,torch.Generator().manual_seed(59))
        for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
        self.assertTrue(torch.all(a[1][~m]==0));self.assertTrue(torch.all(a[2][~m]==0))
        self.assertLessEqual(float(a[1].abs().max()),.02500001)


if __name__=='__main__':unittest.main()
