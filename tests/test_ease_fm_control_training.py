import unittest
import torch
from examples.ease_fm.control_training import batch, losses, response


class ControlTrainingTest(unittest.TestCase):
    def test_matched_batches_mask_and_rank(self):
        a=batch(16,'cpu',torch.Generator().manual_seed(8))
        b=batch(16,'cpu',torch.Generator().manual_seed(8))
        for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
        g,c,m,t,j=a
        self.assertEqual(set(m.sum(-1).tolist()),{6,7})
        self.assertTrue(torch.all(j[~m]==0))
        self.assertTrue(torch.all(torch.linalg.svdvals(j[:,:,3:])[:,-1]<1e-6))
        self.assertTrue(torch.isfinite(torch.linalg.solve(g,c)).all())

    def test_control_loss_optimum_and_gradient(self):
        _,_,m,_,j=batch(12,'cpu',torch.Generator().manual_seed(9))
        self.assertAlmostEqual(float(losses(j,j,j,m,j)[2]),0.)
        p=(j+.01*torch.randn_like(j)).detach().requires_grad_(True)
        loss=losses(p,j,j,m,j)[2];self.assertGreater(float(loss.detach()),0.)
        loss.backward();self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.abs().sum()),0.)
        self.assertTrue(torch.isfinite(response(j)).all())


if __name__=='__main__':unittest.main()
