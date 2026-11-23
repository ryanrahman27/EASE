import unittest
import numpy as np
import torch
from examples.ease_fm.response_objective import corrected_loss,load_contexts
from examples.ease_fm.local_action_training import local_loss
from examples.ease_fm.control_training import response


class ResponseObjectiveTest(unittest.TestCase):
    def test_identity_and_differentiable_affine_response(self):
        torch.manual_seed(61)
        p=(torch.randn(3,7,6)*.15).requires_grad_();j=torch.randn(3,7,6)*.15
        mask=torch.arange(7)[None]<torch.tensor([6,7,6])[:,None]
        e=torch.randn(3,6)*.01;b=torch.randn(3,7)*.003;q=torch.zeros(3,7)
        torch.testing.assert_close(corrected_loss(p,j,mask,(e,b,q,mask.float(),-b)),local_loss(p,j,mask,(e,b,q)))
        alpha=torch.full_like(b,.4);offset=torch.full_like(b,.001)
        loss=corrected_loss(p,j,mask,(e,b,q,alpha,offset));loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0)
        command=((response(p.detach())@(.35*e[:,:,None])).squeeze(-1)+b).clamp(-.04,.04)
        dq=(alpha*command+offset).masked_fill(~mask,0)
        residual=e-torch.einsum('bnk,bn->bk',j,dq)
        torch.testing.assert_close(loss,(residual.square().sum(-1)/(e.square().sum(-1)+.002**2)).mean())


if __name__=='__main__':unittest.main()
