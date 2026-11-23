import numpy as np
import torch
from examples.ease_fm.task_retention_training import action_tensor


def test_retention_action_matches_executed_controller_before_geometry_filter():
    rng=np.random.default_rng(99);pred=rng.normal(size=(2,7,6))*.2;mask=np.array([[True]*6+[False],[True]*7]);pred[0,6]=0
    error=rng.normal(size=(2,6))*.2;bias=rng.normal(size=(2,7))*.003
    actual=action_tensor(torch.tensor(pred),torch.tensor(error),torch.tensor(bias),torch.tensor(mask)).numpy()
    for i in range(2):
        n=int(mask[i].sum());j=pred[i,:n].T;expected=np.clip(j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*error[i])+bias[i,:n],-.04,.04)
        np.testing.assert_allclose(actual[i,:n],expected,rtol=1e-10,atol=1e-12)
    assert actual[0,6]==0


def test_retention_action_gradients_are_finite():
    pred=(torch.randn(3,7,6)*.1).requires_grad_();mask=torch.ones(3,7,dtype=torch.bool)
    action_tensor(pred,torch.ones(3,6)*.001,torch.zeros(3,7),mask).square().sum().backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum()>0
