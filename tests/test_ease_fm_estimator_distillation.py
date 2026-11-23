import numpy as np
import torch
from examples.ease_fm.estimator_distillation import analytical_targets,fit_anchor


def test_analytical_targets_solve_active_blocks_without_padded_singularity():
    gen=torch.Generator().manual_seed(66)
    mask=torch.arange(7)[None]<torch.tensor([[6],[7]])
    x=torch.randn(2,20,7,generator=gen)*.002
    x=x.masked_fill(~mask[:,None],0)
    g=x.transpose(1,2)@x+torch.diag_embed(mask.float()*1e-6)
    c=torch.randn(2,7,6,generator=gen)*1e-5;c=c.masked_fill(~mask[:,:,None],0)
    target=analytical_targets(g,c,mask)
    for i,n in enumerate([6,7]):
        expected=np.linalg.solve(g[i,:n,:n].numpy().astype(float),c[i,:n].numpy().astype(float))
        np.testing.assert_allclose(target[i,:n].numpy(),expected,rtol=1e-6,atol=1e-7)
    assert torch.equal(target[0,6],torch.zeros(6))


def test_fit_anchor_ignores_inactive_joint_and_keeps_active_gradients():
    mask=torch.arange(7)[None]<6
    prediction=torch.zeros(1,7,6,requires_grad=True)
    target=torch.ones_like(prediction);target[:,6]=10000
    fit,anchor=fit_anchor(prediction,target,mask,torch.zeros_like(prediction))
    assert fit.item()==1 and anchor.item()==0
    (fit+anchor).backward()
    assert torch.count_nonzero(prediction.grad[:,6])==0
    assert torch.count_nonzero(prediction.grad[:,:6])==36
