"""Check the direct-task quadratic objective before evaluating interventions."""
import numpy as np
from .descent_progress import ROOT,prior
from .abstaining_task_value_gate import save

def run():
    assert not any((ROOT/c/'records.json').exists() for c in ('direct_task','limit_gated'))
    rng=np.random.default_rng(171);checks=[]
    for _ in range(20):
        j=rng.normal(size=(6,6));e=rng.normal(size=6)*.1;b=rng.normal(size=6)*.001;h=j.T@j+.015**2*np.eye(6);target=j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*e)+b
        u=rng.normal(size=6)*.02;v=rng.normal(size=6)*.02
        task=lambda x:.5*np.linalg.norm(j@(x-b)-.35*e)**2+.5*.015**2*np.linalg.norm(x-b)**2
        centered=lambda x:.5*(x-target)@h@(x-target)
        np.testing.assert_allclose(task(u)-task(v),centered(u)-centered(v),atol=1e-12,rtol=0);checks.append(True)
    # Saturated first joint: direct task objective reallocates motion to second.
    j=np.zeros((6,6));j[0,:2]=[1,1];e=np.array([.2,0,0,0,0,0]);b=np.zeros(6);h=j.T@j+.015**2*np.eye(6);target=j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),e)+b;q=np.zeros(6);q[0]=2.5
    g=np.zeros((1,6));d=np.ones(1);direct,info,_=prior.project(target,b,q,d,g,h);clipped,_,_=prior.project(np.clip(target,-.04,.04),b,q,d,g,h)
    assert info['solver_success'];assert np.linalg.norm(j@direct-e)<=np.linalg.norm(j@clipped-e)+1e-10
    save(ROOT/'preflight.json',dict(before_interventions=True,quadratic_equivalence_checks=len(checks),saturated_joint_example_direct_error=float(np.linalg.norm(j@direct-e)),saturated_joint_example_clipped_error=float(np.linalg.norm(j@clipped-e))))
    print('Direct-task objective checks passed',flush=True)

if __name__=='__main__':run()
