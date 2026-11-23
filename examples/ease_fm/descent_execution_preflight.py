"""Pre-intervention arithmetic checks and reference descent-error diagnosis."""
import numpy as np
from .descent_execution import ROOT,REFERENCE,prior
from .abstaining_task_value_gate import read,save

def run():
    assert not any((ROOT/c/'records.json').exists() for c in ('fixed_allowance','history_allowance'))
    values=[];bycase=[]
    for row in read(REFERENCE/'records.json'):
        stem=f"{row['case']}_e120_ensemble_frozen_bias";logs=read(REFERENCE/'clearance'/f'{stem}.json');errors=[]
        for l in logs:
            if l['phase']!='descend' or l['gradient'] is None:continue
            prediction=np.array(l['distance_before'])+np.array(l['gradient'])@(np.array(l['final_action'])-l['bias'])
            e=float(np.max(prediction[6:]-l['distance_after'][6:]));errors.append(e);values.append(e)
        bycase.append(dict(case=row['case'],success=row['success'],collision=row['collision'],max_positive_descent_error_m=max([0.,*errors])))
    d=np.full(9,.01);g=np.zeros((9,6));g[6:,0]=1.;bias=np.zeros(6);a=np.zeros(6);allowance=np.r_[np.zeros(6),np.full(3,.003)]
    u,info,rhs=prior.project(a,bias,np.zeros(6),d-allowance/prior.ALPHA,g,np.eye(6))
    np.testing.assert_allclose(rhs,g@bias-prior.ALPHA*(d-prior.MARGIN)+allowance,atol=1e-16)
    assert np.min(g@u-rhs)>=-1e-7
    save(ROOT/'preflight.json',dict(before_intervention=True,allowance_sign_and_units_verified=True,reference_descent_error_quantiles_m=np.quantile(values,[0,.5,.9,.99,1]).tolist(),cases=bycase))
    print('Preflight passed; reference error quantiles',np.quantile(values,[0,.5,.9,.99,1]),flush=True)

if __name__=='__main__':run()
