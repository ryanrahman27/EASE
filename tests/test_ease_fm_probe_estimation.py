import numpy as np
from examples.ease_fm.probe_estimation import estimate_metrics


def test_exact_linear_measurements_distinguish_wrong_estimator():
    exact=np.eye(6); dq=.01*np.eye(6); history=list(zip(dq,dq)); error=np.ones(6)*.01
    good=estimate_metrics(exact,exact,history,error)
    wrong=estimate_metrics(-exact,exact,history,error)
    assert good['relative_error']==0 and good['history_relative_residual']==0
    assert wrong['relative_error']==2 and wrong['history_relative_residual']==2
    assert wrong['task_linear_residual']>good['task_linear_residual']


def test_zero_motion_metrics_are_finite_and_do_not_imply_identification():
    exact=np.eye(6); history=[(np.zeros(6),np.zeros(6))]*32
    r=estimate_metrics(np.zeros((6,6)),exact,history,np.ones(6))
    assert r['relative_error']==1 and r['history_relative_residual']==0
    assert np.isfinite(r['task_linear_residual'])
