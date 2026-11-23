import numpy as np
from examples.ease_fm.history_excitation import excitation


def test_ridge_is_not_counted_as_observed_excitation():
    r=excitation(1e-6*np.eye(7))
    assert r['energy']==0 and r['effective_rank']==0 and r['above_ridge']==0


def test_rank_one_motion_and_isotropic_motion_are_distinguished():
    q=np.array([1.,2.,3.,4.,5.,6.])*.01
    one=excitation(1e-6*np.eye(6)+np.outer(q,q))
    full=excitation(1e-6*np.eye(6)+.001*np.eye(6))
    assert abs(one['effective_rank']-1)<1e-12 and one['above_ridge']==1
    assert abs(full['effective_rank']-6)<1e-12 and full['above_ridge']==6
