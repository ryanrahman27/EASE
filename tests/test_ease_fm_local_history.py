import numpy as np
from examples.ease_fm.local_history import LocalPose
from examples.ease_fm.pick_place_control import PoseController


def test_recent_statistics_match_last32_replay_without_resetting_bias():
    c=LocalPose(2,None);full=PoseController(2);short=PoseController(2)
    for i in range(40):
        before=dict(q=np.array([i*.001,0.]),position=np.array([i*.002,0.,0.]),rotation=np.eye(3))
        after=dict(q=np.array([(i+1)*.001,.0001*(i+1)]),position=np.array([(i+1)*.002,.0002*i,0.]),rotation=np.eye(3))
        command=np.array([.003,.002])
        c.update(before,after,command);full.update(before,after,command)
        if i>=8:short.update(before,after,command)
    np.testing.assert_array_equal(c.gram,full.gram)
    np.testing.assert_array_equal(c.cross,full.cross)
    c.local=True;c.rebuild()
    np.testing.assert_array_equal(c.gram,short.gram)
    np.testing.assert_array_equal(c.cross,short.cross)
    np.testing.assert_array_equal(c.bias,full.bias)
    assert len(c.history)==32


def test_recent_history_copies_measurements():
    c=LocalPose(1,None)
    before=dict(q=np.zeros(1),position=np.zeros(3),rotation=np.eye(3))
    after=dict(q=np.ones(1)*.01,position=np.ones(3)*.02,rotation=np.eye(3))
    c.update(before,after,np.ones(1)*.03)
    after['q'][:]=10;after['position'][:]=20
    dq,dx=c.history[0]
    np.testing.assert_array_equal(dq,[.01]);np.testing.assert_array_equal(dx[:3],[.02]*3)
