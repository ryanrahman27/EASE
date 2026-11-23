from examples.ease_fm.phase_retention import transition_ok


def test_final_consecutive_pose_gate():
    good=dict(position_error=.001,rotation_error=.01,bilateral=True)
    assert transition_ok('lift',[good]*5)
    assert not transition_ok('lift',[good]*4)
    assert not transition_ok('lift',[good]*4+[dict(good,position_error=.008)])
    assert transition_ok('lift',[dict(good,position_error=.1)]+[good]*5)


def test_close_requires_pose_and_final_contact():
    good=dict(position_error=.001,rotation_error=.01,bilateral=True)
    assert transition_ok('close',[good]*8)
    assert not transition_ok('close',[good]*7)
    assert not transition_ok('close',[good]*8+[dict(good,bilateral=False)])
    assert not transition_ok('close',[dict(good,rotation_error=.09)]*8)
    assert transition_ok('release',[])
