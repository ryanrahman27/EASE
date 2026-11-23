from examples.ease_fm.grasp_diagnostics import summarize


def sample(phase,bilateral=True,relative=0):
    return dict(phase=phase,bilateral=bilateral,relative_object=[relative,0,0],bad=[],position_error=.001,rotation_error=.01,prediction_error=.0001)


def test_acquisition_loss_and_drift_timing():
    trace=[sample('close') for _ in range(8)]+[sample('lift',False,.006) for _ in range(3)]
    r=summarize(trace)
    assert r['stable_close'] and r['first_sustained_contact_loss']==8
    assert r['first_relative_drift_5mm']==8 and r['first_collision'] is None


def test_brief_contact_loss_and_release_excluded():
    trace=[sample('close') for _ in range(7)]+[sample('lift',False),sample('lift'),sample('release',False)]*3
    r=summarize(trace)
    assert not r['stable_close'] and r['first_sustained_contact_loss'] is None
