import numpy as np
from examples.ease_fm.response_prediction import ResponseModel


def test_prediction_is_read_only_and_velocity_improves_identification():
    rng=np.random.default_rng(75);plain=ResponseModel(2,False);dynamic=ResponseModel(2,True)
    for _ in range(400):
        command=rng.uniform(-.04,.04,2);velocity=rng.uniform(-1,1,2)
        measured=.3*command+.4*.04*velocity+.002
        plain.update(command,velocity,measured);dynamic.update(command,velocity,measured)
    before=(dynamic.gram.copy(),dynamic.cross.copy(),dynamic.updates);errors=[0.,0.]
    for _ in range(50):
        command=rng.uniform(-.04,.04,2);velocity=rng.uniform(-1,1,2);truth=.3*command+.4*.04*velocity+.002
        for i,model in enumerate((plain,dynamic)):errors[i]+=float(np.sum((model.predict(command,velocity)[0]-truth)**2))
    assert errors[1]<errors[0]*.01
    np.testing.assert_array_equal(dynamic.gram,before[0]);np.testing.assert_array_equal(dynamic.cross,before[1]);assert dynamic.updates==before[2]
