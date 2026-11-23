from types import SimpleNamespace
import numpy as np
import pytest
from examples.ease_fm.local_probe import probe_offset,probe_rollout


@pytest.mark.parametrize('n',[6,7])
def test_probe_has_balanced_fixed_offsets_and_return_steps(n):
    offsets=np.array([probe_offset(i,n) for i in range(32)])
    np.testing.assert_array_equal(offsets.sum(axis=0),np.zeros(n))
    assert np.max(np.abs(offsets))==.004
    assert np.all(np.count_nonzero(offsets,axis=1)<=1)
    assert np.all(np.count_nonzero(offsets,axis=0)==4)
    assert np.all(offsets[4*n:]==0)


@pytest.mark.parametrize('phase',['probe','hold'])
def test_probe_and_hold_cannot_early_exit_at_stationary_goal(phase):
    position=np.array([0.,0.,.215])
    obs=dict(q=np.zeros(1),position=position,rotation=np.eye(3))
    env=SimpleNamespace(steps=0,grasped=False,lifted=False,bad=False,obj=0,oj=0,target=np.zeros(2),
        data=SimpleNamespace(xpos=position[None],qvel=np.zeros(6)),model=SimpleNamespace(jnt_dofadr=np.array([0])))
    env.observe=lambda:obs;env.contacts=lambda:(set(),[])
    def step(action,aperture):
        env.steps+=1
        return obs,np.zeros(1),dict(bad=[],pads=0)
    env.step=step
    controller=SimpleNamespace(predict=lambda:(np.zeros((6,1)),np.zeros(1)),update=lambda *args:None)
    result=probe_rollout(env,np.zeros(1),controller,reset=False,sequence=[(phase,position,0.,32)])
    assert result['steps']==32 and len(result['trace'])==32
