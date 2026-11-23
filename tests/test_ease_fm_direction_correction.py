from types import SimpleNamespace
import numpy as np
from examples.ease_fm.direction_correction import DirectionArm


class IdentityResponse:
    def predict(self,command,velocity):
        return command.copy(),0


def arm():
    env=object.__new__(DirectionArm)
    env.n=1
    env.data=SimpleNamespace(qpos=np.zeros(3),qvel=np.zeros(3))
    env.clearance=lambda q,fingers: min(.002,float(q[0]))
    return env


def test_direction_improves_predicted_clearance_without_mutating_inputs():
    env=arm();action=np.array([-.01]);q=env.data.qpos.copy()
    chosen,log=env.corrected_action(action,0.,np.zeros((6,1)),np.zeros(6),IdentityResponse())
    assert chosen[0]==0 and log['chosen'][0]>log['base'][0]
    assert np.array_equal(action,[-.01]) and np.array_equal(env.data.qpos,q)
    assert np.max(np.abs(chosen-action))<=.01


def test_task_progress_constraint_rejects_clearance_only_improvement():
    env=arm();jac=np.zeros((6,1));jac[0,0]=1;error=np.zeros(6);error[0]=-.01
    chosen,log=env.corrected_action(np.array([-.01]),0.,jac,error,IdentityResponse())
    assert chosen[0]==-.01 and not log['changed']
