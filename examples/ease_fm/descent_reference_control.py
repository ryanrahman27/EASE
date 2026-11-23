"""E166: conditional high approach with a bounded descent position reference."""
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import conditional_approach_control as prior
from . import history_estimator_control as old
from .arm_pick_place import PickPlace,ROT_SCALE
from .abstaining_task_value_gate import read,save,fingerprint
BASE=prior.BASE
ROOT=Path('outputs/ease_descent_reference_control_v1')

def prepare():
    ROOT.mkdir(exist_ok=False)
    for f in ('trials','states','actions','histories','source','decisions','control'):(ROOT/f).mkdir()
    for f in ('cases.json','folds.json','reference_choices.json'):shutil.copy2(BASE/f,ROOT/f)
    save(ROOT/'protocol.json',dict(experiment='E166',scope='All48 six-joint fitting placements,4 exact seven-joint no-op replays,44 seven-joint baseline results reused. Adaptive fitting study after E164 completed and E165 began; no held outcomes.',
      hypothesis='E164 higher approach removes some elbow contacts but creates a gripper/table descent regression. Limit the descent position reference to reduce the initial large Cartesian request while retaining original phase completion at the actual final goal.',
      rule='Retain E165 once-per-episode lateral-distance>8mm high-approach decision. Only for active six-joint episodes during descend, cap norm of position-error vector to30mm (one cube side), leaving tool-axis orientation error unchanged. Compute original clipped DLS base and capped-position DLS base with the same live E120 reference Jacobian and frozen bias. Preserve the original bounded E120 action residual (original complete action minus original DLS base) and add it to the new base, then original +/- .04 clip. No gain/damping change, global rate limit, body lookup, geometry, exact Jacobian, object/contact signal or physics selector.',
      preserved='E161265mm lower/release target, original final descent goal240mm, full-goal phase completion and100-step descent budget, all other phase budgets and actions, calibration, fixed probe and frozen bias. Inactive episodes must exactly match the E161 baseline.',
      screen='More than37/48 six-joint successes, fewer than10 approach/descent first contacts, no more than11 collision episodes, and no lost E161 six-joint success. Report all paired changes against baseline and E165, without automatic held test or checkpoint replacement.',
      inputs={str(BASE/p):fingerprint(BASE/p) for p in ('records.json','cases.json','verification.json')},position_error_cap_m=.03))
    for f in ('descent_reference_control.py','conditional_approach_control.py','history_estimator_control.py','training_estimator_core.py','bounded_history_correction.py','local_probe.py','arm_pick_place.py'):shutil.copy2(Path(__file__).with_name(f),ROOT/'source'/f)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py')]})

def run():
    core=old.broad.previous.core;original_rollout=core.probe_rollout;original_evaluate=old.evaluate;logs=[]
    class DescentArm(PickPlace):
        def step(self,action,aperture):
            c=getattr(self,'descent_controller',None)
            if c is not None:
                before=self.observe();original=np.asarray(action).copy();active=self.descent_active and self.phase=='descend';extra={}
                if active:
                    j,bias=c.predict();error=np.r_[self.goal-before['position'],ROT_SCALE*(np.array([0.,0.,1.])-before['rotation'][:,2])];capped=error.copy();capped[:3]*=min(1.,.03/max(np.linalg.norm(error[:3]),1e-12));solve=lambda e:np.clip(j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*e)+bias,-.04,.04);base=solve(error);newbase=solve(capped);residual=original-base;assert np.max(np.abs(residual))<=.00100000001;action=np.clip(newbase+residual,-.04,.04);extra=dict(j=j.tolist(),bias=bias.tolist(),error=error.tolist(),capped_error=capped.tolist(),original_base=base.tolist(),new_base=newbase.tolist(),residual=residual.tolist())
                logs.append(dict(phase=self.phase,active=active,q_before=before['q'].tolist(),original_action=original.tolist(),final_action=np.asarray(action).tolist(),**extra))
            return super().step(action,aperture)
    def rollout(env,home,c,**kwargs):
        env.descent_controller=c;env.descent_active=bool(env.n==6 and np.linalg.norm(env.observe()['position'][:2]-env.xy)>.008)
        return original_rollout(env,home,c,**kwargs)
    def evaluate(*args,**kwargs):
        logs.clear();r=original_evaluate(*args,**kwargs);stem=f"{r['case']}_e120_ensemble_frozen_bias";save(ROOT/'control'/f'{stem}.json',logs);actions=read(ROOT/'actions'/f'{stem}.json');assert len(logs)==len(actions)
        for l,a in zip(logs,actions):
            np.testing.assert_array_equal(l['q_before'],a['q_before'])
            if not l['active']:np.testing.assert_array_equal(l['original_action'],l['final_action'])
            else:
                assert l['phase']=='descend';assert np.linalg.norm(l['capped_error'][:3])<=.030000000001;np.testing.assert_array_equal(l['capped_error'][3:],l['error'][3:]);j=np.array(l['j'],dtype=np.float32);solve=lambda e:np.clip(j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*np.array(e))+l['bias'],-.04,.04);np.testing.assert_array_equal(solve(l['error']),l['original_base']);np.testing.assert_array_equal(solve(l['capped_error']),l['new_base']);np.testing.assert_array_equal(np.clip(np.array(l['new_base'])+l['residual'],-.04,.04),l['final_action'])
            if l['phase']!='probe':np.testing.assert_array_equal(np.clip(np.array(l['q_before'])+np.clip(l['final_action'],-.04,.04),-2.5,2.5)-l['q_before'],a['executed_command'])
        return r
    with patch.object(prior,'ROOT',ROOT),patch.object(core,'PickPlace',DescentArm),patch.object(core,'probe_rollout',rollout),patch.object(old,'evaluate',evaluate):prior.run()
    v=read(ROOT/'verification.json');v.update(descent_error_and_DLS_reproduced=True,original_E120_residual_preserved=True,actual_commands_verified=True,all_other_actions_unmodified=True);save(ROOT/'verification.json',v)

if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()
