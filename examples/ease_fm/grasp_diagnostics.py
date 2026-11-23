"""E69: read-only grasp telemetry on exact E68 baseline replays."""
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from .arm_pick_place import PickPlace, rollout
from .mixed_sim import Arm, AXES
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .real_arm_reach import load_models
from .benchmark import fingerprint

ROOT=Path('outputs/ease_grasp_diagnostics_v1')
REFERENCE=Path('outputs/ease_grasp_settling_v1')


class InstrumentedArm(PickPlace):
    def step(self,action,aperture):
        result=super().step(action,aperture)
        if hasattr(self,'telemetry'):
            obs=self.observe();force={name:0. for name in ('gripper_left_pad','gripper_right_pad')}
            for i,c in enumerate(self.data.contact[:self.data.ncon]):
                pair={int(c.geom1),int(c.geom2)}
                if self.gids['cube'] not in pair:continue
                wrench=np.zeros(6);mujoco.mj_contactForce(self.model,self.data,i,wrench)
                for name in force:
                    if self.gids[name] in pair:force[name]+=float(wrench[0])
            self.telemetry.append(dict(relative_object=(obs['rotation'].T@(self.data.xpos[self.obj]-obs['position'])).tolist(),
                bilateral=len(self.contacts()[0])==2,normal_force=force,
                finger_q=self.data.qpos[self.n:self.n+2].tolist()))
        return result


class InstrumentedController(GuardedPose):
    def predict(self):
        result=super().predict();self.last_jac=result[0].copy();return result
    def update(self,before,after,command):
        dq=after['q']-before['q']
        actual=np.r_[after['position']-before['position'],.15*(after['rotation'][:,2]-before['rotation'][:,2])]
        self.errors.append(float(np.linalg.norm((actual-self.last_jac@dq)[:3])))
        super().update(before,after,command)


def summarize(trace):
    """Descriptive timing only: acquisition, sustained contact loss, relative drift."""
    close=[r for r in trace if r['phase']=='close']; streak=0;longest=0
    for r in close:
        streak=streak+1 if r['bilateral'] else 0;longest=max(longest,streak)
    carry=[(i,r) for i,r in enumerate(trace) if r['phase'] in ('lift','transport','lower')]
    anchor=np.array(close[-1]['relative_object']) if close else None
    missing=0;loss=None;drift=None
    for i,r in carry:
        missing=0 if r['bilateral'] else missing+1
        if missing==3 and loss is None:loss=i-2
        if anchor is not None and drift is None and np.linalg.norm(np.array(r['relative_object'])-anchor)>.005:drift=i
    collision=next((i for i,r in enumerate(trace) if r['bad']),None)
    window=trace[max(0,loss-5):loss+1] if loss is not None else []
    return dict(close_bilateral_streak=longest,stable_close=longest>=8,
        first_sustained_contact_loss=loss,first_relative_drift_5mm=drift,first_collision=collision,
        loss_phase=trace[loss]['phase'] if loss is not None else None,
        preloss_position_error_max=max((r['position_error'] for r in window),default=None),
        preloss_direction_error_max=max((r['rotation_error'] for r in window),default=None),
        preloss_prediction_error_mean=float(np.mean([r['prediction_error'] for r in window])) if window else None,
        max_carry_relative_drift=float(max((np.linalg.norm(np.array(r['relative_object'])-anchor) for _,r in carry),default=0)) if anchor is not None else None)


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    protocol=json.loads((REFERENCE/'protocol.json').read_text());models={'analytical':None};hashes={};tasks=[]
    for group,folder in [('parent','outputs/ease_robust_training_v1'),('candidate','outputs/ease_estimator_distillation_v1/analytical')]:
        mm,hh=load_models(folder);hashes.update(hh);models.update({f'{group}_{s}':mm[f'neural_{s}'] for s in range(3)})
    assert hashes==protocol['checkpoints']
    for split,folder in [('familiar','outputs/ease_arm_pick_place_v2'),('unseen','outputs/ease_arm_pick_place_unseen_v1')]:
        p=Path(folder);AXES.update(json.loads((p/'axis_catalog.json').read_text()))
        tasks.extend(dict(split=split,local_case=i,**t) for i,t in enumerate(json.loads((p/'cases.json').read_text())))
    spec=dict(experiment='E69',checkpoints=hashes,inputs=protocol['inputs'],test_evaluated=False,
        design='All112 E68 baseline trials, unchanged actions/physics/sequencing. Read-only endpoint contact forces, finger positions, tool-frame object translation and one-step measured-motion prediction residual. Require every original trace to replay exactly.',
        criteria='Stable close: at least8 consecutive bilateral endpoint contacts during close. Carry loss: first3 consecutive missing bilateral endpoints in lift/transport/lower, timestamp first missing. Relative drift: first distance above5mm from final-close object/tool translation; report continuously as well. Preloss window includes loss sample and preceding5 control steps. Thresholds fixed before new telemetry.',
        limits='Descriptive associations, not causal attribution. Relative translation includes commanded rotation effects and can reflect rolling/contact compliance. Endpoint sensing misses substep events. Motion residual is observed under contact and is not a pure estimator-error measure. Tracking tolerances at phase transitions are not failure proof. No recovery action, training, native tuning or final-test evaluation.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ['grasp_diagnostics.py','arm_pick_place.py','pick_place_control.py','retention_fresh.py']:
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    rows=[]
    for case,t in enumerate(tasks):
        for method,model in models.items():
            env=InstrumentedArm(Arm(**t['arm']),t['xy'],t['target']);initial=GuardedPose(env.n,None)
            probe=calibrate(env,t['home'],initial,38000+t['local_case'])
            assert probe==json.loads((REFERENCE/f'probe_{case}.json').read_text())
            controller=InstrumentedController(env.n,model);controller.errors=[]
            for attr in ('gram','cross','bias'):setattr(controller,attr,getattr(initial,attr).copy())
            env.telemetry=[]
            with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
                result=rollout(env,t['home'],controller,reset=False)
            old=json.loads((REFERENCE/'trials'/f'{case}_{method}_baseline.json').read_text())
            assert result['trace']==old['trace'] and result['success']==old['success']
            assert len(result['trace'])==len(env.telemetry)==len(controller.errors)
            trace=[dict(**r,**extra,prediction_error=error) for r,extra,error in zip(result['trace'],env.telemetry,controller.errors)]
            row=dict(case=case,method=method,dof=env.n,split=t['split'],success=result['success'],collision=result['collision'],grasped=result['grasped'],lifted=result['lifted'],**summarize(trace))
            (ROOT/'trials'/f'{case}_{method}.json').write_text(json.dumps(dict(**row,trace=trace)))
            rows.append(row)
        print('replayed case',case,'of16',flush=True)
    for path,sha in {**hashes,**protocol['inputs']}.items():assert fingerprint(path)==sha
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'audit.json').write_text(json.dumps(dict(exact_replays=len(rows),hashes_unchanged=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete',trials=len(rows))))


if __name__=='__main__':run()
