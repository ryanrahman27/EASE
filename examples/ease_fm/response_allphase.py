"""E78: frozen response predictor validation; no action intervention."""
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from .geometry_filter import GeometryArm
from .arm_pick_place import PickPlace,rollout
from .mixed_sim import Arm,AXES
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .real_arm_reach import load_models
from .benchmark import fingerprint

ROOT=Path('outputs/ease_response_allphase_v1')
REFERENCE=Path('outputs/ease_geometry_filter_v2')


from .response_prediction import ResponseModel

class AllPhaseResponseArm(GeometryArm):
    def initialize_response(self):
        self.response0=ResponseModel(self.n+2,False);self.response1=ResponseModel(self.n+2,True);self.prediction_log=[]
    def step(self,action,aperture):
        selected=action
        if getattr(self,'filter_enabled',False):
            selected,entry=self.select_action(action,aperture,self.controller.bias);self.filter_log.append(entry)
        q=self.data.qpos[:self.n+2].copy();velocity=self.data.qvel[:self.n+2].copy()
        command=np.r_[np.clip(q[:self.n]+np.clip(selected,-.04,.04),-2.5,2.5),[aperture,aperture]]-q
        p0,clip0=self.response0.predict(command,velocity);p1,clip1=self.response1.predict(command,velocity)
        index=self.steps-32;phase=self.phases[index] if getattr(self,'audit_enabled',False) else None
        evaluate=phase is not None and not self.bad
        clearances={}
        if evaluate:
            paths={name:[] for name in ('command','response_linear','velocity_linear','velocity_curve')}
            for t in np.linspace(0,1,21):
                displacements=dict(command=t*command,response_linear=t*p0,velocity_linear=t*p1,velocity_curve=t*.04*velocity+t*t*(p1-.04*velocity))
                for name,delta in displacements.items():
                    point=q+delta
                    paths[name].append(self.clearance(point[:self.n],point[self.n:]))
            clearances={name:float(min(values)) for name,values in paths.items()}
        updates=self.response0.updates
        result=PickPlace.step(self,selected,aperture)
        actual=self.data.qpos[:self.n+2]-q
        if getattr(self,'audit_enabled',False):
            self.prediction_log.append(dict(step=index,phase=phase,evaluated=evaluate,collision=bool(result[2]['bad']),
                prior_updates=updates,clearances=clearances,clipped_predictions=[clip0,clip1],
                arm_endpoint_mse={name:float(np.mean((pred[:self.n]-actual[:self.n])**2)) for name,pred in [('command',command),('response_linear',p0),('velocity_linear',p1)]}))
        self.response0.update(command,velocity,actual);self.response1.update(command,velocity,actual)
        return result




def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    frozen=Path('outputs/ease_response_prediction_v1/source/response_prediction.py')
    assert fingerprint(Path(__file__).with_name('response_prediction.py'))==fingerprint(frozen)
    prior=json.loads((REFERENCE/'protocol.json').read_text());tasks=[];models={'analytical':None};hashes={}
    for split,folder in [('familiar','outputs/ease_arm_pick_place_v2'),('unseen','outputs/ease_arm_pick_place_unseen_v1')]:
        p=Path(folder);AXES.update(json.loads((p/'axis_catalog.json').read_text()))
        tasks.extend(dict(split=split,local_case=i,**t) for i,t in enumerate(json.loads((p/'cases.json').read_text())))
    for group,folder in [('parent','outputs/ease_robust_training_v1'),('candidate','outputs/ease_estimator_distillation_v1/analytical')]:
        mm,hh=load_models(folder);hashes.update(hh);models.update({f'{group}_{s}':mm[f'neural_{s}'] for s in range(3)})
    assert hashes==prior['checkpoints']
    inputs={**prior['inputs'],str(REFERENCE/'records.json'):fingerprint(REFERENCE/'records.json')}
    inputs[str(frozen)]=fingerprint(frozen)
    spec=dict(experiment='E78',inputs=inputs,checkpoints=hashes,test_evaluated=False,
        design='112 unchanged E72 geometry trajectories on 16 other historical development placements; same bodies, not new embodiments or independent final test. Include successes and failures.',
        prediction='Frozen E75 ResponseModel imported; ResponseArm step copied with only phase eligibility expanded to all task phases. Causal calibration and completed-step updates; same ridge, forgetting, 21 path samples and phases.',
        scoring='All task phases up to first collision, inclusive. Report phase-specific detection and successful-run false alarms; exact warning lead times descriptive only. Primary no-velocity response_linear versus command at clearance < .002-1e-6. Velocity variants secondary only.',
        decision='Primary passes only if more learned first contacts detected with no increase in noncollision false-alarm count. No threshold tuning or action changes. A pass only supports a subsequent complete-task intervention experiment.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('response_allphase.py','response_prediction.py','geometry_filter.py','arm_pick_place.py','pick_place_control.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    rows=[]
    for case,t in enumerate(tasks):
        for method,model in models.items():
            old_path=REFERENCE/'trials'/f'{case}_{method}_geometry.json';old=json.loads(old_path.read_text());inputs[str(old_path)]=fingerprint(old_path)
            env=AllPhaseResponseArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();env.initialize_response();c=GuardedPose(env.n,model)
            probe=calibrate(env,t['home'],c,38000+t['local_case'])
            assert probe==json.loads((Path('outputs/ease_grasp_settling_v1')/f'probe_{case}.json').read_text())
            env.controller=c;env.filter_enabled=True;env.filter_log=[];env.audit_enabled=True;env.phases=[r['phase'] for r in old['trace']]
            with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Task Jacobian forbidden')):
                replay=rollout(env,t['home'],c,reset=False)
            assert replay['trace']==old['trace'] and env.filter_log==old['filter_log'] and replay['success']==old['success']
            assert len(env.prediction_log)==len(old['trace'])
            previous=Path('outputs/ease_response_validation_v1/trials')/f'{case}_{method}.json'
            inputs[str(previous)]=fingerprint(previous)
            old_predictions=json.loads(previous.read_text())['records']
            for new,previous_record in zip(env.prediction_log,old_predictions):
                if previous_record['evaluated']:assert new==previous_record
                assert new['arm_endpoint_mse']==previous_record['arm_endpoint_mse']
            first=next((i for i,r in enumerate(env.prediction_log) if r['collision']),len(env.prediction_log)-1)
            assert all(r['evaluated']==(i<=first) for i,r in enumerate(env.prediction_log))
            assert all(r['prior_updates']==32+i for i,r in enumerate(env.prediction_log))
            row=dict(case=case,method=method,dof=env.n,success=replay['success'],records=env.prediction_log)
            (ROOT/'trials'/f'{case}_{method}.json').write_text(json.dumps(row));rows.append(dict(case=case,method=method,dof=env.n,success=replay['success'],steps=len(env.prediction_log)))
        print('replayed case',case,'of16',flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'audit.json').write_text(json.dumps(dict(exact_replays=len(rows),input_hashes=inputs,hashes_unchanged=True,causal_update_counts_verified=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()
