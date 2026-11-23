"""E75: causal measured-response prediction audit; no action intervention."""
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

ROOT=Path('outputs/ease_response_prediction_v1')
REFERENCE=Path('outputs/ease_geometry_transfer_v1')


class ResponseModel:
    """Independent per-actuator online ridge regression, causal completed steps."""
    def __init__(self,n,velocity):
        self.velocity=velocity;d=3 if velocity else 2
        self.gram=np.tile(.1*np.eye(d),(n,1,1));self.cross=np.zeros((n,d));self.cross[:,0]=.1
        self.updates=0
    def features(self,command,velocity):
        columns=[command/.04]
        if self.velocity:columns.append(velocity)  # dt*v/.04 with dt=.04.
        return np.stack(columns+[np.ones_like(command)],axis=-1)
    def predict(self,command,velocity):
        x=self.features(command,velocity);coef=np.linalg.solve(self.gram,self.cross[...,None])[...,0]
        raw=.04*np.sum(coef*x,axis=-1)
        return np.clip(raw,-.08,.08),int(np.sum(np.abs(raw)>.08))
    def update(self,command,velocity,displacement):
        x=self.features(command,velocity)
        self.gram=.98*self.gram+np.einsum('ni,nj->nij',x,x)+1e-6*np.eye(x.shape[1])[None]
        self.cross=.98*self.cross+x*(displacement/.04)[:,None];self.updates+=1


class ResponseArm(GeometryArm):
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
        evaluate=phase in ('descend','lower') and not self.bad
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
    prior=json.loads((REFERENCE/'protocol.json').read_text());tasks=json.loads(Path('outputs/ease_phase_fresh_v1/cases.json').read_text());models={'analytical':None};hashes={}
    for p in ('outputs/ease_arm_pick_place_v2/axis_catalog.json','outputs/ease_arm_pick_place_unseen_v1/axis_catalog.json'):AXES.update(json.loads(Path(p).read_text()))
    for group,folder in [('parent','outputs/ease_robust_training_v1'),('candidate','outputs/ease_estimator_distillation_v1/analytical')]:
        mm,hh=load_models(folder);hashes.update(hh);models.update({f'{group}_{s}':mm[f'neural_{s}'] for s in range(3)})
    assert hashes==prior['checkpoints']
    inputs={**prior['inputs'],str(REFERENCE/'records.json'):fingerprint(REFERENCE/'records.json')}
    spec=dict(experiment='E75',inputs=inputs,checkpoints=hashes,test_evaluated=False,
        design='All119 E73 geometry trajectories replay unchanged, all successes and failures. Joint response models update on completed measured steps including32 calibration steps. Prediction evaluated before each next step. Neural policy weights frozen; separate analytical online predictors, not a newly trained Transformer.',
        models='Per-actuator ridge: command/.04 and intercept, versus command/.04, current joint velocity (equivalent .04*v/.04), intercept. Gram0.1I, prior command coefficient1, forgetting.98, ridge floor1e-6, output displacement clipped+/-0.08rad/metres. Predict arm and finger joints. Same fixed settings for all bodies/checkpoints; no tuning.',
        paths='Four21-point geometry paths including start: commanded straight path; response endpoint without velocity, straight; velocity-conditioned endpoint, straight; velocity-conditioned endpoint with quadratic path q+t*.04*v+t^2*(predicted_dq-.04*v). No future simulator stepping or future observations used for prediction. Geometry remains privileged known robot/scene state.',
        scoring='All descend/lower steps before first forbidden contact, including the first contact step. These phases were selected from E74 evidence, not an independent evaluation. Report collision-step recall and noncollision false positives at clearance<0 and clearance<2mm-1e-6, plus causal endpoint errors. Include successful trajectories. Other phases replay normally; no action changes.',
        decision='Consider an action-changing trial only if velocity-aware prediction detects more first contacts than command-only prediction without increasing false-positive count at the same threshold. Otherwise preserve baseline and record the limitation. Prediction accuracy alone is not task success; observed development data and state-dependent samples are not independent trials.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('response_prediction.py','geometry_filter.py','arm_pick_place.py','pick_place_control.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    rows=[]
    for case,t in enumerate(tasks):
        for method,model in models.items():
            old_path=REFERENCE/'trials'/f'{case}_{method}_geometry.json';old=json.loads(old_path.read_text());inputs[str(old_path)]=fingerprint(old_path)
            env=ResponseArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();env.initialize_response();c=GuardedPose(env.n,model)
            probe=calibrate(env,t['home'],c,71100+t['candidate'])
            assert probe==json.loads((Path('outputs/ease_phase_fresh_v1')/f'probe_{case}.json').read_text())
            env.controller=c;env.filter_enabled=True;env.filter_log=[];env.audit_enabled=True;env.phases=[r['phase'] for r in old['trace']]
            with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Task Jacobian forbidden')):
                replay=rollout(env,t['home'],c,reset=False)
            assert replay['trace']==old['trace'] and env.filter_log==old['filter_log'] and replay['success']==old['success']
            assert len(env.prediction_log)==len(old['trace'])
            assert all(r['prior_updates']==32+i for i,r in enumerate(env.prediction_log))
            row=dict(case=case,method=method,dof=env.n,success=replay['success'],records=env.prediction_log)
            (ROOT/'trials'/f'{case}_{method}.json').write_text(json.dumps(row));rows.append(dict(case=case,method=method,dof=env.n,success=replay['success'],steps=len(env.prediction_log)))
        print('replayed case',case,'of17',flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'audit.json').write_text(json.dumps(dict(exact_replays=len(rows),input_hashes=inputs,hashes_unchanged=True,causal_update_counts_verified=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()
