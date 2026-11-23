"""E82: bounded-action comparison of successful-run warnings and missed contact."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from .response_prediction import ResponseModel
import copy
from .geometry_filter import allowed_pair,MARGIN
from .arm_pick_place import PickPlace,rollout,targets,ROT_SCALE,rotation_error
from .geometry_filter import GeometryArm
from .action_objective_audit import cloned_state,copy_controller
from .mixed_sim import AXES,Arm
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .real_arm_reach import load_models
from .benchmark import fingerprint

ROOT=Path('outputs/ease_direction_correction_v1')
SOURCE=Path('outputs/ease_response_allphase_v1')


class DirectionArm(GeometryArm):
    def initialize_response(self):self.response=ResponseModel(self.n+2,False)

    def corrected_action(self,selected,aperture,jac,error,response):
        q=self.data.qpos[:self.n+2].copy();v=self.data.qvel[:self.n+2].copy()
        def score(action):
            command=np.r_[np.clip(q[:self.n]+action,-2.5,2.5),[aperture,aperture]]-q
            delta,_=response.predict(command,v)
            clearance=min(self.clearance((q+t*delta)[:self.n],(q+t*delta)[self.n:]) for t in np.linspace(0,1,21))
            progress=float(np.linalg.norm(error-jac@delta[:self.n]))
            return clearance,progress
        base=score(selected);best=selected.copy();best_score=base;queries=21;admissible=0
        if base[0]>=.002-1e-6:return best,dict(changed=False,queries=queries,base=base,chosen=base,admissible=0)
        for i in range(self.n):
            for direction in (-1.,1.):
                candidate=selected.copy();candidate[i]=np.clip(candidate[i]+direction*.01,-.04,.04)
                value=score(candidate);queries+=21
                if value[1]<=base[1]+.001:
                    admissible+=1
                    if value[0]>best_score[0]+1e-6:
                        best=candidate;best_score=value
        assert np.max(np.abs(best-selected))<=.01000000001 and np.max(np.abs(best))<=.04000000001
        return best,dict(changed=bool(not np.array_equal(best,selected)),queries=queries,base=base,chosen=best_score,admissible=admissible)

    def branch(self,mode,aperture,goal):
        before=self.data.qpos.copy();velocity=self.data.qvel.copy();time=self.data.time
        c=copy_controller(self.controller,self.controller.model);response=copy.deepcopy(self.response)
        history=[]
        with cloned_state(self):
            obs=self.observe()
            for j in range(20):
                jac,bias=c.predict()
                error=np.r_[goal-obs['position'],ROT_SCALE*rotation_error(np.eye(3),obs['rotation'])]
                raw=np.clip(jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(6),.35*error)+bias,-.04,.04)
                action,check=self.select_action(raw,aperture,c.bias);correction=None
                if mode=='direction_all' or (mode=='direction_pulse' and j<5):
                    action,correction=self.corrected_action(action,aperture,jac,error,response)
                q=self.data.qpos[:self.n+2].copy();v=self.data.qvel[:self.n+2].copy()
                previous=obs;obs,command,info=PickPlace.step(self,action,aperture);c.update(previous,obs,command)
                response.update(np.r_[command,aperture-q[self.n:]],v,self.data.qpos[:self.n+2]-q)
                pe=float(np.linalg.norm(goal-obs['position']));re=float(np.linalg.norm(rotation_error(np.eye(3),obs['rotation'])))
                history.append(dict(step=j+1,collision=bool(info['bad']),position=obs['position'].tolist(),task_error=float(np.sqrt(pe**2+(ROT_SCALE*re)**2)),pads=info['pads'],correction=correction))
        assert np.array_equal(self.data.qpos,before) and np.array_equal(self.data.qvel,velocity) and self.data.time==time
        return dict(first_contact_step=next((r['step'] for r in history if r['collision']),None),history=history)

    def step(self,action,aperture):
        index=self.steps-32
        if getattr(self,'audit_enabled',False) and index in self.selected_steps:
            assert self.response.updates==32+index
            phase=self.phases[index];goal=next(g for name,g,_,_ in targets(self.xy,self.target) if name==phase)
            branches={mode:self.branch(mode,aperture,goal) for mode in ('selected','direction_pulse','direction_all')}
            assert self.response.updates==32+index
            self.attribution.append(dict(step=index,phase=phase,branches=branches))
        q=self.data.qpos[:self.n+2].copy();v=self.data.qvel[:self.n+2].copy()
        result=super().step(action,aperture)
        self.response.update(np.r_[result[1],aperture-q[self.n:]],v,self.data.qpos[:self.n+2]-q)
        return result


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    inputs={}
    def read(p):
        inputs[str(p)]=fingerprint(p);return json.loads(p.read_text())
    prior=read(SOURCE/'protocol.json');runs=read(SOURCE/'records.json');selected=[];tasks=[];coverage=[]
    for run in runs:
        if run['method']=='analytical':continue
        full=read(SOURCE/'trials'/f"{run['case']}_{run['method']}.json")
        first_contact=next((r['step'] for r in full['records'] if r['collision']),None)
        warnings=[r['step'] for r in full['records'] if r['evaluated'] and r['clearances']['response_linear']<.002-1e-6]
        first_warning=warnings[0] if warnings else None
        coverage.append(dict(**run,first_contact=first_contact,first_warning=first_warning,advance_warning=first_contact is not None and first_warning is not None and first_warning<first_contact))
        if warnings:selected.append(dict(**run,selected_steps=[first_warning],first_contact=first_contact,warning_lead=None if first_contact is None else first_contact-first_warning))
    for folder in ('ease_arm_pick_place_v2','ease_arm_pick_place_unseen_v1'):
        p=Path('outputs')/folder;AXES.update(read(p/'axis_catalog.json'));tasks.extend(dict(local_case=i,**t) for i,t in enumerate(read(p/'cases.json')))
    models={};hashes={}
    for group,folder in [('parent','outputs/ease_robust_training_v1'),('candidate','outputs/ease_estimator_distillation_v1/analytical')]:
        mm,hh=load_models(folder);hashes.update(hh);models.update({f'{group}_{s}':mm[f'neural_{s}'] for s in range(3)})
    assert hashes==prior['checkpoints']
    spec=dict(experiment='E82',selection=selected,coverage=coverage,checkpoints=hashes,
        design='First fixed no-velocity 2mm warning in every learned E78 trajectory, irrespective of outcome; include collision-step-only warnings, and report collisions without advance warning. Historical predictions were causal and replay-verified in E78. No retrospective fixed lookback from collision.',
        measurement='Three 20-step phase-held branches: unchanged baseline; direction corrections first five steps then baseline; direction corrections all20 steps. Frozen no-velocity E75 ResponseModel fit causally through calibration and branch measurements. At fixed 2mm warning, enumerate baseline plus +/-0.01rad single-joint corrections, clipped +/-0.04. Score21-point predicted displacement paths with whole-arm geometry; object held at current pose. Choose only clearance improvements >1e-6 with predicted task error <= baseline+0.001 (task units), using learned Jacobian for progress. No dynamic lookahead in selection. Fixed coordinate order; no tuning.',
        limits='Known privileged geometry and measured simulator state; learned Jacobian and response errors limit selection. Clearance includes initial point, so no improvement may be available when already close. 27 historical first-warning states, including only two successful trajectories. Local phase-held continuations, not full-task success or independent validation.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('direction_correction.py','response_allphase.py','response_prediction.py','geometry_filter.py','arm_pick_place.py'):
        p=Path(__file__).with_name(name);assert not (SOURCE/'source'/name).exists() or fingerprint(p)==fingerprint(SOURCE/'source'/name)
        shutil.copy2(p,ROOT/'source'/name)
    results=[]
    for run in selected:
        case=run['case'];method=run['method'];t=tasks[case]
        old=read(Path('outputs/ease_geometry_filter_v2/trials')/f'{case}_{method}_geometry.json')
        previous=read(SOURCE/'trials'/f'{case}_{method}.json')
        env=DirectionArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();env.initialize_response();c=GuardedPose(env.n,models[method])
        probe=calibrate(env,t['home'],c,38000+t['local_case'])
        assert probe==read(Path('outputs/ease_grasp_settling_v1')/f'probe_{case}.json')
        env.controller=c;env.filter_enabled=True;env.filter_log=[];env.audit_enabled=True;env.phases=[r['phase'] for r in old['trace']];env.selected_steps=set(run['selected_steps']);env.attribution=[]
        with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Task Jacobian forbidden')):
            replay=rollout(env,t['home'],c,reset=False)
        assert replay['trace']==old['trace'] and replay['success']==old['success'] and env.filter_log==old['filter_log']
        for event in env.attribution:
            reference=old['trace'][event['step']]
            assert event['branches']['selected']['history'][0]['position']==reference['position']
            assert event['branches']['selected']['history'][0]['collision']==bool(reference['bad'])

        matched=read(Path('outputs/ease_early_warning_v1/trials')/f'{case}_{method}.json')['attribution'][0]['branches']['selected']
        branch=env.attribution[0]['branches']['selected']
        assert branch['first_contact_step']==matched['first_contact_step']
        assert [{k:v for k,v in r.items() if k!='correction'} for r in branch['history']]==matched['history']
        assert len(env.attribution)==len(run['selected_steps'])
        result=dict(**run,attribution=env.attribution)
        (ROOT/'trials'/f'{case}_{method}.json').write_text(json.dumps(result));results.append(run)
        print('exact replay',case,method,'diagnosed steps',len(env.attribution),flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(results,indent=2))
    (ROOT/'audit.json').write_text(json.dumps(dict(inputs=inputs,hashes_unchanged=True,exact_replays=len(results),selected_branches_exact=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()
