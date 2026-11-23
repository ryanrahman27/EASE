"""E81: bounded-action comparison of successful-run warnings and missed contact."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from .response_allphase import AllPhaseResponseArm
from .geometry_filter import allowed_pair,MARGIN
from .arm_pick_place import PickPlace,rollout,targets,ROT_SCALE,rotation_error
from .geometry_filter import GeometryArm
from .action_objective_audit import cloned_state,copy_controller
from .mixed_sim import AXES,Arm
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .real_arm_reach import load_models
from .benchmark import fingerprint

ROOT=Path('outputs/ease_early_warning_v1')
SOURCE=Path('outputs/ease_response_allphase_v1')


class EarlyWarningArm(GeometryArm):
    def branch(self,mode,aperture,goal):
        before=self.data.qpos.copy();velocity=self.data.qvel.copy();time=self.data.time
        c=copy_controller(self.controller,self.controller.model)
        history=[]
        with cloned_state(self):
            obs=self.observe()
            for j in range(20):
                jac,bias=c.predict()
                error=np.r_[goal-obs['position'],ROT_SCALE*rotation_error(np.eye(3),obs['rotation'])]
                raw=np.clip(jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(6),.35*error)+bias,-.04,.04)
                action,check=self.select_action(raw,aperture,c.bias)
                if j<5 and mode!='selected':
                    scale={'half':.5,'quarter':.25,'bias_only':0.}[mode];bb=np.clip(c.bias,-.04,.04)
                    action=np.clip(bb+scale*(action-bb),-.04,.04)
                previous=obs;obs,command,info=PickPlace.step(self,action,aperture);c.update(previous,obs,command)
                pe=float(np.linalg.norm(goal-obs['position']));re=float(np.linalg.norm(rotation_error(np.eye(3),obs['rotation'])))
                history.append(dict(step=j+1,collision=bool(info['bad']),position=obs['position'].tolist(),task_error=float(np.sqrt(pe**2+(ROT_SCALE*re)**2)),pads=info['pads']))
        assert np.array_equal(self.data.qpos,before) and np.array_equal(self.data.qvel,velocity) and self.data.time==time
        return dict(first_contact_step=next((r['step'] for r in history if r['collision']),None),history=history)

    def step(self,action,aperture):
        index=self.steps-32
        if getattr(self,'audit_enabled',False) and index in self.selected_steps:
            phase=self.phases[index];goal=next(g for name,g,_,_ in targets(self.xy,self.target) if name==phase)
            branches={mode:self.branch(mode,aperture,goal) for mode in ('selected','half','quarter','bias_only')}
            self.attribution.append(dict(step=index,phase=phase,branches=branches))
        return super().step(action,aperture)


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
    spec=dict(experiment='E81',selection=selected,coverage=coverage,checkpoints=hashes,
        design='First fixed no-velocity 2mm warning in every learned E78 trajectory, irrespective of outcome; include collision-step-only warnings, and report collisions without advance warning. Historical predictions were causal and replay-verified in E78. No retrospective fixed lookback from collision.',
        measurement='Four cloned 20-step (0.8 s) branches. Selected controller baseline; half, quarter or bias-only task motion for first five steps (0.2 s), then same baseline for fifteen. Recompute controller action, measured bias and geometry filter each step and update cloned controller from branch measurements. Freeze triggering phase goal and aperture for all 20 steps; no future reference phase information. Same incoming velocity; no analytical switch.',
        limits='Phase-held local continuations, not full-task rollouts. Compare against matched phase-held baseline, not original later trajectory. Five-step pulse and 20-step horizon fixed before results. Repeated historical development data and only two successful warning trajectories; no generalization or task-recovery claim.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('early_warning_audit.py','response_allphase.py','response_prediction.py','geometry_filter.py','arm_pick_place.py'):
        p=Path(__file__).with_name(name);assert not (SOURCE/'source'/name).exists() or fingerprint(p)==fingerprint(SOURCE/'source'/name)
        shutil.copy2(p,ROOT/'source'/name)
    results=[]
    for run in selected:
        case=run['case'];method=run['method'];t=tasks[case]
        old=read(Path('outputs/ease_geometry_filter_v2/trials')/f'{case}_{method}_geometry.json')
        previous=read(SOURCE/'trials'/f'{case}_{method}.json')
        env=EarlyWarningArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();c=GuardedPose(env.n,models[method])
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

        assert len(env.attribution)==len(run['selected_steps'])
        result=dict(**run,attribution=env.attribution)
        (ROOT/'trials'/f'{case}_{method}.json').write_text(json.dumps(result));results.append(run)
        print('exact replay',case,method,'diagnosed steps',len(env.attribution),flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(results,indent=2))
    (ROOT/'audit.json').write_text(json.dumps(dict(inputs=inputs,hashes_unchanged=True,exact_replays=len(results),selected_branches_exact=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()
