"""E80: bounded-action comparison of successful-run warnings and missed contact."""
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
from .action_objective_audit import cloned_state
from .mixed_sim import AXES,Arm
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .real_arm_reach import load_models
from .benchmark import fingerprint

ROOT=Path('outputs/ease_action_retention_audit_v1')
SOURCE=Path('outputs/ease_response_allphase_v1')


class ActionAuditArm(GeometryArm):
    def branch(self,action,aperture,goal):
        before=self.data.qpos.copy();velocity=self.data.qvel.copy();time=self.data.time
        with cloned_state(self):
            obs,command,info=PickPlace.step(self,action,aperture)
            pe=float(np.linalg.norm(goal-obs['position']));re=float(np.linalg.norm(rotation_error(np.eye(3),obs['rotation'])))
            result=dict(collision=bool(info['bad']),bad=info['bad'],position=obs['position'].tolist(),position_error=pe,direction_error=re,
                task_error=float(np.sqrt(pe**2+(ROT_SCALE*re)**2)),pads=info['pads'],object=self.data.xpos[self.obj].tolist())
        assert np.array_equal(self.data.qpos,before) and np.array_equal(self.data.qvel,velocity) and self.data.time==time
        return result

    def step(self,action,aperture):
        index=self.steps-32
        if getattr(self,'audit_enabled',False) and index in self.selected_steps:
            phase=self.phases[index];goal=next(g for name,g,_,_ in targets(self.xy,self.target) if name==phase)
            obs=self.observe();selected,check=self.select_action(action,aperture,self.controller.bias)
            bias=np.clip(self.controller.bias,-.04,.04)
            jac=np.linalg.solve(self.controller.gram,self.controller.cross).T
            error=np.r_[goal-obs['position'],ROT_SCALE*rotation_error(np.eye(3),obs['rotation'])]
            analytical=np.clip(jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(6),.35*error)+self.controller.bias,-.04,.04)
            actions=dict(selected=selected,half=np.clip(bias+.5*(selected-bias),-.04,.04),quarter=np.clip(bias+.25*(selected-bias),-.04,.04),bias_only=bias,analytical_same_history=analytical)
            branches={name:self.branch(command,aperture,goal) for name,command in actions.items()}
            self.attribution.append(dict(step=index,phase=phase,before_task_error=float(np.linalg.norm(error)),check=check,branches=branches))
        return super().step(action,aperture)


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    inputs={}
    def read(p):
        inputs[str(p)]=fingerprint(p);return json.loads(p.read_text())
    prior=read(SOURCE/'protocol.json');runs=read(SOURCE/'records.json');selected=[];tasks=[]
    for run in runs:
        if run['method']=='analytical':continue
        full=read(SOURCE/'trials'/f"{run['case']}_{run['method']}.json")
        steps=[r['step'] for r in full['records'] if r['evaluated'] and ((run['success'] and r['clearances']['response_linear']<.002-1e-6) or r['collision'])]
        if steps:selected.append(dict(**run,selected_steps=steps))
    for folder in ('ease_arm_pick_place_v2','ease_arm_pick_place_unseen_v1'):
        p=Path('outputs')/folder;AXES.update(read(p/'axis_catalog.json'));tasks.extend(dict(local_case=i,**t) for i,t in enumerate(read(p/'cases.json')))
    models={};hashes={}
    for group,folder in [('parent','outputs/ease_robust_training_v1'),('candidate','outputs/ease_estimator_distillation_v1/analytical')]:
        mm,hh=load_models(folder);hashes.update(hh);models.update({f'{group}_{s}':mm[f'neural_{s}'] for s in range(3)})
    assert hashes==prior['checkpoints']
    spec=dict(experiment='E80',selection=selected,checkpoints=hashes,
        design='All E78 successful learned trajectories containing response warnings plus every learned first contact. Selected after outcomes for causal-geometry diagnosis, not representative performance estimation.',
        measurement='Five one-step cloned branches: selected action; half and quarter task component with clipped measured bias retained; bias-only; analytical Jacobian fitted from identical measured history. Same aperture and unchanged incoming velocity. Every action clipped +/- .04. Branches bypass geometry filtering and do not update controllers. Report collisions, pads, object pose and task error.',
        limits='Simulator clones provide privileged diagnostic counterfactuals, not an inference-time selection oracle. Outcome-selected controls from two successful trajectories are dependent. One-step avoidance can postpone contact and is not task recovery. E78 gate remains failed.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('action_retention_audit.py','response_allphase.py','response_prediction.py','geometry_filter.py','arm_pick_place.py'):
        p=Path(__file__).with_name(name);assert not (SOURCE/'source'/name).exists() or fingerprint(p)==fingerprint(SOURCE/'source'/name)
        shutil.copy2(p,ROOT/'source'/name)
    results=[]
    for run in selected:
        case=run['case'];method=run['method'];t=tasks[case]
        old=read(Path('outputs/ease_geometry_filter_v2/trials')/f'{case}_{method}_geometry.json')
        previous=read(SOURCE/'trials'/f'{case}_{method}.json')
        env=ActionAuditArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();c=GuardedPose(env.n,models[method])
        probe=calibrate(env,t['home'],c,38000+t['local_case'])
        assert probe==read(Path('outputs/ease_grasp_settling_v1')/f'probe_{case}.json')
        env.controller=c;env.filter_enabled=True;env.filter_log=[];env.audit_enabled=True;env.phases=[r['phase'] for r in old['trace']];env.selected_steps=set(run['selected_steps']);env.attribution=[]
        with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Task Jacobian forbidden')):
            replay=rollout(env,t['home'],c,reset=False)
        assert replay['trace']==old['trace'] and replay['success']==old['success'] and env.filter_log==old['filter_log']
        for event in env.attribution:
            reference=old['trace'][event['step']]
            assert event['branches']['selected']['position']==reference['position']
            assert event['branches']['selected']['collision']==bool(reference['bad'])
            assert event['check']==old['filter_log'][event['step']]
        assert len(env.attribution)==len(run['selected_steps'])
        result=dict(**run,attribution=env.attribution)
        (ROOT/'trials'/f'{case}_{method}.json').write_text(json.dumps(result));results.append(run)
        print('exact replay',case,method,'diagnosed steps',len(env.attribution),flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(results,indent=2))
    (ROOT/'audit.json').write_text(json.dumps(dict(inputs=inputs,hashes_unchanged=True,exact_replays=len(results),selected_branches_exact=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()
