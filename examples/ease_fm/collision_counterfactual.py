"""E74: exact pre-contact replay and one-step diagnostic counterfactuals."""
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from .geometry_filter import GeometryArm
from .arm_pick_place import PickPlace,rollout,targets,ROT_SCALE,rotation_error
from .mixed_sim import Arm,AXES
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .action_objective_audit import cloned_state
from .real_arm_reach import load_models
from .benchmark import fingerprint

ROOT=Path('outputs/ease_collision_counterfactual_v1')
REFERENCE=Path('outputs/ease_geometry_transfer_v1')


class DiagnosticArm(GeometryArm):
    def branch(self,action,aperture,zero_velocity=False):
        original_step=mujoco.mj_step;samples=[]
        before_q=self.data.qpos.copy();before_v=self.data.qvel.copy();before_time=self.data.time
        def observed_step(model,data,*args):
            original_step(model,data,*args)
            samples.append(dict(q=data.qpos[:self.n].copy(),fingers=data.qpos[self.n:self.n+2].copy()))
        with cloned_state(self):
            if zero_velocity:
                self.data.qvel[:self.n+2]=0
                mujoco.mj_forward(self.model,self.data)
            with patch.object(mujoco,'mj_step',side_effect=observed_step):
                after,command,info=PickPlace.step(self,action,aperture)
            result=dict(collision=bool(info['bad']),bad=sorted({tuple(sorted(x)) for x in info['bad']}),
                        position=after['position'].tolist(),q=after['q'].tolist(),executed=command.tolist(),
                        min_actual_path_clearance=min(self.clearance(x['q'],x['fingers']) for x in samples))
        assert np.array_equal(self.data.qpos,before_q) and np.array_equal(self.data.qvel,before_v) and self.data.time==before_time
        assert len(samples)==20
        return result

    def diagnose(self,action,aperture):
        obs=self.observe();selected,check=self.select_action(action,aperture,self.controller.bias)
        bias=np.clip(self.controller.bias,-.04,.04)
        jac=np.linalg.solve(self.controller.gram,self.controller.cross).T
        error=np.r_[self.event_goal-obs['position'],ROT_SCALE*rotation_error(np.eye(3),obs['rotation'])]
        analytical=np.clip(jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(6),.35*error)+self.controller.bias,-.04,.04)
        commands={'selected':selected,'half_selected':np.clip(bias+.5*(selected-bias),-.04,.04),'bias_only':bias,'analytical_same_history':analytical}
        q=obs['q'];fingers=self.data.qpos[self.n:self.n+2].copy();target=np.clip(q+selected,-2.5,2.5)
        dense=[dict(t=float(t),clearance=self.clearance(q+t*(target-q),fingers+t*(aperture-fingers))) for t in np.linspace(0,1,21)]
        branches={name:self.branch(command,aperture) for name,command in commands.items()}
        branches['selected_zero_incoming_velocity']=self.branch(selected,aperture,True)
        for value in branches.values():value['position_error_after']=float(np.linalg.norm(self.event_goal-np.array(value['position'])))
        return dict(check=check,dense_kinematic=dense,min_dense_clearance=min(x['clearance'] for x in dense),
                    before_position_error=float(np.linalg.norm(error[:3])),before_direction_error=float(np.linalg.norm(rotation_error(np.eye(3),obs['rotation']))),
                    incoming_arm_speed=float(np.linalg.norm(self.data.qvel[:self.n])),incoming_finger_speed=float(np.linalg.norm(self.data.qvel[self.n:self.n+2])),
                    commands={k:v.tolist() for k,v in commands.items()},branches=branches)

    def step(self,action,aperture):
        if getattr(self,'diagnostic_enabled',False) and self.steps-32==self.event_step:
            self.diagnostic=self.diagnose(action,aperture)
        return super().step(action,aperture)


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    all_rows=json.loads((REFERENCE/'records.json').read_text())
    failures=[r for r in all_rows if r['mode']=='geometry' and not r['success']]
    tasks=json.loads(Path('outputs/ease_phase_fresh_v1/cases.json').read_text())
    for p in ('outputs/ease_arm_pick_place_v2/axis_catalog.json','outputs/ease_arm_pick_place_unseen_v1/axis_catalog.json'):AXES.update(json.loads(Path(p).read_text()))
    models={'analytical':None};hashes={}
    for group,folder in [('parent','outputs/ease_robust_training_v1'),('candidate','outputs/ease_estimator_distillation_v1/analytical')]:
        mm,hh=load_models(folder);hashes.update(hh);models.update({f'{group}_{s}':mm[f'neural_{s}'] for s in range(3)})
    inputs={str(REFERENCE/'records.json'):fingerprint(REFERENCE/'records.json'), 'outputs/ease_phase_fresh_v1/cases.json':fingerprint('outputs/ease_phase_fresh_v1/cases.json')}
    spec=dict(experiment='E74',checkpoints=hashes,inputs=inputs,test_evaluated=False,
        selection='Every unsuccessful E73 geometry-filter trial, including analytical. Replay original full trace exactly, diagnose first forbidden-contact control step only. No new task selection or success screening.',
        branches='Five one-control-step branches from identical pre-contact simulator state: filtered selected action; half of its motion component with measured bias retained; bias-only; analytical estimate computed from the SAME measured Gram/cross/bias; selected action after zeroing incoming arm/finger velocities (object velocity unchanged). Commands stay clipped. No future control updates or full task recovery measured.',
        geometry='Compare frozen filter midpoint/endpoint check with21 evenly spaced samples INCLUDING start along the commanded joint/finger line. Also reconstruct forbidden clearance at each of20 actual dynamic substep joint/finger states. Object pose is not advanced in that diagnostic reconstruction; all observed first collisions are finger/table, so that pair is unaffected.',
        limits='Failure-conditioned diagnostic, not a policy success evaluation. Exact simulator clones and velocity resets are privileged interventions, not deployable controls. Zeroing velocity changes state and constraint initialization; it does not simulate an implementable brake. One-step contact avoidance may only postpone failure. Denser samples do not prove continuous clearance. Same-history analytical action is not the independently rolled-out analytical controller.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('collision_counterfactual.py','geometry_filter.py','arm_pick_place.py','pick_place_control.py','action_objective_audit.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    results=[]
    for row in failures:
        case=row['case'];method=row['method'];t=tasks[case]
        old_path=REFERENCE/'trials'/f'{case}_{method}_geometry.json';inputs[str(old_path)]=fingerprint(old_path);old=json.loads(old_path.read_text())
        first=next(i for i,x in enumerate(old['trace']) if x['bad']);phase=old['trace'][first]['phase']
        env=DiagnosticArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();c=GuardedPose(env.n,models[method])
        probe=calibrate(env,t['home'],c,71100+t['candidate'])
        assert probe==json.loads((Path('outputs/ease_phase_fresh_v1')/f'probe_{case}.json').read_text())
        env.controller=c;env.filter_enabled=True;env.filter_log=[];env.diagnostic_enabled=True;env.event_step=first
        env.event_goal=next(goal for name,goal,_,_ in targets(env.xy,env.target) if name==phase)
        with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Task Jacobian forbidden')):
            replay=rollout(env,t['home'],c,reset=False)
        assert replay['trace']==old['trace'] and env.filter_log==old['filter_log'] and replay['success']==old['success']
        d=env.diagnostic
        assert d['branches']['selected']['collision'] and d['check']==old['filter_log'][first]
        assert d['branches']['selected']['position']==old['trace'][first]['position']
        analytic=next(r for r in all_rows if r['case']==case and r['method']=='analytical' and r['mode']=='geometry')
        result=dict(case=case,method=method,dof=env.n,phase=phase,step=first,first_contact=old['trace'][first]['bad'],analytical_full_task_success=analytic['success'],**d)
        (ROOT/'trials'/f'{case}_{method}.json').write_text(json.dumps(result,indent=2));results.append(result)
        print(case,method,phase,'avoids',[k for k,v in d['branches'].items() if not v['collision']],flush=True)
    for p,h in {**hashes,**inputs}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(results,indent=2))
    (ROOT/'audit.json').write_text(json.dumps(dict(exact_full_replays=len(results),branches=5*len(results),exact_selected_branches=len(results),input_hashes=inputs,hashes_unchanged=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()
