"""E72: shared privileged-geometry kinematic action filter."""
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from .arm_pick_place import PickPlace,rollout
from .mixed_sim import Arm,AXES
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .action_objective_audit import cloned_state,copy_controller
from .real_arm_reach import load_models
from .benchmark import fingerprint

ROOT=Path('outputs/ease_geometry_filter_v2')
MARGIN=.002


def allowed_pair(pair,gids):
    return gids['cube'] in pair and bool(pair.intersection({gids['table'],gids['gripper_left_pad'],gids['gripper_right_pad']}))


class GeometryArm(PickPlace):
    def prepare_geometry(self):
        self.geometry_model=mujoco.MjModel.from_xml_string(self.xml)
        # Expanded detection is confined to a separate model; physical margins
        # and constraints of the executed simulation are never changed.
        self.geometry_model.geom_margin[:]+=MARGIN
        self.geometry_data=mujoco.MjData(self.geometry_model)

    def clearance(self,q,fingers):
        mujoco.mj_copyData(self.geometry_data,self.geometry_model,self.data)
        self.geometry_data.qpos[:self.n]=q
        self.geometry_data.qpos[self.n:self.n+2]=fingers
        mujoco.mj_forward(self.geometry_model,self.geometry_data)
        distances=[float(c.dist) for c in self.geometry_data.contact[:self.geometry_data.ncon]
                   if not allowed_pair({int(c.geom1),int(c.geom2)},self.gids)]
        return min([MARGIN]+distances)

    def select_action(self,action,aperture,bias):
        q=self.observe()['q'];fingers=self.data.qpos[self.n:self.n+2].copy()
        action=np.clip(action,-.04,.04);bias=np.clip(bias,-.04,.04)
        current=self.clearance(q,fingers);queries=1;options=[]
        for scale in (1.,.5,.25,0.):
            proposed=action.copy() if scale==1. else bias.copy() if scale==0. else np.clip(bias+scale*(action-bias),-.04,.04)
            target=np.clip(q+proposed,-2.5,2.5)
            score=min(self.clearance(q+t*(target-q),fingers+t*(aperture-fingers)) for t in (.5,1.))
            queries+=2;options.append((score,scale,proposed))
            if score>=min(MARGIN,current)-1e-6:break
        feasible=options[-1][0]>=min(MARGIN,current)-1e-6
        score,scale,selected=options[-1] if feasible else max(options,key=lambda x:(x[0],-x[1]))
        return selected,dict(scale=scale,current_clearance=current,selected_clearance=score,queries=queries,feasible=feasible,
                             changed=bool(not np.array_equal(selected,action)))

    def step(self,action,aperture):
        if getattr(self,'filter_enabled',False):
            action,entry=self.select_action(action,aperture,self.controller.bias)
            self.filter_log.append(entry)
        return super().step(action,aperture)


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    prior=json.loads(Path('outputs/ease_grasp_settling_v1/protocol.json').read_text());models={'analytical':None};hashes={};tasks=[]
    for group,folder in [('parent','outputs/ease_robust_training_v1'),('candidate','outputs/ease_estimator_distillation_v1/analytical')]:
        mm,hh=load_models(folder);hashes.update(hh);models.update({f'{group}_{s}':mm[f'neural_{s}'] for s in range(3)})
    assert hashes==prior['checkpoints']
    for split,folder in [('familiar','outputs/ease_arm_pick_place_v2'),('unseen','outputs/ease_arm_pick_place_unseen_v1')]:
        p=Path(folder);AXES.update(json.loads((p/'axis_catalog.json').read_text()))
        tasks.extend(dict(split=split,local_case=i,**t) for i,t in enumerate(json.loads((p/'cases.json').read_text())))
    spec=dict(experiment='E72',checkpoints=hashes,inputs=prior['inputs'],test_evaluated=False,
        design='All16 original procedural placements, seven frozen estimators, original baseline and shared geometry filter:224 trials. Original phase budgets, physics, collision masks, success rules and measured32-step probes unchanged. All failed cases retained. V2 passes scale1 actions through bit-exactly; v1 interrupted and preserved because reconstructing them through bias arithmetic introduced roundoff.',
        filter='Known robot/scene geometry and exact current simulator pose; separate model with2mm geometry margin. Evaluate midpoint/end of joint-target interpolation, including commanded finger position, object fixed at current pose. Candidate action is bias + scale*(clipped action-bias), scale1,.5,.25,0; clipped bias retained. First candidate whose minimum forbidden clearance is at least min(2mm,current clearance)-1e-6. If none qualifies, choose greatest clearance (ties prefer smaller scale). No task Jacobian access, model training or dynamic lookahead. Geometry queries cost recorded separately.',
        limits='Privileged kinematics and object/scene pose are extra information shared equally across estimators. Forward kinematic checks do not predict dynamic overshoot or carried-object motion and are not a safety guarantee. Two interpolation samples do not guarantee continuous-path clearance. An already penetrating state permits nonworsening motion. This is not a history-only policy or new embodiment validation.',
        decision='Advance to frozen fresh-placement validation only if pooled learned success improves, pooled collisions do not increase and every checkpoint retains its aggregate six-joint success count. Report all casewise losses, filter fallback and query costs. Otherwise retain original controller and do not tune margins/scales on outcomes.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('geometry_filter.py','arm_pick_place.py','pick_place_control.py','retention_fresh.py','action_objective_audit.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    rows=[]
    for case,t in enumerate(tasks):
        env=GeometryArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();initial=GuardedPose(env.n,None)
        probe=calibrate(env,t['home'],initial,38000+t['local_case'])
        assert probe==json.loads((Path('outputs/ease_grasp_settling_v1')/f'probe_{case}.json').read_text())
        margin=env.model.geom_margin.copy()
        for method,model in models.items():
            for mode in ('baseline','geometry'):
                env.filter_enabled=mode=='geometry';env.filter_log=[]
                with cloned_state(env),patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged task Jacobian forbidden')):
                    env.controller=copy_controller(initial,model)
                    result=rollout(env,t['home'],env.controller,reset=False)
                assert env.steps==32 and np.array_equal(env.model.geom_margin,margin)
                if mode=='baseline':
                    old=json.loads((Path('outputs/ease_grasp_settling_v1/trials')/f'{case}_{method}_baseline.json').read_text())
                    assert result['trace']==old['trace'] and result['success']==old['success']
                else:assert len(env.filter_log)==len(result['trace'])
                row=dict(case=case,method=method,mode=mode,dof=env.n,split=t['split'],queries=sum(x['queries'] for x in env.filter_log),
                         changed_actions=sum(x['changed'] for x in env.filter_log),fallbacks=sum(not x['feasible'] for x in env.filter_log),**result)
                (ROOT/'trials'/f'{case}_{method}_{mode}.json').write_text(json.dumps(dict(**row,filter_log=env.filter_log)))
                rows.append({k:v for k,v in row.items() if k!='trace'})
        print('case',case,'baseline/geometry',[sum(r['success'] for r in rows if r['case']==case and r['mode']==m) for m in ('baseline','geometry')],flush=True)
    assert len(rows)==224
    for p,h in {**hashes,**prior['inputs']}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'audit.json').write_text(json.dumps(dict(trials=224,exact_baselines=112,physical_margins_unchanged=True,hashes_unchanged=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()
