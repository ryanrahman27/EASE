"""E73: frozen geometry filter versus fixed slowing on E71 placements."""
import json
from pathlib import Path
import shutil
from contextlib import nullcontext
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from .geometry_filter import GeometryArm
from .arm_pick_place import rollout
from .mixed_sim import Arm,AXES
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .action_objective_audit import cloned_state,copy_controller
from .real_arm_reach import load_models
from .benchmark import fingerprint

ROOT=Path('outputs/ease_geometry_transfer_v1')
REFERENCE=Path('outputs/ease_phase_fresh_v1')


def slow_action(action,bias):
    action=np.clip(action,-.04,.04);bias=np.clip(bias,-.04,.04)
    return np.clip(bias+.5*(action-bias),-.04,.04)


class ComparisonArm(GeometryArm):
    def step(self,action,aperture):
        if getattr(self,'slow_enabled',False):
            selected=slow_action(action,self.controller.bias)
            self.slow_changed+=int(not np.array_equal(action,selected));action=selected
        return super().step(action,aperture)


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    tasks=json.loads((REFERENCE/'cases.json').read_text());prior=json.loads((REFERENCE/'protocol.json').read_text())
    inputs={**prior['inputs'],str(REFERENCE/'cases.json'):fingerprint(REFERENCE/'cases.json'),str(REFERENCE/'screening.json'):fingerprint(REFERENCE/'screening.json')}
    for p in ('outputs/ease_arm_pick_place_v2/axis_catalog.json','outputs/ease_arm_pick_place_unseen_v1/axis_catalog.json'):AXES.update(json.loads(Path(p).read_text()))
    sources={}
    for name in ('geometry_filter.py','arm_pick_place.py','pick_place_control.py','retention_fresh.py','action_objective_audit.py'):
        p=Path(__file__).with_name(name);assert fingerprint(p)==fingerprint(Path('outputs/ease_geometry_filter_v2/source')/name)
        sources[str(p)]=fingerprint(p);shutil.copy2(p,ROOT/'source'/name)
    shutil.copy2(Path(__file__),ROOT/'source'/'geometry_transfer.py')
    models={'analytical':None};hashes={}
    for group,folder in [('parent','outputs/ease_robust_training_v1'),('candidate','outputs/ease_estimator_distillation_v1/analytical')]:
        mm,hh=load_models(folder);hashes.update(hh);models.update({f'{group}_{s}':mm[f'neural_{s}'] for s in range(3)})
    assert hashes==prior['checkpoints']
    spec=dict(experiment='E73',inputs=inputs,checkpoints=hashes,sources=sources,test_evaluated=False,
        design='All17 accepted E71 placements, seven frozen estimators, baseline/geometry/slow:357 trials. E71 screening (17/18) retained without rerunning or replacing rejected cases. Same bodies, new relative to E72 placements; already observed E71 development data, not an independent reserved test.',
        geometry='Exact frozen E72v2 2mm midpoint/endpoint filter; same known robot/scene geometry and exact current pose. No changes to margins, scales, physics, horizons, weights or task-success criteria.',
        slow='Geometry-free fixed0.5 task-motion scale, preserving clipped measured bias: bias+0.5*(clipped action-bias), clipped+/-0.04. No gripper scaling. Same original phase budgets. Constant0.5 fixed before E73 outcomes; no sweep. Geometry query function blocked in baseline/slow trials.',
        matching='One32-step probe seed71100+candidate per case, exact E71 replay. Clone full simulator state and measured controller statistics across all conditions; retain probe contact. All baseline traces must reproduce E71 exactly.',
        decision='Replication requires geometry pooled learned success greater than baseline, no pooled collision increase and no per-checkpoint aggregate six-joint success loss. To support benefit beyond this slowing control also require greater pooled success than slow. Report casewise regressions and costs regardless. One fixed slowing control cannot exclude all speed schedules or prove geometry is uniquely necessary.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));rows=[]
    for case,t in enumerate(tasks):
        env=ComparisonArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();initial=GuardedPose(env.n,None)
        probe=calibrate(env,t['home'],initial,71100+t['candidate'])
        assert probe==json.loads((REFERENCE/f'probe_{case}.json').read_text())
        margins=env.model.geom_margin.copy()
        for method,model in models.items():
            for mode in ('baseline','geometry','slow'):
                env.filter_enabled=mode=='geometry';env.slow_enabled=mode=='slow';env.filter_log=[];env.slow_changed=0
                guard=nullcontext() if mode=='geometry' else patch.object(env,'clearance',side_effect=RuntimeError('Geometry access forbidden in control'))
                with cloned_state(env),guard,patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged task Jacobian forbidden')):
                    env.controller=copy_controller(initial,model);result=rollout(env,t['home'],env.controller,reset=False)
                assert env.steps==32 and np.array_equal(margins,env.model.geom_margin)
                if mode=='baseline':
                    old=json.loads((REFERENCE/'trials'/f'{case}_{method}_baseline.json').read_text())
                    assert result['trace']==old['trace'] and result['success']==old['success']
                if mode=='geometry':assert len(env.filter_log)==len(result['trace'])
                row=dict(case=case,candidate=t['candidate'],family=t['arm']['family'],split=t['split'],dof=env.n,method=method,mode=mode,
                    queries=sum(x['queries'] for x in env.filter_log),changed_actions=env.slow_changed+sum(x['changed'] for x in env.filter_log),fallbacks=sum(not x['feasible'] for x in env.filter_log),**result)
                (ROOT/'trials'/f'{case}_{method}_{mode}.json').write_text(json.dumps(dict(**row,filter_log=env.filter_log)))
                rows.append({k:v for k,v in row.items() if k!='trace'})
        print('case',case,'baseline/geometry/slow',[sum(r['success'] for r in rows if r['case']==case and r['mode']==m) for m in ('baseline','geometry','slow')],flush=True)
    assert len(rows)==357
    for p,h in {**inputs,**hashes,**sources}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'audit.json').write_text(json.dumps(dict(trials=357,exact_baselines=119,hashes_unchanged=True,control_geometry_blocked=True,physical_margins_unchanged=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()
