"""E70: bounded phase budgets and transition gates, frozen estimators."""
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import torch
from .arm_pick_place import PickPlace,rollout,targets
from .mixed_sim import Arm,AXES
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .action_objective_audit import cloned_state,copy_controller
from .real_arm_reach import load_models
from .benchmark import fingerprint

ROOT=Path('outputs/ease_phase_retention_v1')
MOVING={'approach','descend','lift','transport','lower','retreat'}


def transition_ok(phase,trace):
    if phase in MOVING:
        return len(trace)>=5 and all(r['position_error']<.008 and r['rotation_error']<.08 for r in trace[-5:])
    if phase=='close':
        return len(trace)>=8 and all(r['bilateral'] and r['position_error']<.008 and r['rotation_error']<.08 for r in trace[-8:])
    return True


class ContactArm(PickPlace):
    def step(self,action,aperture):
        result=super().step(action,aperture)
        if hasattr(self,'end_contacts'):self.end_contacts.append(len(self.contacts()[0])==2)
        return result


def trial(env,home,c,mode):
    if mode=='baseline':return dict(**rollout(env,home,c,reset=False),abort=None)
    assert mode in ('extended','gated')
    trace=[];phases=[];abort=None
    for phase,goal,aperture,budget in targets(env.xy,env.target):
        budget+=40 if phase in MOVING else 20 if phase=='close' else 0
        env.end_contacts=[]
        result=rollout(env,home,c,reset=False,sequence=[(phase,goal,aperture,budget)])
        assert len(result['trace'])==len(env.end_contacts)
        annotated=[dict(**r,bilateral=b) for r,b in zip(result['trace'],env.end_contacts)]
        trace+=result['trace'];phases+=result['phases']
        if mode=='gated' and (env.bad or not transition_ok(phase,annotated)):
            abort=('collision:' if env.bad else 'transition:')+phase
            break
    result.update(trace=trace,phases=phases,abort=abort)
    if abort is not None:result['success']=False
    return result


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    models={'analytical':None};hashes={};tasks=[];inputs={}
    for group,folder in [('parent','outputs/ease_robust_training_v1'),('candidate','outputs/ease_estimator_distillation_v1/analytical')]:
        mm,hh=load_models(folder);hashes.update(hh);models.update({f'{group}_{s}':mm[f'neural_{s}'] for s in range(3)})
    for split,folder in [('familiar','outputs/ease_arm_pick_place_v2'),('unseen','outputs/ease_arm_pick_place_unseen_v1')]:
        p=Path(folder);AXES.update(json.loads((p/'axis_catalog.json').read_text()))
        for name in ('cases.json','axis_catalog.json'):inputs[str(p/name)]=fingerprint(p/name)
        tasks.extend(dict(split=split,local_case=i,**t) for i,t in enumerate(json.loads((p/'cases.json').read_text())))
    protocol=dict(experiment='E70',inputs=inputs,checkpoints=hashes,test_evaluated=False,
        design='All16 procedural placements, seven frozen estimators, baseline/extended/gated:336 trials. Same32-step calibration seed38000+local_case and cloned state. No path geometry, collision mask or physics change.',
        budget='Extended and gated add40 maximum steps to each moving phase and20 to close, at most260 additional steps. Original early-exit pose rule retained. Other phases unchanged. No parameter search.',
        gates='At phase end only: abort on any accumulated forbidden contact; moving phases require five final consecutive steps within8mm and .08 tool-Z direction error. Close requires eight final consecutive steps satisfying the same pose tolerance plus bilateral endpoint contacts. Release/settle have no new gate. Aborts count as failures even if terminal object happens to be at target. This is not an immediate collision shield or clearance planner.',
        decision='Adopt neither intervention unless learned success improves over baseline with no per-checkpoint six-joint success loss and no pooled collision increase. Gate-only collision reduction through aborted tasks is not success. Compare gated to identical-budget extended control; verify their trajectories share exact prefixes. Development cases only; no native expansion without retention.')
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for name in ('phase_retention.py','arm_pick_place.py','pick_place_control.py','retention_fresh.py','action_objective_audit.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    rows=[]
    for case,t in enumerate(tasks):
        env=ContactArm(Arm(**t['arm']),t['xy'],t['target']);initial=GuardedPose(env.n,None)
        probe=calibrate(env,t['home'],initial,38000+t['local_case'])
        assert probe==json.loads((Path('outputs/ease_grasp_settling_v1')/f'probe_{case}.json').read_text())
        for method,model in models.items():
            results={}
            for mode in ('baseline','extended','gated'):
                with cloned_state(env),patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
                    result=trial(env,t['home'],copy_controller(initial,model),mode)
                assert env.steps==32
                row=dict(case=case,method=method,mode=mode,dof=env.n,split=t['split'],**result)
                (ROOT/'trials'/f'{case}_{method}_{mode}.json').write_text(json.dumps(row))
                rows.append({k:v for k,v in row.items() if k!='trace'});results[mode]=result
            old=json.loads((Path('outputs/ease_grasp_settling_v1/trials')/f'{case}_{method}_baseline.json').read_text())
            assert results['baseline']['trace']==old['trace'] and results['baseline']['success']==old['success']
            assert results['gated']['trace']==results['extended']['trace'][:len(results['gated']['trace'])]
        print('case',case,'baseline/extended/gated',[sum(r['success'] for r in rows if r['case']==case and r['mode']==mode) for mode in ('baseline','extended','gated')],flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    assert len(rows)==336
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'audit.json').write_text(json.dumps(dict(trials=336,exact_baselines=112,exact_gate_prefixes=112,hashes_unchanged=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()
