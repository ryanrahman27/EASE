"""E68: bounded procedural grasp settling with frozen motion estimators."""
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import torch
from .arm_pick_place import PickPlace, rollout, targets
from .mixed_sim import Arm, AXES
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .action_objective_audit import cloned_state, copy_controller
from .real_arm_reach import load_models
from .benchmark import fingerprint

ROOT = Path('outputs/ease_grasp_settling_v1')


def settling_sequence(env, mode, events):
    """Add at most 20 control steps before lift/transport; never filter a case."""
    assert mode in ('baseline', 'fixed', 'contact')
    for item in targets(env.xy, env.target):
        if mode != 'baseline' and item[0] in ('lift', 'transport'):
            goal = env.observe()['position'].copy()
            stable = 0
            for step in range(20):
                # Existing close semantics run exactly one step and keep updating
                # measured statistics/bias. Only XYZ reference is held fixed.
                yield ('close', goal, .035, 1)
                bilateral = len(env.contacts()[0]) == 2
                stable = stable + 1 if bilateral else 0
                if mode == 'contact' and stable >= 8:
                    break
            events.append(dict(before=item[0],steps=step+1,stable=stable,
                               stable_contact=stable>=8))
        yield item


def run():
    torch.set_num_threads(4)
    ROOT.mkdir(exist_ok=False); (ROOT/'trials').mkdir(); (ROOT/'source').mkdir()
    models={'analytical':None}; hashes={}; tasks=[]
    for group,folder in [('parent','outputs/ease_robust_training_v1'),('candidate','outputs/ease_estimator_distillation_v1/analytical')]:
        loaded,hh=load_models(folder); hashes.update(hh)
        models.update({f'{group}_{s}':loaded[f'neural_{s}'] for s in range(3)})
    inputs={}
    for split,folder in [('familiar','outputs/ease_arm_pick_place_v2'),('unseen','outputs/ease_arm_pick_place_unseen_v1')]:
        source=Path(folder)
        for name in ('cases.json','axis_catalog.json'): inputs[str(source/name)]=fingerprint(source/name)
        AXES.update(json.loads((source/'axis_catalog.json').read_text()))
        tasks.extend(dict(split=split,local_case=i,**t) for i,t in enumerate(json.loads((source/'cases.json').read_text())))
    protocol=dict(experiment='E68',checkpoints=hashes,inputs=inputs,test_evaluated=False,
        design='All 12 familiar and four IK-screened unseen procedural manipulation cases, seven frozen estimators, baseline/fixed/contact settling: 336 trials. Same 32-step calibration seed38000+local_case cloned across all conditions.',
        intervention='Before lift and transport hold current measured XYZ reference, upright tool-Z, closed gripper; fixed waits20 steps; contact waits until8 consecutive end-of-control-step bilateral contacts or20 steps. No early rejection on unstable contact; continue original sequence and retain every outcome. At most40 extra steps (1.6 seconds). Controller statistics and bias keep updating.',
        limitations='Simulator contact sensing is an extra privileged observation for the contact condition. Existing sequencer tolerates phase misses; no new safety or success gate. Fixed-duration control isolates some time effects but is not equal-duration per case. Previously observed development placements, only four screened unseen cases, no native extrapolation claim.',
        decision='Advance contact condition to native retention only if it improves pooled learned success over baseline and fixed, without pooled six-joint success or collision regression. Otherwise retain baseline and record a negative result. No parameter sweep or checkpoint selection.')
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for name in ['grasp_settling.py','arm_pick_place.py','pick_place_control.py','pick_place_fixture.py','retention_fresh.py','action_objective_audit.py','motion_transformer.py']:
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    records=[]
    for case,t in enumerate(tasks):
        env=PickPlace(Arm(**t['arm']),t['xy'],t['target']); initial=GuardedPose(env.n,None)
        probe=calibrate(env,t['home'],initial,38000+t['local_case'])
        (ROOT/f'probe_{case}.json').write_text(json.dumps(probe))
        for method,model in models.items():
            for mode in ('baseline','fixed','contact'):
                with cloned_state(env),patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
                    events=[]; controller=copy_controller(initial,model)
                    result=rollout(env,t['home'],controller,reset=False,sequence=settling_sequence(env,mode,events))
                assert env.steps==32
                row=dict(case=case,split=t['split'],local_case=t['local_case'],dof=env.n,family=env.arm.family,method=method,mode=mode,events=events,**result)
                (ROOT/'trials'/f'{case}_{method}_{mode}.json').write_text(json.dumps(row))
                records.append({k:v for k,v in row.items() if k!='trace'})
        print('case',case,t['split'],env.n,'baseline/fixed/contact', [sum(r['success'] for r in records if r['case']==case and r['mode']==mode) for mode in ('baseline','fixed','contact')],flush=True)
    assert len(records)==336
    for path,sha in {**inputs,**hashes}.items(): assert fingerprint(path)==sha
    (ROOT/'records.json').write_text(json.dumps(records,indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete',trials=len(records),test_evaluated=False)))


if __name__=='__main__': run()
