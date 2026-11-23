"""E84: frozen direction correction, complete-task retention."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from .direction_correction import DirectionArm
from .arm_pick_place import PickPlace,rollout,targets,ROT_SCALE,rotation_error
from .mixed_sim import AXES,Arm
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .real_arm_reach import load_models
from .benchmark import fingerprint

ROOT=Path('outputs/ease_direction_transfer_v1')
REFERENCE=Path('outputs/ease_geometry_transfer_v1')


from .direction_fulltask import FullTaskArm,sequence


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    inputs={}
    def read(p):
        inputs[str(p)]=fingerprint(p);return json.loads(p.read_text())
    prior=read(REFERENCE/'protocol.json');tasks=[];models={'analytical':None};hashes={}
    for folder in ('ease_arm_pick_place_v2','ease_arm_pick_place_unseen_v1'):
        p=Path('outputs')/folder;AXES.update(read(p/'axis_catalog.json'))
    tasks=read(Path('outputs/ease_phase_fresh_v1/cases.json'))
    for group,folder in [('parent','outputs/ease_robust_training_v1'),('candidate','outputs/ease_estimator_distillation_v1/analytical')]:
        mm,hh=load_models(folder);hashes.update(hh);models.update({f'{group}_{s}':mm[f'neural_{s}'] for s in range(3)})
    assert hashes==prior['checkpoints']
    spec=dict(experiment='E84',checkpoints=hashes,design='All17 separate E73 placements x7 frozen estimators x2 modes =238 complete tasks; no outcome selection. Original phase transitions, budgets, contacts and success criteria. Baseline geometry filtering, sustained correction exploratory primary, No pulse selection or retuning. Same bodies and previously studied development placements; not new embodiments or a reserved final test.',
        settings='Import E82 corrected_action unchanged: fixed no-velocity response model, +/-0.01 single-joint candidates, +/-0.04 bounds, 21 path points, 2mm warning, >1e-6 clearance improvement, predicted task-error allowance0.001. Known whole-arm geometry with object held at current pose. No simulator future outcomes in action selection.',
        pulse='One five-step pulse starting on first online warning, including that step and continuing across phase boundaries if needed. No retrigger. Sustained mode checks throughout task. Calibration32 and all subsequent measured updates retained; each mode calibrated identically.',
        gate='Primary must improve pooled learned task success, not increase pooled collisions, and retain EVERY baseline-successful six-joint case. Report every loss, checkpoint and DOF breakdown, interactions and additional geometry queries. No tuning on outcomes; historical development data, not independent generalization.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('direction_transfer.py','direction_fulltask.py','direction_correction.py','response_prediction.py','geometry_filter.py','arm_pick_place.py','pick_place_control.py'):
        p=Path(__file__).with_name(name);frozen=Path('outputs/ease_direction_fulltask_v1/source')/name
        if frozen.exists():assert fingerprint(p)==fingerprint(frozen)
        shutil.copy2(p,ROOT/'source'/name)
    rows=[];exact=0
    for case,t in enumerate(tasks):
        for method,model in models.items():
            old=read(REFERENCE/'trials'/f'{case}_{method}_geometry.json')
            for mode in ('baseline','sustained'):
                env=FullTaskArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();env.initialize_response();c=GuardedPose(env.n,model)
                probe=calibrate(env,t['home'],c,71100+t['candidate'])
                assert probe==read(Path('outputs/ease_phase_fresh_v1')/f'probe_{case}.json')
                env.controller=c;env.filter_enabled=True;env.filter_log=[];env.correction_log=[];env.mode=mode;env.first_warning=None;env.pulse_used=0
                with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Task Jacobian forbidden')):
                    result=rollout(env,t['home'],c,reset=False,sequence=sequence(env))
                assert env.response.updates==env.steps and len(env.correction_log)==len(result['trace'])
                if mode=='baseline':
                    assert result['trace']==old['trace'] and result['success']==old['success'] and env.filter_log==old['filter_log'];exact+=1
                logs=[r['correction'] for r in env.correction_log if r['correction'] is not None]
                row=dict(case=case,method=method,mode=mode,dof=env.n,first_warning=env.first_warning,pulse_used=env.pulse_used,
                    changed_actions=sum(r['changed'] for r in logs),correction_queries=sum(r['queries'] for r in logs),filter_queries=sum(r['queries'] for r in env.filter_log),**result)
                (ROOT/'trials'/f'{case}_{method}_{mode}.json').write_text(json.dumps(dict(**row,correction_log=env.correction_log)))
                rows.append({k:v for k,v in row.items() if k!='trace'})
        print('case',case,'of17 complete',flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'audit.json').write_text(json.dumps(dict(inputs=inputs,exact_baselines=exact,hashes_unchanged=True,causal_updates_verified=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()
