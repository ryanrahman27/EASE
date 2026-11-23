"""E66 fixed candidates: reaching retention, common tracking, native manipulation."""
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import torch
from .estimator_distillation import ROOT
from .robust_training_gate import run as gate
from .real_arm_reach import load_models
from .real_arm_pick_place import NativePickPlace
from .real_arms import verify_assets
from .retention_fresh import GuardedPose
from .pick_place_control import calibrate
from .action_objective_audit import cloned_state,copy_controller
from .native_phase_interventions import step
from .native_common_schedule import summarize_phase
from .native_reliability import rollout
from .benchmark import fingerprint


def run():
    torch.set_num_threads(4)
    assert json.loads((ROOT/'status.json').read_text())['status']=='trained'
    for condition in ['geometric','analytical']:
        gate('outputs/ease_paired_factorial_v2',ROOT/condition,ROOT/f'gate_{condition}')
    dest=ROOT/'native';dest.mkdir();(dest/'trials').mkdir()
    source=Path('outputs/ease_retention_fresh_v1');common=Path('outputs/ease_native_common_schedule_v2')
    tasks=json.loads((source/'cases.json').read_text());old=json.loads((source/'records.json').read_text());schedules=json.loads((common/'schedules.json').read_text())
    models={};hashes={}
    for condition in ['geometric','analytical']:
        mm,hh=load_models(ROOT/condition);hashes.update(hh)
        for seed in range(3):models[f'{condition}_{seed}']=mm[f'neural_{seed}']
    asset=verify_assets('outputs/ease_real_arm_assets_v1')
    (dest/'protocol.json').write_text(json.dumps(dict(checkpoints=hashes,asset_commit=asset['commit'],cases_sha256=fingerprint(source/'cases.json'),schedules_sha256=fingerprint(common/'schedules.json'),native_reference_sha256=fingerprint(source/'records.json'),common_reference_sha256=fingerprint(common/'records.json'),bias='Measured bias retained in every condition',test_evaluated=False),indent=2))
    for name in ['estimator_distillation_evaluate.py','native_phase_interventions.py','native_common_schedule.py','native_reliability.py','action_objective_audit.py','real_arm_pick_place.py','real_arms.py','pick_place_control.py','retention_fresh.py','robust_training_gate.py']:
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    starts=[]
    for case,t in enumerate(tasks):
        env=NativePickPlace('outputs/ease_real_arm_assets_v1',t['robot'],t['xy'],t['target'],.2,True);initial=GuardedPose(env.n,None)
        probe=calibrate(env,t['home'],initial,41000+case)
        (dest/f'probe_{case}.json').write_text(json.dumps(probe))
        assert probe==next(r['probe'] for r in old if r['case']==case and r['method']=='analytical')
        starts.append((env,initial,env.model.geom_contype.copy(),env.model.geom_conaffinity.copy()))
    records=[]
    for method,model in models.items():
        for case,t in enumerate(tasks):
            env,initial,contype,affinity=starts[case]
            for task in ['common','manipulation']:
                env.model.geom_contype[:]=0 if task=='common' else contype
                env.model.geom_conaffinity[:]=0 if task=='common' else affinity
                with cloned_state(env),patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
                    c=copy_controller(initial,model)
                    if task=='common':
                        trace=[step(env,c,s)[0] for s in schedules[case]]
                        phases=[summarize_phase(p,[x for x in trace if x['phase']==p]) for p in dict.fromkeys(x['phase'] for x in trace)]
                        result=dict(success=all(p['passes'] for p in phases),phases=phases,trace=trace,steps=env.steps)
                    else:result=rollout(env,c,object_release=t['robot']=='panda')
                assert env.steps==32
                record=dict(method=method,case=case,robot=t['robot'],task=task,**result)
                (dest/'trials'/f'{method}_{case}_{task}.json').write_text(json.dumps(record));records.append(record)
            print(method,case,'tracking/manipulation',records[-2]['success'],records[-1]['success'],records[-1]['failure'],flush=True)
    assert len(records)==144
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (dest/'records.json').write_text(json.dumps(records))
    (dest/'audit.json').write_text(json.dumps(dict(records=144,common_steps=122400,manipulation_control_steps=sum(r['steps']-32 for r in records if r['task']=='manipulation'),physical_calibration_steps=384,records_sha256=fingerprint(dest/'records.json'),unchanged_weights=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='evaluated',test_evaluated=False)))


if __name__=='__main__':run()
