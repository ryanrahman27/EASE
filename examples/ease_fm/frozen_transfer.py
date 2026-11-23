"""E67: frozen E66/E35 transfer to new axis families and native placements."""
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .benchmark import fingerprint
from .paired_data import register
from .broad_embodiment import build
from .robustness import evaluate
from .real_arm_reach import load_models
from .real_arm_pick_place import NativePickPlace,endpoint_ik
from .real_arms import verify_assets
from .retention_fresh import GuardedPose
from .pick_place_control import calibrate
from .action_objective_audit import cloned_state,copy_controller
from .native_common_schedule import make_schedule,summarize_phase
from .native_phase_interventions import step
from .native_reliability import sequence,rollout

ROOT=Path('outputs/ease_frozen_transfer_v1')


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir()
    models={'analytical':None};hashes={}
    for group,folder in [('parent','outputs/ease_robust_training_v1'),('candidate','outputs/ease_estimator_distillation_v1/analytical')]:
        mm,hh=load_models(folder);hashes.update(hh)
        for seed in range(3):models[f'{group}_{seed}']=mm[f'neural_{seed}']
    candidates=[dict(robot=robot,xy=[x,y],target=[x,y+(.04 if y<0 else -.04)]) for robot,xs in [('piper',[.215,.245,.275]),('panda',[.33,.38,.43])] for x in xs for y in [-.015,.045]]
    protocol=dict(experiment='E67',checkpoints=hashes,generation_seed=670000,native_ik_seed=67000,native_probe_seed=67100,native_candidates=candidates,test_evaluated=False,
        design='Frozen E66 analytical-target and E35 parents, all three seeds, versus fixed-probe analytical reaching / measured-history analytical native control. No training, checkpoint choice or parameter tuning.',
        procedural='Use E33 structural generator with seed670000; exclude all registered prior paired and E33 broad axis sequences. Sixteen fresh axis arrangements, four geometry/dynamics variants, two homes and eight goals per home: 64 bodies, 1024 cases. Same 150-step reaching horizon, no separate calibration; analytical first action +.01 rad, neural cold start. All structural rejections and policy outcomes retained. Body IDs are local to this manifest.',
        native='Twelve placements fixed before IK and policy evaluation. Same PiPER/Panda robot designs, new XY positions, same 4cm Y transfer. Nine nominal endpoint IK checks, bounded seed67000+candidate; no policy-outcome filtering. Retain rejected placements and do not replace them. One 32-step measured probe per accepted case, clone full simulator state/statistics for every method. Both complete 1700-step contact-disabled tracking and contact-enabled original manipulation. Bias retained throughout.',
        limits='Prospective development evaluation after E66, not the reserved final test. Native transfer changes placements, not robot designs. Structural screening limits scope. Seeds share bodies/goals; report per-seed, per-DoF, per-family and paired gains/losses with collision and interaction costs.',
        prior_manifests={p:fingerprint(p) for p in ['outputs/ease_paired_factorial_v2/manifest.json','outputs/ease_broad_development_v2/data/manifest.json']})
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for name in ['frozen_transfer.py','broad_embodiment.py','robustness.py','motion_transformer.py','mixed_sim.py','native_common_schedule.py','native_phase_interventions.py','native_reliability.py','action_objective_audit.py','real_arm_pick_place.py','real_arms.py','pick_place_control.py','retention_fresh.py']:
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    register('outputs/ease_broad_development_v2/data')
    manifest=build(ROOT/'data','outputs/ease_paired_factorial_v2',seed=670000)
    assert not manifest['goals']['test'] and not manifest['goals']['train']
    assert not ({r['sequence'] for r in manifest['families']} & set(manifest['excluded_axis_sequences']))
    # Keep generator schema provenance; evaluation identity is E67 in the parent protocol.
    (ROOT/'procedural').mkdir();solve=np.linalg.solve
    def guarded(a,b):
        assert a.shape[-2:]==(3,3),'Neural analytical estimator solve forbidden'
        return solve(a,b)
    for method,model in models.items():
        with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
            if model is None:rows=evaluate(manifest,model,dict(name='clean'),'cuda')
            else:
                with patch.object(np.linalg,'solve',side_effect=guarded),patch.object(torch.linalg,'solve',side_effect=RuntimeError('Neural inverse forbidden')):
                    rows=evaluate(manifest,model,dict(name='clean'),'cuda')
        assert len(rows)==1024 and len({r['goal_id'] for r in rows})==1024
        (ROOT/'procedural'/f'{method}.json').write_text(json.dumps(rows))
        print('procedural',method,sum(r['success'] for r in rows),'collisions',sum(r['collision'] for r in rows),flush=True)
    assets=verify_assets('outputs/ease_real_arm_assets_v1');native=ROOT/'native';native.mkdir();(native/'trials').mkdir()
    tasks=[];screen=[]
    for index,t in enumerate(candidates):
        env=NativePickPlace('outputs/ease_real_arm_assets_v1',t['robot'],t['xy'],t['target'],.2,True)
        q=env.observe()['q'];poses=[];rng=np.random.default_rng(67000+index)
        for _,goal,_,_ in sequence(env):
            q=endpoint_ik(env,goal,rng,q)
            if q is None:break
            poses.append(q.tolist())
        screen.append(dict(candidate=index,**t,endpoints=len(poses),accepted=len(poses)==9))
        if len(poses)==9:tasks.append(dict(candidate=index,**t,home=poses[0]))
        (native/'screening.json').write_text(json.dumps(screen,indent=2))
        print('native screen',index,t['robot'],len(poses),flush=True)
    (native/'cases.json').write_text(json.dumps(tasks,indent=2));starts=[];schedules=[]
    for case,t in enumerate(tasks):
        env=NativePickPlace('outputs/ease_real_arm_assets_v1',t['robot'],t['xy'],t['target'],.2,True);c=GuardedPose(env.n,None)
        probe=calibrate(env,t['home'],c,67100+t['candidate'])
        (native/f'probe_{case}.json').write_text(json.dumps(probe))
        starts.append((env,c,env.model.geom_contype.copy(),env.model.geom_conaffinity.copy()))
        schedules.append(make_schedule(env,env.observe()['position']))
    (native/'schedules.json').write_text(json.dumps(schedules));records=[]
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
                        result=dict(success=all(p['passes'] for p in phases),trace=trace,phases=phases,steps=env.steps)
                    else:result=rollout(env,c,object_release=t['robot']=='panda')
                assert env.steps==32
                record=dict(method=method,case=case,candidate=t['candidate'],robot=t['robot'],task=task,**result)
                (native/'trials'/f'{method}_{case}_{task}.json').write_text(json.dumps(record));records.append(record)
            print('native',method,case,'tracking/manipulation',records[-2]['success'],records[-1]['success'],records[-1].get('failure'),flush=True)
    assert len(records)==14*len(tasks)
    (native/'records.json').write_text(json.dumps(records))
    for path,sha in {**hashes,**protocol['prior_manifests']}.items():assert fingerprint(path)==sha
    audit=dict(procedural_records=7168,native_records=len(records),native_candidates=12,native_accepted=len(tasks),asset_commit=assets['commit'],manifest_sha256=fingerprint(ROOT/'data/manifest.json'),
        file_hashes={str(p.relative_to(ROOT)):fingerprint(p) for p in [*sorted((ROOT/'procedural').glob('*.json')),native/'cases.json',native/'screening.json',native/'schedules.json',native/'records.json']},unchanged_weights=True,test_evaluated=False)
    (ROOT/'audit.json').write_text(json.dumps(audit,indent=2));(ROOT/'status.json').write_text(json.dumps(dict(status='evaluated',test_evaluated=False)))


if __name__=='__main__':run()
