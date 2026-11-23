"""E63: complete model-independent native pose schedule without geometry contact."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .benchmark import fingerprint
from .real_arms import verify_assets
from .real_arm_reach import load_models
from .real_arm_pick_place import NativePickPlace
from .retention_fresh import GuardedPose
from .pick_place_control import calibrate
from .native_reliability import sequence,bounded_error
from .action_objective_audit import cloned_state,copy_controller


def make_schedule(env,start):
    reference=np.asarray(start).copy();schedule=[]
    for phase,goal,aperture,budget in sequence(env):
        for step in range(budget):
            reference+=bounded_error(goal,reference,.002)
            schedule.append(dict(phase=phase,phase_step=step,goal=goal.tolist(),reference=reference.tolist(),aperture=aperture))
    assert len(schedule)==1700
    return schedule


def summarize_phase(phase,trace):
    tail=trace[-8:]
    return dict(phase=phase,steps=len(trace),passes=bool(len(tail)==8 and all(r['goal_error']<.006 and r['direction_error']<.08 for r in tail)),
        final_goal_error=trace[-1]['goal_error'],final_direction_error=trace[-1]['direction_error'],
        tracking_rmse=float(np.sqrt(np.mean([r['tracking_error']**2 for r in trace]))),
        final_joint_speed=trace[-1]['joint_speed'],clipped_steps=sum(r['clipped'] for r in trace))


def run(assets,parents,training,reference,output):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    asset=verify_assets(assets);ref=Path(reference);tasks=json.loads((ref/'cases.json').read_text());old=json.loads((ref/'records.json').read_text())
    models={'analytical':None};hashes={}
    for group,folder in [('original',parents),('uncorrected',str(Path(training)/'prediction')),('corrected',str(Path(training)/'control'))]:
        mm,hh=load_models(folder);hashes.update(hh)
        for seed in range(3):models[f'{group}_{seed}']=mm[f'neural_{seed}']
    protocol=dict(experiment='E63',checkpoints=hashes,asset_commit=asset['commit'],cases_sha256=fingerprint(ref/'cases.json'),reference_sha256=fingerprint(ref/'records.json'),
        design='All twelve observed E46 cases, ten frozen methods. One 32-step calibration per case is verified against E46, then full simulator state and measured controller statistics are cloned for every method. All geometry contype/conaffinity disabled. Fixed complete 1700-step schedule, no early stopping or object feedback. Native nominal nine pose waypoints and original phase budgets (220,260,60,260,260,260,60,260,60). One schedule per case starts at calibrated analytical tool position; ramp .002 m/step continuously across phases, independent of model observations. Native aperture sequence retained. Tool-Z target +Z. Same gains/damping/clips/limits/online updates.',
        metrics='Phase passes only if all final eight samples have goal XYZ error <.006 m and tool-Z direction norm <.08. Complete schedule passes all nine phases. Also retain tracking RMS, joint speeds and action clipping. No velocity threshold, task-success termination or outcome filtering.',
        limits='Artificial unconstrained tracking, not manipulation, safety or deployable success. Fixed nominal tool poses omit object-offset correction and object-height early release. Observed development cases and two native robot designs, not independent novel-embodiment validation. No training or parameter tuning.',test_evaluated=False)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'source').mkdir();(root/'trials').mkdir()
    for name in ['native_common_schedule.py','native_reliability.py','action_objective_audit.py','real_arm_pick_place.py','real_arms.py','pick_place_control.py','retention_fresh.py']:
        shutil.copy2(Path(__file__).with_name(name),root/'source'/name)
    schedules=[];initial_states=[]
    for case,t in enumerate(tasks):
        env=NativePickPlace(assets,t['robot'],t['xy'],t['target'],.2,True);c=GuardedPose(env.n,None)
        probe=calibrate(env,t['home'],c,41000+case)
        assert probe==next(r['probe'] for r in old if r['case']==case and r['method']=='analytical')
        schedules.append(make_schedule(env,env.observe()['position']))
        env.model.geom_contype[:]=0;env.model.geom_conaffinity[:]=0
        initial_states.append((env,c,probe))
    (root/'schedules.json').write_text(json.dumps(schedules));schedule_hash=fingerprint(root/'schedules.json');rows=[]
    for method,model in models.items():
        group=method.rsplit('_',1)[0] if '_' in method else method;seed=int(method[-1]) if '_' in method else None
        for case,t in enumerate(tasks):
            env,initial,probe=initial_states[case];c=copy_controller(initial,model)
            with cloned_state(env),patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged control forbidden')):
                env.model.geom_contype[:]=0;env.model.geom_conaffinity[:]=0;obs=env.observe();trace=[]
                for s in schedules[case]:
                    error=np.r_[np.asarray(s['reference'])-obs['position'],.15*(np.array([0,0,1])-obs['rotation'][:,2])]
                    jac,bias=c.predict();motion=jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(6),.35*error)
                    raw=motion+bias;before=obs;obs,command,info=env.step(np.clip(raw,-.04,.04),s['aperture']);c.update(before,obs,command)
                    assert not info['bad']
                    trace.append(dict(phase=s['phase'],phase_step=s['phase_step'],position=obs['position'].tolist(),q=obs['q'].tolist(),
                        tracking_error=float(np.linalg.norm(np.asarray(s['reference'])-obs['position'])),
                        goal_error=float(np.linalg.norm(np.asarray(s['goal'])-obs['position'])),direction_error=float(np.linalg.norm(np.array([0,0,1])-obs['rotation'][:,2])),
                        joint_speed=float(np.linalg.norm(env.data.qvel[env.vids])),clipped=bool(np.any(np.abs(raw)>.04))))
                total_steps=env.steps
            phases=[summarize_phase(phase,[r for r in trace if r['phase']==phase]) for phase in dict.fromkeys(r['phase'] for r in trace)]
            row=dict(method=method,group=group,seed=seed,case=case,robot=t['robot'],schedule_sha256=schedule_hash,steps=total_steps,
                passes=all(p['passes'] for p in phases),phases=phases,trace=trace)
            assert len(trace)==1700 and total_steps==1732 and env.steps==32
            (root/'trials'/f'{method}_{case}.json').write_text(json.dumps(row));rows.append(row)
            print(method,case,'passes',row['passes'],'failed_phases',[p['phase'] for p in phases if not p['passes']],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    assert fingerprint(root/'schedules.json')==schedule_hash
    (root/'records.json').write_text(json.dumps(rows));(root/'audit.json').write_text(json.dumps(dict(records=120,control_steps=204000,calibration_matches=12,shared_calibrated_starts=120,schedules_sha256=schedule_hash,records_sha256=fingerprint(root/'records.json'),unchanged_weights=True,test_evaluated=False),indent=2))
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)));report(root)


def report(root):
    root=Path(root);rows=json.loads((root/'records.json').read_text());audit=json.loads((root/'audit.json').read_text())
    assert len(rows)==audit['records']==120 and fingerprint(root/'records.json')==audit['records_sha256']
    assert fingerprint(root/'schedules.json')==audit['schedules_sha256'];p=json.loads((root/'protocol.json').read_text())
    for path,sha in p['checkpoints'].items():assert fingerprint(path)==sha
    for r in rows:
        assert json.loads((root/'trials'/f"{r['method']}_{r['case']}.json").read_text())==r
        assert len(r['trace'])==1700 and r['steps']==1732 and r['schedule_sha256']==audit['schedules_sha256']
        for phase in r['phases']:assert summarize_phase(phase['phase'],[x for x in r['trace'] if x['phase']==phase['phase']])==phase
        assert r['passes']==all(x['passes'] for x in r['phases'])
    lines=['# E63: complete common native pose-tracking schedule','',p['design'],'',p['metrics'],'',p['limits'],'',
        '| Group | Robot | Complete schedules pass | Per-seed passes /6 | Phase holds pass | Tracking RMS (m) |','|---|---|---:|---|---:|---:|']
    for group in ['analytical','original','uncorrected','corrected']:
        for robot in ['piper','panda']:
            rr=[r for r in rows if r['group']==group and r['robot']==robot]
            seeds=[None] if group=='analytical' else [0,1,2]
            scores=[sum(r['passes'] for r in rr if r['seed']==seed) for seed in seeds]
            rms=float(np.sqrt(np.mean([x['tracking_error']**2 for r in rr for x in r['trace']])))
            lines.append(f"| {group} | {robot} | {sum(r['passes'] for r in rr)}/{len(rr)} | {scores} | {sum(x['passes'] for r in rr for x in r['phases'])}/{9*len(rr)} | {rms:.6f} |")
    lines+=['','## Missed phase holds','']
    for group in ['analytical','original','uncorrected','corrected']:
        for robot in ['piper','panda']:
            misses=Counter(x['phase'] for r in rows if r['group']==group and r['robot']==robot for x in r['phases'] if not x['passes'])
            lines.append(f'{group} / {robot}: {dict(misses)}')
    lines+=['','All 120 complete schedules run without early termination: 204,000 control steps plus 384 physical calibration steps reused as 120 identical starts (32-step interaction cost charged per trial). Nine checkpoint hashes and archived traces verified. This is a fair common-schedule tracking comparison within this diagnostic, not a native manipulation result. No checkpoint promotion or final-test evaluation.']
    (root/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','parents','training','reference','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))
