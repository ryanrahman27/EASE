"""E64: failure-conditioned, matched-state motion/bias interventions."""
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .native_common_schedule import summarize_phase
from .action_objective_audit import cloned_state, copy_controller
from .real_arm_pick_place import NativePickPlace
from .real_arm_reach import load_models
from .real_arms import verify_assets
from .retention_fresh import GuardedPose
from .pick_place_control import calibrate
from .benchmark import fingerprint


def step(env, c, s, analytical=False, zero_bias=False):
    obs=env.observe()
    error=np.r_[np.asarray(s['reference'])-obs['position'],.15*(np.array([0,0,1])-obs['rotation'][:,2])]
    jac,bias=c.predict()
    if analytical: jac=np.linalg.solve(c.gram,c.cross).T
    motion=jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(6),.35*error)
    raw=motion+(0 if zero_bias else bias)
    after,command,info=env.step(np.clip(raw,-.04,.04),s['aperture'])
    dq=after['q']-obs['q']
    dx=np.r_[after['position']-obs['position'],.15*(after['rotation'][:,2]-obs['rotation'][:,2])]
    c.update(obs,after,command)
    assert not info['bad']
    trace=dict(phase=s['phase'],phase_step=s['phase_step'],position=after['position'].tolist(),q=after['q'].tolist(),
        tracking_error=float(np.linalg.norm(np.asarray(s['reference'])-after['position'])),
        goal_error=float(np.linalg.norm(np.asarray(s['goal'])-after['position'])),
        direction_error=float(np.linalg.norm(np.array([0,0,1])-after['rotation'][:,2])),
        joint_speed=float(np.linalg.norm(env.data.qvel[env.vids])),clipped=bool(np.any(np.abs(raw)>.04)))
    telemetry=dict(bias_norm=float(np.linalg.norm(bias)),motion_norm=float(np.linalg.norm(motion)),
        response_residual=float(np.linalg.norm(command-dq-bias)),motion_prediction_error=float(np.linalg.norm(jac@dq-dx)),
        singular_values=np.linalg.svd(jac,compute_uv=False).tolist(),bias=bias.tolist(),motion=motion.tolist(),command=command.tolist(),dq=dq.tolist())
    return trace,telemetry


def run():
    torch.set_num_threads(4)
    root=Path('outputs/ease_native_phase_interventions_v1');root.mkdir(exist_ok=False)
    source=Path('outputs/ease_native_common_schedule_v2')
    rows=json.loads((source/'records.json').read_text());schedules=json.loads((source/'schedules.json').read_text())
    cases=json.loads(Path('outputs/ease_retention_fresh_v1/cases.json').read_text())
    old=json.loads(Path('outputs/ease_retention_fresh_v1/records.json').read_text())
    selected=[r for r in rows if not r['passes']]
    modes={'baseline':(False,False),'analytical_motion':(True,False),'zero_bias':(False,True),'analytical_zero_bias':(True,True)}
    protocol=dict(experiment='E64',source_sha256=fingerprint(source/'records.json'),selection='All 15 failed E63 schedules; all 37 failed phases. Outcome-conditioned diagnostic, not a performance estimate.',
        design='Replay each full original schedule exactly. At each failed phase entry clone full simulator state and measured controller statistics for four full-phase branches. Swap learned Jacobian for ridge analytical estimate from the same evolving statistics and/or remove additive measured bias. Online history updates continue in every branch. No privileged dynamics or Jacobian. Zero bias is an ablation, not an oracle. Original reference, thresholds, gains, contact disabling and phase budget retained. Later branches start from original history, not preceding interventions.',
        test_evaluated=False)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'trials').mkdir();(root/'source').mkdir()
    shutil.copy2(__file__,root/'source'/Path(__file__).name)
    verify_assets('outputs/ease_real_arm_assets_v1')
    models={};hashes={}
    for group,folder in [('original','outputs/ease_robust_training_v1'),('uncorrected','outputs/ease_response_corrected_training_v1/prediction'),('corrected','outputs/ease_response_corrected_training_v1/control')]:
        mm,hh=load_models(folder);hashes.update(hh)
        for seed in range(3):models[f'{group}_{seed}']=mm[f'neural_{seed}']
    protocol['checkpoints']=hashes;(root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    starts={};results=[];replays=0
    for case in sorted({r['case'] for r in selected}):
        t=cases[case];env=NativePickPlace('outputs/ease_real_arm_assets_v1',t['robot'],t['xy'],t['target'],.2,True)
        c=GuardedPose(env.n,None);probe=calibrate(env,t['home'],c,41000+case)
        assert probe==next(r['probe'] for r in old if r['case']==case and r['method']=='analytical')
        env.model.geom_contype[:]=0;env.model.geom_conaffinity[:]=0;starts[case]=(env,c)
    for original in selected:
        case=original['case'];method=original['method'];env,initial=starts[case];model=models[method]
        failed={p['phase'] for p in original['phases'] if not p['passes']}
        with cloned_state(env),patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged control forbidden')):
            c=copy_controller(initial,model)
            for index,s in enumerate(schedules[case]):
                if s['phase_step']==0 and s['phase'] in failed:
                    phase_schedule=[x for x in schedules[case] if x['phase']==s['phase']]
                    for mode,(analytic,zero) in modes.items():
                        with cloned_state(env):
                            branch=copy_controller(c,model);trace=[];telemetry=[]
                            for x in phase_schedule:
                                a,b=step(env,branch,x,analytic,zero);trace.append(a);telemetry.append(b)
                        result=dict(method=method,case=case,robot=original['robot'],phase=s['phase'],mode=mode,
                            summary=summarize_phase(s['phase'],trace),trace=trace,telemetry=telemetry)
                        (root/'trials'/f'{method}_{case}_{s["phase"]}_{mode}.json').write_text(json.dumps(result))
                        if mode=='baseline': assert trace==[x for x in original['trace'] if x['phase']==s['phase']], 'Branch replay mismatch; trace saved'
                        results.append(result)
                actual,telemetry=step(env,c,s)
                if actual!=original['trace'][index]:
                    (root/'replay_mismatch.json').write_text(json.dumps(dict(method=method,case=case,index=index,actual=actual,expected=original['trace'][index])))
                    raise AssertionError('Original replay mismatch; evidence saved')
                replays+=1
        print(method,case,'exact replay; branches',len(results),flush=True)
    assert len(results)==148 and replays==25500
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'records.json').write_text(json.dumps(results))
    audit=dict(exact_replay_steps=replays,exact_baseline_branches=37,branches=len(results),branch_steps=sum(len(r['trace']) for r in results),unchanged_weights=True,test_evaluated=False,records_sha256=fingerprint(root/'records.json'))
    (root/'audit.json').write_text(json.dumps(audit,indent=2))
    lines=['# E64: matched-state phase interventions','',protocol['selection'],'',protocol['design'],'',
        '| Mode | Holds recovered /37 | PiPER /2 | Panda /35 | Mean clipped-step fraction | Mean motion prediction error |',
        '|---|---:|---:|---:|---:|---:|']
    for mode in modes:
        rr=[r for r in results if r['mode']==mode]
        lines.append(f'| {mode} | {sum(r["summary"]["passes"] for r in rr)} | {sum(r["summary"]["passes"] for r in rr if r["robot"]=="piper")} | {sum(r["summary"]["passes"] for r in rr if r["robot"]=="panda")} | {np.mean([np.mean([x["clipped"] for x in r["trace"]]) for r in rr]):.4f} | {np.mean([np.mean([x["motion_prediction_error"] for x in r["telemetry"]]) for r in rr]):.6f} |')
    lines+=['','All 25,500 original replay steps and 37 baseline phase branches match E63 exactly. Telemetry includes estimated singular values, additive bias, motion, executed command, realized joint displacement, command-minus-displacement-minus-bias residual and realized task-delta prediction error. Prediction error uses realized displacement; it is not a causal Jacobian-error oracle. Phase averages weight phases equally.','', 'No training, checkpoint promotion, final-test evaluation or manipulation success claim. Branches are dependent and selected for baseline failure; recovery does not establish whole-task reliability.']
    (root/'report.md').write_text('\n'.join(lines)+'\n');(root/'status.json').write_text(json.dumps({'status':'complete'}))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':run()
