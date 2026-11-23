"""E65: complete-schedule estimator/bias factorial with identical starts."""
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import torch
from .native_phase_interventions import step
from .native_common_schedule import summarize_phase
from .action_objective_audit import cloned_state, copy_controller
from .real_arm_pick_place import NativePickPlace
from .real_arm_reach import load_models
from .real_arms import verify_assets
from .retention_fresh import GuardedPose
from .pick_place_control import calibrate
from .benchmark import fingerprint

ROOT=Path('outputs/ease_native_full_interventions_v1')
SOURCE=Path('outputs/ease_native_common_schedule_v2')


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False)
    (ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    rows=json.loads((SOURCE/'records.json').read_text())
    schedules=json.loads((SOURCE/'schedules.json').read_text())
    ref=Path('outputs/ease_retention_fresh_v1')
    cases=json.loads((ref/'cases.json').read_text());old=json.loads((ref/'records.json').read_text())
    asset=verify_assets('outputs/ease_real_arm_assets_v1')
    models={'analytical':None};hashes={}
    for group,folder in [('original','outputs/ease_robust_training_v1'),('uncorrected','outputs/ease_response_corrected_training_v1/prediction'),('corrected','outputs/ease_response_corrected_training_v1/control')]:
        mm,hh=load_models(folder);hashes.update(hh)
        for seed in range(3):models[f'{group}_{seed}']=mm[f'neural_{seed}']
    protocol=dict(experiment='E65',checkpoints=hashes,asset_commit=asset['commit'],source_sha256=fingerprint(SOURCE/'records.json'),
        schedules_sha256=fingerprint(SOURCE/'schedules.json'),cases_sha256=fingerprint(ref/'cases.json'),
        design='All twelve E63 development cases, including successes. Ten estimators (analytical plus nine frozen learned models), crossed with measured additive bias versus zero additive bias for all 1700 steps after shared 32-step calibration. All geometry contacts disabled. Same fixed references, gains, thresholds and evolving history statistics. Analytical substitution from calibration is independent of the unused learned checkpoint, so run it once per case/bias condition rather than counting nine identical copies. All 120 measured-bias trials must exactly replay E63. No tuning, training, phase-dependent switching or early stopping.',
        metric='All nine phases must hold XYZ error <.006 m and direction norm <.08 for their final eight steps. Report paired gains/losses, robot and seed breakdown. Diagnostic unconstrained tracking, not manipulation or independent embodiment validation.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for name in ['native_full_interventions.py','native_phase_interventions.py','native_common_schedule.py','action_objective_audit.py','pick_place_control.py','retention_fresh.py','real_arm_pick_place.py','real_arms.py']:
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    starts=[]
    for case,t in enumerate(cases):
        env=NativePickPlace('outputs/ease_real_arm_assets_v1',t['robot'],t['xy'],t['target'],.2,True)
        c=GuardedPose(env.n,None);probe=calibrate(env,t['home'],c,41000+case)
        (ROOT/f'probe_{case}.json').write_text(json.dumps(probe))
        assert probe==next(r['probe'] for r in old if r['case']==case and r['method']=='analytical')
        env.model.geom_contype[:]=0;env.model.geom_conaffinity[:]=0;starts.append((env,c))
    results=[]
    for method,model in models.items():
        for case,t in enumerate(cases):
            env,initial=starts[case]
            for zero in [False,True]:
                mode='zero_bias' if zero else 'measured_bias'
                with cloned_state(env),patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged control forbidden')):
                    c=copy_controller(initial,model);trace=[]
                    for s in schedules[case]:
                        a,_=step(env,c,s,zero_bias=zero);trace.append(a)
                    assert env.steps==1732
                assert env.steps==32
                phases=[summarize_phase(phase,[x for x in trace if x['phase']==phase]) for phase in dict.fromkeys(x['phase'] for x in trace)]
                r=dict(method=method,case=case,robot=t['robot'],mode=mode,passes=all(p['passes'] for p in phases),phases=phases,trace=trace)
                (ROOT/'trials'/f'{method}_{case}_{mode}.json').write_text(json.dumps(r))
                if not zero:
                    original=next(x for x in rows if x['method']==method and x['case']==case)
                    assert trace==original['trace'], 'Replay mismatch; actual trace saved'
                results.append(r)
            print(method,case,'measured/zero',results[-2]['passes'],results[-1]['passes'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    assert len(results)==240
    (ROOT/'records.json').write_text(json.dumps(results))
    (ROOT/'audit.json').write_text(json.dumps(dict(trials=240,control_steps=408000,physical_calibration_steps=384,charged_steps_per_trial=1732,exact_source_replays=120,unchanged_weights=True,records_sha256=fingerprint(ROOT/'records.json'),test_evaluated=False),indent=2))
    report()
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


def report():
    rows=json.loads((ROOT/'records.json').read_text());audit=json.loads((ROOT/'audit.json').read_text());protocol=json.loads((ROOT/'protocol.json').read_text())
    source=json.loads((SOURCE/'records.json').read_text())
    assert len(rows)==240 and fingerprint(ROOT/'records.json')==audit['records_sha256']
    assert fingerprint(SOURCE/'records.json')==protocol['source_sha256']
    assert fingerprint(SOURCE/'schedules.json')==protocol['schedules_sha256']
    keys={(r['method'],r['case'],r['mode']) for r in rows};assert len(keys)==240
    for r in rows:
        assert json.loads((ROOT/'trials'/f'{r["method"]}_{r["case"]}_{r["mode"]}.json').read_text())==r
        assert len(r['trace'])==1700 and len(r['phases'])==9
        assert all(summarize_phase(p['phase'],[x for x in r['trace'] if x['phase']==p['phase']])==p for p in r['phases'])
        assert r['passes']==all(p['passes'] for p in r['phases'])
        if r['mode']=='measured_bias':assert r['trace']==next(x['trace'] for x in source if x['method']==r['method'] and x['case']==r['case'])
    for path,sha in protocol['checkpoints'].items():assert fingerprint(path)==sha
    lines=['# E65: continuous full-schedule interventions','',protocol['design'],'',protocol['metric'],'',
        '| Estimator group | Robot | Measured bias | Zero bias | Paired gains | Paired losses |','|---|---|---:|---:|---:|---:|']
    for group in ['analytical','original','uncorrected','corrected']:
        for robot in ['piper','panda']:
            rr=[r for r in rows if (r['method']==group or r['method'].startswith(group+'_')) and r['robot']==robot]
            base=[r for r in rr if r['mode']=='measured_bias'];zero=[r for r in rr if r['mode']=='zero_bias']
            pairs=[(b,next(z for z in zero if z['case']==b['case'] and z['method']==b['method'])) for b in base]
            lines.append(f'| {group} | {robot} | {sum(r["passes"] for r in base)}/{len(base)} | {sum(r["passes"] for r in zero)}/{len(zero)} | {sum(not b["passes"] and z["passes"] for b,z in pairs)} | {sum(b["passes"] and not z["passes"] for b,z in pairs)} |')
    lines+=['','## Per-seed complete schedules','', '| Method | Measured PiPER /6 | Zero PiPER /6 | Measured Panda /6 | Zero Panda /6 |','|---|---:|---:|---:|---:|']
    for method in dict.fromkeys(r['method'] for r in rows):
        values=[sum(r['passes'] for r in rows if r['method']==method and r['robot']==robot and r['mode']==mode) for robot in ['piper','panda'] for mode in ['measured_bias','zero_bias']]
        lines.append('| '+method+' | '+' | '.join(map(str,values))+' |')
    lines+=['','All 240 trial files and phase summaries audited; 120 complete measured-bias replays match E63 exactly. 408,000 control steps, 384 shared physical calibration steps, 32-step calibration cost per trial. Nine checkpoints unchanged. Analytical rows are deduplicated controls, not nine independent seed replicas. No model promotion, training, final-test evaluation or manipulation-success claim.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)


if __name__=='__main__':run()
