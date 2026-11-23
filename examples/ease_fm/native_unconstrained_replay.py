"""E62 companion: same recorded prefixes without any geom contact forces."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .benchmark import fingerprint
from .native_contact_isolation import replay
from .real_arm_pick_place import NativePickPlace
from .real_arm_reach import load_models
from .real_arms import verify_assets
from .retention_fresh import GuardedPose
from .pick_place_control import calibrate


def run(assets,parents,training,reference,source,output):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False);src=Path(source)
    p=json.loads((src/'protocol.json').read_text());verify_assets(assets)
    tasks=json.loads((Path(reference)/'cases.json').read_text());assert fingerprint(Path(reference)/'cases.json')==p['cases_sha256']
    models={'analytical':None};hashes={}
    for group,folder in [('original',parents),('uncorrected',str(Path(training)/'prediction')),('corrected',str(Path(training)/'control'))]:
        mm,hh=load_models(folder);hashes.update(hh)
        for s in range(3):models[f'{group}_{s}']=mm[f'neural_{s}']
    assert hashes==p['checkpoints']
    protocol=dict(experiment='E62 companion',source=source,checkpoints=hashes,
        intervention='Disable contype/conaffinity on every geom only after unchanged calibration. Keep masses, gains, gripper apertures, gravity, joint limits, original finite reference schedule and all controller settings. This artificial unconstrained replay is diagnostic, never a deployable success metric. Object/table/self geom contact forces removed together.',test_evaluated=False)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'source').mkdir();(root/'trials').mkdir()
    for name in ['native_unconstrained_replay.py','native_contact_isolation.py','native_reliability.py']:
        shutil.copy2(Path(__file__).with_name(name),root/'source'/name)
    results=[]
    def completed_trials():
        for method in models:
            for case in range(len(tasks)):
                path=src/'trials'/f'{method}_{case}.json'
                while True:
                    try:
                        yield json.loads(path.read_text());break
                    except (FileNotFoundError,json.JSONDecodeError):time.sleep(.5)
    for r in completed_trials():
        t=tasks[r['case']];env=NativePickPlace(assets,t['robot'],t['xy'],t['target'],.2,True);c=GuardedPose(env.n,models[r['method']])
        with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged control forbidden')):
            calibrate(env,t['home'],c,41000+r['case'])
            env.model.geom_contype[:]=0;env.model.geom_conaffinity[:]=0
            trace=replay(env,c,r['schedule'])
        assert len(trace)==r['schedule_steps'] and not any(x['bad'] for x in trace)
        row=dict(method=r['method'],group=r['group'],robot=r['robot'],case=r['case'],seed=r['seed'],original_success=r['original_success'],original_failure=r['original_failure'],
            tracks=bool(trace[-1]['tracking_error']<.006 and trace[-1]['direction_error']<.08),
            final_phase=trace[-1]['phase'],final_position_error=trace[-1]['tracking_error'],final_direction_error=trace[-1]['direction_error'],
            tracking_rmse=float(np.sqrt(np.mean([x['tracking_error']**2 for x in trace]))),
            original_tracking_rmse=float(np.sqrt(np.mean([x['tracking_error']**2 for x in r['original']]))),trace=trace)
        results.append(row);(root/'trials'/f"{r['method']}_{r['case']}.json").write_text(json.dumps(row))
        print(r['method'],r['case'],'unconstrained_tracks',row['tracks'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    while not (src/'status.json').exists():time.sleep(.5)
    assert json.loads((src/'status.json').read_text())['status']=='complete'
    protocol['source_sha256']=fingerprint(src/'records.json')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (root/'records.json').write_text(json.dumps(results));(root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))
    lines=['# E62 companion: separate remaining contact forces from pose tracking','',protocol['intervention'],'',
        '| Group | Robot | Prefixes | Original task successes | Unconstrained endpoints track | Original failures whose endpoints track unconstrained | Original RMS XYZ | Unconstrained RMS XYZ |','|---|---|---:|---:|---:|---:|---:|---:|']
    for group in ['analytical','original','uncorrected','corrected']:
        for robot in ['piper','panda']:
            rr=[r for r in results if r['group']==group and r['robot']==robot]
            lines.append(f'| {group} | {robot} | {len(rr)} | {sum(r["original_success"] for r in rr)} | {sum(r["tracks"] for r in rr)} | {sum(r["tracks"] and not r["original_success"] for r in rr)} | '+ ' | '.join(f"{np.sqrt(np.mean([r[k]**2 for r in rr])):.6f}" for k in ['original_tracking_rmse','tracking_rmse'])+' |')
    failures=[{k:v for k,v in r.items() if k!='trace'} for r in results if not r['tracks']]
    lines+=['','Endpoint tracking uses the original 6 mm/0.08 tool-Z tolerances for a single endpoint, not an eight-step hold or full task completion. Prefix lengths come from original outcomes, so groups have different tasks/horizons. No manipulation success or contact-safe control is inferred.', '',f'Unconstrained endpoints failing tolerance: {len(failures)}; detailed records in failures.json. Frozen weights, no training, no final-test evaluation.']
    (root/'failures.json').write_text(json.dumps(failures,indent=2));(root/'report.md').write_text('\n'.join(lines)+'\n')
    (root/'audit.json').write_text(json.dumps(dict(records=len(results),full_prefix_replays=len(results),unchanged_weights=True,records_sha256=fingerprint(root/'records.json'),test_evaluated=False),indent=2))
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','parents','training','reference','source','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))
