"""E47: matched pre-grasp excitation with frozen E35 checkpoints."""
import argparse
from contextlib import nullcontext
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
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .native_reliability import rollout


def refresh(env, controller, condition, seed):
    """32 physical steps, no reset. Frozen control changes neither statistics nor bias."""
    assert condition in ('wait','probe','frozen_probe')
    rng=np.random.default_rng(seed);obs=env.observe();anchor=obs['q'].copy();trace=[]
    for step in range(32):
        if step%4==0:
            offset=rng.uniform(-.006,.006,env.n) if condition!='wait' else np.zeros(env.n)
        before=obs
        obs,command,info=env.step(np.clip(anchor+offset-obs['q'],-.04,.04),0.)
        if controller is not None and condition!='frozen_probe':controller.update(before,obs,command)
        trace.append(dict(phase='refresh',position=obs['position'].tolist(),q=obs['q'].tolist(),
            command=command.tolist(),object=env.data.xpos[env.obj].tolist(),
            pads=len(env.contacts()[0]),bad=info['bad']))
        if env.bad:break
    return trace


def run(assets,models,cases,output):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    manifest=verify_assets(assets);loaded,hashes=load_models(models)
    source=Path(cases);tasks=json.loads(source.read_text())
    conditions=['baseline','wait','probe','frozen_probe']
    protocol=dict(experiment='E47',checkpoints=hashes,cases_sha256=fingerprint(source),
        asset_commit=manifest['commit'],conditions=conditions,initial_probe_steps=32,refresh_steps=32,
        refresh_seed=47000,initial_probe_seed=41000,test_evaluated=False,
        design='Original E35 weights. Reuse all twelve E46 placements, now observed development. Baseline exactly replays E46 old models. All intervention arms add the same gated hover 4 cm above grasp, followed by 32 steps. Thus baseline comparison also includes hover; causal refresh comparisons use wait/probe/frozen_probe.',
        excitation='Eight random joint targets uniform +/- .006 rad about measured hover joint pose, four steps each, open gripper. No reset; retain history with normal .98 decay and bias update. Wait holds measured hover joint pose and updates history. Frozen_probe executes identical physical probing without any statistics or bias updates. Stop early on forbidden contact; retain failures before intervention.',
        accounting='All actual simulator steps counted, including initial calibration, hover, refresh, failures and remaining task. Equal maximum refresh budget, not artificially equal total episode durations. No gradient updates or native training data.',
        decision='Diagnostic only. Do not promote if any neural seed/robot regresses versus baseline. Compare probe to both matched controls; retain every outcome.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));shutil.copyfile(source,root/'cases.json')
    (root/'source').mkdir()
    for name in ['local_refresh.py','native_reliability.py','real_arm_pick_place.py','real_arms.py','pick_place_control.py','motion_transformer.py','retention_fresh.py']:
        shutil.copyfile(Path(__file__).with_name(name),root/'source'/name)
    rows=[]
    for condition in conditions:
        for method,model in loaded.items():
            for i,t in enumerate(tasks):
                env=NativePickPlace(assets,t['robot'],t['xy'],t['target'],.2,True)
                controller=GuardedPose(env.n,model)
                guard=nullcontext() if method=='privileged' else patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden'))
                hook=None if condition=='baseline' else lambda e,c:refresh(e,c,condition,47000+i)
                with guard:
                    initial=calibrate(env,t['home'],controller,41000+i)
                    result=rollout(env,None if method=='privileged' else controller,
                        object_release=t['robot']=='panda',pregrasp_hook=hook)
                assert result['steps']==len(initial)+len(result['trace'])
                rows.append(dict(condition=condition,method=method,case=i,robot=t['robot'],probe=initial,**result))
                (root/'records.json').write_text(json.dumps(rows,indent=2))
                print(condition,method,i,result['success'],result['failure'],result['steps'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))
    summarize(root,source.parent/'records.json')


def summarize(root,reference):
    root=Path(root);rows=json.loads((root/'records.json').read_text());old=json.loads(Path(reference).read_text())
    lookup={(r['condition'],r['method'],r['case']):r for r in rows}
    assert len(rows)==len(lookup)==240
    baseline_checks=0;paired_checks=0
    for r in rows:
        if r['condition']!='baseline':continue
        method=r['method'].replace('neural_','old_')
        ref=next(x for x in old if x['method']==method and x['case']==r['case'])
        for key in ['success','failure','trace','probe','steps']:assert r[key]==ref[key],(method,r['case'],key)
        baseline_checks+=1
        pair=[lookup[(c,r['method'],r['case'])] for c in ['wait','probe','frozen_probe']]
        before=lambda x:[t for t in x['trace'] if t['phase'] in ('approach','hover')]
        assert before(pair[0])==before(pair[1])==before(pair[2])
        assert [t for t in pair[1]['trace'] if t['phase']=='refresh']==[t for t in pair[2]['trace'] if t['phase']=='refresh']
        for x in pair:
            assert x['probe']==r['probe']
            n=sum(t['phase']=='refresh' for t in x['trace'])
            assert n==32 or not x['success']
        paired_checks+=1
    summary=[]
    for c in ['baseline','wait','probe','frozen_probe']:
        for robot in ['piper','panda']:
            for method in ['privileged','analytical','neural_0','neural_1','neural_2']:
                subset=[r for r in rows if (r['condition'],r['robot'],r['method'])==(c,robot,method)]
                summary.append(dict(condition=c,robot=robot,method=method,success=sum(r['success'] for r in subset),n=len(subset),
                    collisions=sum(r['collision'] for r in subset),mean_steps=float(np.mean([r['steps'] for r in subset])),
                    refresh_reached=sum(any(t['phase']=='refresh' for t in r['trace']) for r in subset),
                    failures=[dict(case=r['case'],failure=r['failure']) for r in subset if not r['success']]))
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    (root/'audit.json').write_text(json.dumps(dict(exact_baseline_replays=baseline_checks,matched_intervention_groups=paired_checks,step_accounting=True,unchanged_weights=True,test_evaluated=False),indent=2))
    lines=['# E47: local pre-grasp refresh','',
        'Frozen original E35 models on all twelve now-observed E46 development placements. No weight updates. Each intervention adds a shared gated hover 4 cm above grasp and up to 32 physical steps. Probe uses +/- .006 rad targets; frozen probe has identical motion but skips history and bias updates. Wait holds joints and updates history. All initial probes and pre-intervention trajectories match; all 60 historical baseline replays are exact. Total interaction cost includes calibration, hover, probing and failed trials.','',
        '| Condition | Robot | Controller | Success | Collisions | Mean steps | Reached refresh |',
        '|---|---|---|---:|---:|---:|---:|']
    for s in summary:lines.append(f"| {s['condition']} | {s['robot']} | {s['method']} | {s['success']}/{s['n']} | {s['collisions']} | {s['mean_steps']:.1f} | {s['refresh_reached']}/{s['n']} |")
    lines+=['','Baseline lacks the added hover, so comparisons to it combine waypoint and calibration effects. The matched intervention comparisons isolate excitation versus waiting and updating versus freezing the estimator. Freezing also freezes bias adaptation. Small, repeatedly observed tasks are diagnostic; these are not independent generalization estimates. Native mechanics and scripted state-based task sequencing remain unchanged. Per-case failures are retained in summary.json and complete traces in records.json.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','models','cases','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))
