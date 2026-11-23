"""E48: factorial motion-statistic versus actuator-bias refresh updates."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .benchmark import fingerprint
from .retention_fresh import GuardedPose
from .pick_place_control import calibrate
from .local_refresh import refresh
from .native_reliability import rollout
from .real_arm_pick_place import NativePickPlace
from .real_arm_reach import load_models
from .real_arms import verify_assets

CONDITIONS={'both':(True,True),'motion_only':(True,False),'bias_only':(False,True),'neither':(False,False)}


class SplitPose(GuardedPose):
    def __init__(self,n,model=None,condition='both'):
        super().__init__(n,model)
        self.condition=condition;self.in_refresh=False;self.diagnostics=[];self.states={}

    def state(self):
        return {key:getattr(self,key).tolist() for key in ['gram','cross','bias']}

    def update(self,before,after,command):
        if not self.in_refresh:
            return super().update(before,after,command)
        # Prediction uses only statistics available before this transition.
        # These diagnostics never choose a probe action or enter task control.
        jac,bias=self.predict()
        dq=after['q']-before['q']
        dx=np.r_[after['position']-before['position'],.15*(after['rotation'][:,2]-before['rotation'][:,2])]
        residual=jac@dq-dx
        self.diagnostics.append(dict(position_error=float(np.linalg.norm(residual[:3])),
            direction_error=float(np.linalg.norm(residual[3:])),
            position_motion=float(np.linalg.norm(dx[:3])),direction_motion=float(np.linalg.norm(dx[3:])),
            bias_error=float(np.linalg.norm(command-bias-dq)),
            gram_condition=float(np.linalg.cond(self.gram))))
        saved={key:getattr(self,key).copy() for key in ['gram','cross','bias']}
        super().update(before,after,command)
        motion,actuator=CONDITIONS[self.condition]
        if not motion:self.gram=saved['gram'];self.cross=saved['cross']
        if not actuator:self.bias=saved['bias']

    def hook(self,env,controller,seed):
        assert controller is self
        self.states['before']=self.state();self.in_refresh=True
        try:trace=refresh(env,self,'probe',seed)
        finally:self.in_refresh=False
        self.states['after']=self.state()
        return trace


def run(assets,models,reference,output):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    ref=Path(reference);tasks=json.loads((ref/'cases.json').read_text())
    old=json.loads((ref/'records.json').read_text());prior=json.loads((ref/'protocol.json').read_text())
    loaded,hashes=load_models(models);manifest=verify_assets(assets)
    assert hashes==prior['checkpoints'];assert manifest['commit']==prior['asset_commit']
    assert len(tasks)==12
    protocol=dict(experiment='E48',reference=reference,reference_records_sha256=fingerprint(ref/'records.json'),
        cases_sha256=fingerprint(ref/'cases.json'),checkpoints=hashes,asset_commit=manifest['commit'],
        conditions=CONDITIONS,initial_probe_seed=41000,refresh_seed=47000,refresh_steps=32,test_evaluated=False,
        design='Four update combinations during the identical E47 pre-grasp probe only. Normal updates resume immediately afterward. All twelve observed E46 placements; original E35 models; no training or tuning. Re-run analytical plus three neural seeds for all conditions (192 trials). Privileged E47 references are reused, since they do not consume either estimator state.',
        mechanics='Exact E47 hover, +/- .006 rad eight targets, four steps each; open gripper; no reset after initial calibration. All physical steps and failures counted. Freeze motion means freeze Gram and cross including decay/ridge; freeze bias means retain pre-refresh bias.',
        diagnostics='One-step probe motion residual J(previous statistics) @ measured dq - measured dx, separately XYZ and .15-scaled tool-Z. Bias residual command - previous bias - measured dq. Computed before updating with the transition, diagnostic only. All four conditions share actual probe motions; normal task trajectories may diverge after refresh.',
        decision='Diagnostic factorial decomposition, not a checkpoint or controller promotion. Retain every outcome and compare paired gains/losses. No final test.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));shutil.copyfile(ref/'cases.json',root/'cases.json')
    (root/'source').mkdir()
    for name in ['split_refresh.py','local_refresh.py','native_reliability.py','retention_fresh.py','pick_place_control.py','motion_transformer.py','real_arm_pick_place.py','real_arms.py']:
        shutil.copyfile(Path(__file__).with_name(name),root/'source'/name)
    rows=[]
    for condition in CONDITIONS:
        for method,model in loaded.items():
            if method=='privileged':continue
            for i,t in enumerate(tasks):
                env=NativePickPlace(assets,t['robot'],t['xy'],t['target'],.2,True)
                controller=SplitPose(env.n,model,condition)
                with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
                    probe=calibrate(env,t['home'],controller,41000+i)
                    result=rollout(env,controller,object_release=t['robot']=='panda',
                        pregrasp_hook=lambda e,c:controller.hook(e,c,47000+i))
                assert result['steps']==len(probe)+len(result['trace'])
                if condition in ('both','neither'):
                    prior_condition='probe' if condition=='both' else 'frozen_probe'
                    expected=next(r for r in old if (r['condition'],r['method'],r['case'])==(prior_condition,method,i))
                    for key in ['trace','phases','steps','success','failure','final_object']:assert result[key]==expected[key],(condition,method,i,key)
                    assert probe==expected['probe']
                row=dict(condition=condition,method=method,case=i,robot=t['robot'],probe=probe,
                    diagnostics=controller.diagnostics,states=controller.states,**result)
                rows.append(row)
                # Persist each complete trial separately; avoid rewriting all traces per trial.
                (root/f'{condition}_{method}_{i}.json').write_text(json.dumps(row,separators=(',',':')))
                print(condition,method,i,result['success'],result['failure'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'records.json').write_text(json.dumps(rows,separators=(',',':')))
    report(root)
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


def report(root):
    root=Path(root);p=json.loads((root/'protocol.json').read_text())
    rows=json.loads((root/'records.json').read_text());lookup={(r['condition'],r['method'],r['case']):r for r in rows}
    assert len(rows)==len(lookup)==192
    for path,sha in p['checkpoints'].items():assert fingerprint(path)==sha
    assert fingerprint(root/'cases.json')==p['cases_sha256']
    assert fingerprint(Path(p['reference'])/'records.json')==p['reference_records_sha256']
    old=json.loads((Path(p['reference'])/'records.json').read_text());replays=0
    for method in ['analytical','neural_0','neural_1','neural_2']:
        for case in range(12):
            group={c:lookup[(c,method,case)] for c in CONDITIONS}
            prefix=lambda r:[t for t in r['trace'] if t['phase'] in ('approach','hover','refresh')]
            for r in group.values():
                assert r['probe']==group['both']['probe'];assert prefix(r)==prefix(group['both'])
                assert r['states']['before']==group['both']['states']['before']
                assert r['steps']==32+len(r['trace'])
                assert len(r['diagnostics'])==sum(t['phase']=='refresh' for t in r['trace'])
                for key in ['gram','cross','bias']:
                    active=CONDITIONS[r['condition']][1 if key=='bias' else 0]
                    expected=group['both']['states']['after'][key] if active else r['states']['before'][key]
                    assert r['states']['after'][key]==expected,(method,case,r['condition'],key)
            for c,old_c in [('both','probe'),('neither','frozen_probe')]:
                expected=next(r for r in old if (r['condition'],r['method'],r['case'])==(old_c,method,case))
                for key in ['trace','phases','steps','success','failure','final_object','probe']:assert group[c][key]==expected[key]
                replays+=1
            # Bias cannot affect the conditional motion prediction on matched dq.
            for a,b in [('both','motion_only'),('bias_only','neither')]:
                for x,y in zip(group[a]['diagnostics'],group[b]['diagnostics']):
                    for key in ['position_error','direction_error','gram_condition']:assert x[key]==y[key]
    summary=[];effects=[]
    for condition in CONDITIONS:
        for robot in ['piper','panda']:
            for method in ['analytical','neural_0','neural_1','neural_2']:
                sub=[r for r in rows if (r['condition'],r['robot'],r['method'])==(condition,robot,method)]
                summary.append(dict(condition=condition,robot=robot,method=method,success=sum(r['success'] for r in sub),n=len(sub),
                    collisions=sum(r['collision'] for r in sub),mean_steps=float(np.mean([r['steps'] for r in sub])),
                    **{key:float(np.mean([d[key] for r in sub for d in r['diagnostics']])) for key in ['position_error','direction_error','bias_error']},
                    failures=[dict(case=r['case'],failure=r['failure']) for r in sub if not r['success']]))
    for robot in ['piper','panda']:
        for method in ['analytical','neural_0','neural_1','neural_2']:
            for target,control in [('both','bias_only'),('motion_only','neither'),('both','motion_only'),('bias_only','neither')]:
                sub=[r for r in rows if (r['condition'],r['robot'],r['method'])==(target,robot,method)]
                pairs=[(r,lookup[(control,method,r['case'])]) for r in sub]
                effects.append(dict(robot=robot,method=method,target=target,control=control,
                    gained=[r['case'] for r,c in pairs if r['success'] and not c['success']],
                    lost=[r['case'] for r,c in pairs if c['success'] and not r['success']]))
    (root/'summary.json').write_text(json.dumps(summary,indent=2));(root/'effects.json').write_text(json.dumps(effects,indent=2))
    (root/'audit.json').write_text(json.dumps(dict(records=len(rows),exact_E47_replays=replays,matched_groups=48,
        selective_state_updates=True,matched_motion_diagnostics=True,unchanged_weights=True,step_accounting=True,test_evaluated=False),indent=2))
    (root/'decision.json').write_text(json.dumps(dict(promote=False,reason='Factorial diagnostic on observed development tasks; no independent validation.',test_evaluated=False),indent=2))
    lines=['# E48: separate motion and actuator-bias updates','',p['design'],'',p['mechanics'],'',p['diagnostics'],'',
        '| Updates during probe | Robot | Analytical /6 | Neural seeds 0 / 1 / 2 (each /6) |','|---|---|---:|---|']
    for c in CONDITIONS:
        for robot in ['piper','panda']:
            scores=[next(s['success'] for s in summary if (s['condition'],s['robot'],s['method'])==(c,robot,m)) for m in ['analytical','neural_0','neural_1','neural_2']]
            lines.append(f"| {c} | {robot} | {scores[0]} | {' / '.join(map(str,scores[1:]))} |")
    lines+=['','## Matched probe prediction residuals and total cost','',
        'Errors are mean Euclidean norms per transition. XYZ is in millimetres; direction is the residual in .15-scaled tool-Z coordinates multiplied by 1000, not a physical displacement or angle. Bias residual is in radians. These are conditional motion predictions using measured dq, not complete action-to-pose predictions.','',
        '| Updates | Robot | Method | XYZ error (mm) | Scaled direction error ×1000 | Bias error (rad) | Collisions | Mean total steps |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for s in summary:lines.append(f"| {s['condition']} | {s['robot']} | {s['method']} | {1000*s['position_error']:.4f} | {1000*s['direction_error']:.4f} | {s['bias_error']:.5f} | {s['collisions']} | {s['mean_steps']:.1f} |")
    lines+=['','## Paired successes gained and lost','', '| Robot | Method | Target − control | Gained cases | Lost cases |','|---|---|---|---|---|']
    for e in effects:lines.append(f"| {e['robot']} | {e['method']} | {e['target']} − {e['control']} | {e['gained']} | {e['lost']} |")
    lines+=['','All 96 both/neither controls exactly reproduce E47. All 48 four-condition groups have identical initial calibration and physical trajectories through refresh. Enabled state components match the both-update state; frozen components equal their pre-refresh state. Normal online updates resume after refresh, so downstream differences include their subsequent closed-loop effects. The intervention tests the finite refresh interval, not permanently disabling bias or identification.','',
        'No promotion. Six placements per robot are repeatedly observed development data, and seeds share tasks. Preserve per-case failures; early failure can shorten mean cost. Privileged references and no-hover baselines are retained in E47, not rerun here. No simulator Jacobian enters learned/analytical control. No checkpoint training or final-test evaluation.']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines[8:19]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','models','reference','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))
