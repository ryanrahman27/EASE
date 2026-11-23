"""E49: exact E48 prefix replays and descent command decomposition."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .benchmark import fingerprint
from .split_refresh import SplitPose,CONDITIONS
from .native_reliability import rollout
from .pick_place_control import calibrate
from .real_arm_pick_place import NativePickPlace
from .real_arm_reach import load_models
from .real_arms import verify_assets


class PrefixComplete(Exception):pass


class Observer:
    def __init__(self):self.trace=[];self.descent=[];self.last_probe=None

    def __call__(self,trace,before,after,command,action,motion,bias,jac,error):
        self.trace.append(trace)
        if trace['phase']!='descend':return
        raw=motion+bias;dq=after['q']-before['q']
        dx=np.r_[after['position']-before['position'],.15*(after['rotation'][:,2]-before['rotation'][:,2])]
        sv=np.linalg.svd(jac,compute_uv=False)
        self.descent.append(dict(motion=motion.tolist(),bias=bias.tolist(),raw=raw.tolist(),
            action=action.tolist(),command=command.tolist(),dq=dq.tolist(),error=error.tolist(),
            motion_norm=float(np.linalg.norm(motion)),bias_norm=float(np.linalg.norm(bias)),
            action_norm=float(np.linalg.norm(action)),command_jump=float(np.linalg.norm(action-self.last_probe)),
            clipped_joints=int(np.sum(np.abs(raw)>.04)),motion_clipped_joints=int(np.sum(np.abs(motion)>.04)),
            bias_motion_dot=float(bias@motion),singular_values=sv.tolist(),
            position_prediction_error=float(np.linalg.norm((jac@dq-dx)[:3])),
            direction_prediction_error=float(np.linalg.norm((jac@dq-dx)[3:])),
            position_error=trace['position_error'],rotation_error=trace['rotation_error']))
        if len(self.descent)==16:raise PrefixComplete()


def run(assets,models,reference,output):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    ref=Path(reference);old=json.loads((ref/'records.json').read_text());tasks=json.loads((ref/'cases.json').read_text())
    prior=json.loads((ref/'protocol.json').read_text());loaded,hashes=load_models(models)
    assert hashes==prior['checkpoints'];manifest=verify_assets(assets)
    assert manifest['commit']==prior['asset_commit']
    protocol=dict(experiment='E49',reference=reference,reference_sha256=fingerprint(ref/'records.json'),
        checkpoints=hashes,cases_sha256=fingerprint(ref/'cases.json'),asset_commit=manifest['commit'],
        scope='192 exact prefix replays: all E48 conditions/controllers/cases. Stop after 16 descent steps (0.64 s), or earlier task termination. Episode outcomes are inherited from E48, not new full-episode evaluations. No controller intervention, tuning or training.',
        diagnostics='Log the actual damped-Jacobian motion term, pre-update bias, raw sum, clipped action, executed command, measured dq, singular values and pose response. Command_jump compares each descent action with the last probe command; only the first value is the handoff discontinuity. Diagnostic only.',test_evaluated=False)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));shutil.copyfile(ref/'cases.json',root/'cases.json');(root/'source').mkdir()
    for name in ['handoff_diagnostic.py','split_refresh.py','native_reliability.py','local_refresh.py','pick_place_control.py','retention_fresh.py','motion_transformer.py','real_arm_pick_place.py','real_arms.py']:
        shutil.copyfile(Path(__file__).with_name(name),root/'source'/name)
    rows=[]
    for expected in old:
        method=expected['method'];case=expected['case'];condition=expected['condition'];t=tasks[case]
        env=NativePickPlace(assets,t['robot'],t['xy'],t['target'],.2,True)
        controller=SplitPose(env.n,loaded[method],condition);observer=Observer()
        def hook(e,c):
            trace=controller.hook(e,c,47000+case);observer.trace.extend(trace)
            observer.last_probe=np.array(trace[-1]['command']);return trace
        with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
            probe=calibrate(env,t['home'],controller,41000+case)
            try:rollout(env,controller,object_release=t['robot']=='panda',pregrasp_hook=hook,step_observer=observer)
            except PrefixComplete:pass
        assert probe==expected['probe'];assert observer.trace==expected['trace'][:len(observer.trace)]
        assert controller.states==expected['states']
        assert env.steps==32+len(observer.trace)
        expected_count=min(16,sum(x['phase']=='descend' for x in expected['trace']))
        assert len(observer.descent)==expected_count
        for d in observer.descent:
            np.testing.assert_array_equal(np.clip(np.array(d['motion'])+d['bias'],-.04,.04),d['action'])
        row=dict(condition=condition,method=method,case=case,robot=t['robot'],
            inherited_success=expected['success'],inherited_failure=expected['failure'],
            prefix_steps=env.steps,exact_prefix=True,descent=observer.descent)
        rows.append(row);(root/f'{condition}_{method}_{case}.json').write_text(json.dumps(row))
        print(condition,method,case,'exact',len(observer.descent),flush=True)
    assert len(rows)==192
    for path,sha in hashes.items():assert fingerprint(path)==sha
    assert fingerprint(ref/'records.json')==protocol['reference_sha256']
    (root/'records.json').write_text(json.dumps(rows,indent=2))
    summary=[]
    for condition in CONDITIONS:
        for robot in ['piper','panda']:
            for method in ['analytical','neural_0','neural_1','neural_2']:
                sub=[r for r in rows if (r['condition'],r['robot'],r['method'])==(condition,robot,method)]
                first=[r['descent'][0] for r in sub if r['descent']];all_steps=[d for r in sub for d in r['descent']]
                summary.append(dict(condition=condition,robot=robot,method=method,n=len(sub),
                    inherited_success=sum(r['inherited_success'] for r in sub),
                    first_motion=float(np.mean([d['motion_norm'] for d in first])),
                    first_bias=float(np.mean([d['bias_norm'] for d in first])),
                    first_jump=float(np.mean([d['command_jump'] for d in first])),
                    first_clipped_cases=sum(d['clipped_joints']>0 for d in first),
                    clipping_steps=sum(d['clipped_joints']>0 for d in all_steps),descent_steps=len(all_steps),
                    mean_position_prediction_error=float(np.mean([d['position_prediction_error'] for d in all_steps])),
                    mean_direction_prediction_error=float(np.mean([d['direction_prediction_error'] for d in all_steps]))))
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    (root/'audit.json').write_text(json.dumps(dict(exact_prefixes=192,action_decomposition=True,refresh_states_match=True,
        unchanged_weights=True,new_full_episode_evaluations=0,test_evaluated=False),indent=2))
    lines=['# E49: probe-to-descent command diagnostic','',protocol['scope'],'',protocol['diagnostics'],'',
        '| Condition | Robot | Method | E48 successes /6 | First motion norm | First bias norm | First command jump | First clipped cases /6 | Clipped descent steps / observed |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summary:lines.append(f"| {s['condition']} | {s['robot']} | {s['method']} | {s['inherited_success']} | {s['first_motion']:.5f} | {s['first_bias']:.5f} | {s['first_jump']:.5f} | {s['first_clipped_cases']} | {s['clipping_steps']}/{s['descent_steps']} |")
    lines+=['','Command norms are in radians; the first jump compares bounded descent action with the last executed probe command. Later descent trajectories diverge across conditions, so aggregate differences are descriptive. Early termination is retained with actual denominators. All 192 prefixes and refresh states reproduce E48 exactly; this is an observational diagnostic, not a successful intervention or 192 new complete task evaluations. No model or controller promotion.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','models','reference','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))
