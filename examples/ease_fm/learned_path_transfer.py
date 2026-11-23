"""E91: frozen learned estimators with disclosed privileged route assistance."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from .novelaxis_path_execution import PathArm, PATHS, PRIOR
from .arm_pick_place import rollout,targets
from .mixed_sim import AXES,Arm
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .real_arm_reach import load_models
from .benchmark import fingerprint

ROOT=Path('outputs/ease_learned_path_transfer_v1')


def route_check(env,end):
    start=env.observe()['q'];fingers=env.data.qpos[env.n:env.n+2].copy()
    before=[env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy()];time=env.data.time
    route=np.array([(1-a)*start+a*end for a in np.linspace(0,1,101)])
    distances=[env.clearance(q,fingers) for q in route]
    for old,new in zip(before,(env.data.qpos,env.data.qvel,env.data.ctrl)):np.testing.assert_array_equal(old,new)
    assert env.data.time==time
    return route,distances


def evaluate(case,t,method,model,mode,record_step=None):
    p=json.loads((PATHS/'cases'/f'{case}.json').read_text());selected=next((a for a in p['attempts'] if a['attempt']==p['selected_attempt']),None)
    available=selected is not None and selected['contact_free'] and selected['margin_2mm']
    env=PathArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();env.route=None;env.path_log=[];env.waypoint=0
    c=GuardedPose(env.n,model);assert calibrate(env,t['home'],c,872000+t['candidate'])==json.loads((PRIOR/f'probe_{case}.json').read_text())
    env.controller=c;env.filter_enabled=True;env.filter_log=[];check={};descent={}
    if record_step is not None:
        original=env.step
        def captured(action,aperture):
            result=original(action,aperture);record_step(env);return result
        env.step=captured
    def phases():
        for name,goal,aperture,budget in targets(env.xy,env.target):
            env.phase=name
            if name=='descend':
                descent.update(qpos=env.data.qpos.copy().tolist(),qvel=env.data.qvel.copy().tolist())
                if available and mode!='baseline':
                    route,distances=route_check(env,np.array(selected['endpoint']));accepted=min(distances)>=.002-1e-6
                    check.update(accepted=accepted,clearances=distances,queries=101)
                    if accepted:
                        env.route=route
                        if mode=='route300':budget=300
            yield name,goal,aperture,budget
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Learned task Jacobian access forbidden')):
        result=rollout(env,t['home'],c,reset=False,sequence=phases())
    used=env.route is not None
    if not used:
        old=json.loads((PRIOR/'trials'/f'{case}_{method}_baseline.json').read_text());assert all(result[k]==old[k] for k in result)
        assert sum(x['queries'] for x in env.filter_log)==old['filter_queries']
    return dict(case=case,body=t['body'],family=t['arm']['family'],dof=env.n,method=method,mode=mode,
        route_available=available,route_used=used,route_check=check,descent_state=descent,last_waypoint=env.waypoint,path_log=env.path_log,
        filter_queries=sum(x['queries'] for x in env.filter_log),**result)


def models():
    loaded={};hashes={}
    for group,folder in [('parent','outputs/ease_robust_training_v1'),('candidate','outputs/ease_estimator_distillation_v1/analytical')]:
        mm,hh=load_models(folder);hashes.update(hh);loaded.update({f'{group}_{s}':mm[f'neural_{s}'] for s in range(3)})
    return loaded,hashes


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    tasks=json.loads((PRIOR/'cases.json').read_text());AXES.update(json.loads((PRIOR/'axis_catalog.json').read_text()));loaded,hashes=models()
    assert hashes==json.loads((PRIOR/'protocol.json').read_text())['checkpoints']
    paths=[PRIOR/'cases.json',PRIOR/'axis_catalog.json',PATHS/'protocol.json',*sorted((PATHS/'cases').glob('*.json')),*sorted((PRIOR/'trials').glob('*_baseline.json'))]
    inputs={str(p):fingerprint(p) for p in paths}
    spec=dict(experiment='E91',inputs=inputs,checkpoints=hashes,test_evaluated=False,
        design='All16 cases x6 frozen learned estimators x baseline/route100/route300 =288 complete tasks. Baseline must exactly reproduce E87 learned geometry-filter baseline. Frozen E89 selected endpoints, E90 waypoint tracker, original calibration/approach/physics/phase rules. No E82 response correction. At learned descent start recheck101 interpolated samples from actual arm posture to frozen endpoint with fingers/object fixed at actual current pose. Accept only sampled2mm margin throughout; otherwise baseline fallback and no budget extension. No search or alternate endpoint selection. Route300 extends only accepted-route descent to300 steps. All other phases remain learned-Jacobian control.',
        attribution='Hybrid learned motion estimator plus privileged offline IK and online scene geometry. Route tracking itself is joint feedback with measured bias, not Transformer actions. Unknown-axis learned transfer scores from E87 remain unchanged; development intervention outcomes cannot establish independent generalization or learning advantage.',
        verification='No task mj_jacSite calls; neural inverse guard retained. Exact E87 baseline and fallback results. Same learned descent state across three modes. Checkpoint/input/source hashes unchanged. Preserve all paired gains/losses and DOF/model breakdowns.',
        gate='Report gains/losses without promotion. Require no lost baseline-successful six-joint case before considering retention established. Original and longer budgets reported separately.',
        videos='Four illustrative executed replays: candidate_0 cases0 and4 in baseline and route300. If these lack either successes or failures, append first missing outcome in case/method/mode order. Label hybrid/privileged planner and budget; compare complete replay with saved result.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('learned_path_transfer.py','novelaxis_path_execution.py','geometry_filter.py','arm_pick_place.py','pick_place_control.py','retention_fresh.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    rows=[];fallbacks=0
    for case,t in enumerate(tasks):
        for method,model in loaded.items():
            state=None
            for mode in ('baseline','route100','route300'):
                r=evaluate(case,t,method,model,mode)
                if state is None:state=r['descent_state']
                assert state==r['descent_state']
                if mode!='baseline' and not r['route_used']:fallbacks+=1
                (ROOT/'trials'/f'{case}_{method}_{mode}.json').write_text(json.dumps(r))
                rows.append({k:v for k,v in r.items() if k not in ('trace','path_log','descent_state','route_check')})
        print('case',case,'complete',flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2));(ROOT/'audit.json').write_text(json.dumps(dict(exact_baselines=96,exact_fallbacks=fallbacks,same_descent_states=True,hashes_unchanged=True,task_jacobians_forbidden=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()
