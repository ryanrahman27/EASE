"""E105: frozen retreat controller on prospectively sampled bodies/placements."""
import argparse
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from . import task_retention_training as retained
from .local_probe import ProbeArm, probe_rollout
from .local_history import LocalPose
from .mixed_sim import AXES, Arm
from .arm_pick_place import targets, ik, rollout
from .geometry_filter import GeometryArm
from .pick_place_control import calibrate
from .history_excitation import excitation
from .learned_path_transfer import route_check
from .novelaxis_paths import solve_endpoint, sample_path
from .novelaxis_oracle import ExactPose
from .single_phase import ramp_action
from .benchmark import fingerprint

ROOT=Path('outputs/ease_fresh_retreat_v1')
OLD=Path('outputs/ease_direction_novelaxis_v1')
MODES=('baseline','budget','gentle')

def evaluate(case,t,seed,mode,parents,students,record_step=None,path=None):
    p=json.loads((ROOT/'paths'/f'{case}.json').read_text()) if path is None else path
    selected=next((a for a in p['attempts'] if a['attempt']==p['selected_attempt']),None)
    available=selected is not None and selected['contact_free'] and selected['margin_2mm']
    env=ProbeArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry()
    env.route=None;env.path_log=[];env.waypoint=0;env.probe_step=0
    class Controller(LocalPose):
        switched=False
        def predict(self):
            if getattr(env,'phase',None)=='close':self.switched=env.route is not None
            saved=self.model
            if self.switched:self.model=students[('retained',seed)]
            try:return super().predict()
            finally:self.model=saved
    c=Controller(env.n,parents[f'candidate_{seed}'])
    calibration=calibrate(env,t['home'],c,t.get('calibration_seed',872000+t['candidate']))
    env.controller=c;env.filter_enabled=True;env.filter_log=[];handoff={};measurements={};counter={'phase':None,'index':0}
    original=env.step
    def step(action,aperture):
        if counter['phase']!=env.phase:counter.update(phase=env.phase,index=0)
        if mode=='gentle' and env.phase=='retreat' and c.switched:action=ramp_action(action,c.bias,counter['index'])
        counter['index']+=1
        result=original(action,aperture)
        if record_step is not None:record_step(env)
        return result
    env.step=step
    def phases():
        for name,goal,aperture,budget in targets(env.xy,env.target):
            env.phase=name
            if name=='descend' and available:
                route,distances=route_check(env,np.array(selected['endpoint']))
                if min(distances)>=.002-1e-6:env.route=route;budget=300
            if name=='close':
                handoff.update(qpos=env.data.qpos.tolist(),qvel=env.data.qvel.tolist(),gram=c.gram.tolist(),cross=c.cross.tolist(),bias=c.bias.tolist())
                if env.route is not None:
                    c.local=True;c.rebuild();measurements['before']=excitation(c.gram)
                    env.anchor=env.observe()['q'].copy();env.phase='probe'
                    yield 'probe',env.observe()['position'].copy(),0.,32
                    assert env.probe_step==32
                    measurements['after']=excitation(c.gram);env.phase=name
            if mode!='baseline' and name=='retreat' and env.route is not None:budget+=20
            yield name,goal,aperture,budget
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('No task Jacobians in learned execution')):
        result=probe_rollout(env,t['home'],c,reset=False,sequence=phases())
    return dict(case=case,seed=seed,mode=mode,affected_phase=None if mode=='baseline' else 'retreat',body=t['body'],dof=env.n,route_used=env.route is not None,calibration=calibration,handoff=handoff,excitation=measurements,probe_steps=env.probe_step,**result)

def validate(parents,students):
    tasks=json.loads((OLD/'cases.json').read_text());AXES.update(json.loads((OLD/'axis_catalog.json').read_text()));checks=[]
    # Six-joint recovery, seven-joint recovery, and unchanged no-route fallback.
    for case,seed in ((2,1),(10,0),(5,0)):
        path=json.loads((Path('outputs/ease_novelaxis_paths_v1/cases')/f'{case}.json').read_text())
        for mode in MODES:
            r=evaluate(case,tasks[case],seed,mode,parents,students,path=path)
            phase=None if mode=='baseline' else 'retreat'
            old=json.loads((Path('outputs/ease_single_phase_v1/transfer/trials')/f'{case}_{seed}_{phase}_{mode}.json').read_text())
            for key in ('trace','phases','success','collision','grasped','lifted','steps','final_object','handoff','excitation','probe_steps','route_used'):assert r[key]==old[key],(case,mode,key)
            checks.append(dict(case=case,seed=seed,mode=mode,exact=True))
    (ROOT/'compatibility.json').write_text(json.dumps(checks,indent=2))

def prepare():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False)
    for name in ('source','paths','trials'):(ROOT/name).mkdir()
    tasks=json.loads((OLD/'cases.json').read_text());catalog=json.loads((OLD/'axis_catalog.json').read_text());AXES.update(catalog)
    candidates=[]
    for body in range(8):
        original=tasks[2*body];arm=dict(original['arm']);rng=np.random.default_rng(1050100+body)
        arm.update(seed=1050100+body,split='prospective_replication')
        for name in ('lengths','masses'):arm[name]=(np.asarray(arm[name])*rng.uniform(.9,1.1,len(arm[name]))).tolist()
        for name in ('strength','damping'):arm[name]=float(arm[name]*rng.uniform(.9,1.1))
        for xy in ([.34,-.015],[.315,.04]):
            i=len(candidates);candidates.append(dict(candidate=i,body=body,arm=arm,xy=xy,target=[xy[0],xy[1]+.06],ik_start=original['home'],calibration_seed=1052000+i))
    parents,hashes=retained.previous.models();students,hh=retained.load_students();hashes.update(hh)
    inputs={str(p):fingerprint(p) for p in (OLD/'cases.json',OLD/'axis_catalog.json',Path('outputs/ease_single_phase_v1/transfer/summary.json'))}
    spec=dict(experiment='E105',candidates=candidates,checkpoints=hashes,inputs=inputs,design='Prospective body/placement replication: eight new bodies (two per existing E87 axis order), two new fixed placements each; all three frozen checkpoint seeds, baseline/budget/gentle retreat. Same axis orders as development; NOT unseen-axis replication. Body parameters perturbed independently +/-10% from each old body; seeds1050100..1050107. No replacement or tuning after screening or outcomes.',screening='Nine endpoint IK poses,10 attempts, RNG1051000+candidate; save every rejection. Accepted home is first endpoint. Same fixed oracle-approach/20-start static endpoint planner as E89, seed1053000+candidate;101 path samples,2mm margin. Save every route rejection; execute no-route fallback.',control='Frozen E99 retained after E66 parent and32-step probe on admitted routes. Only gentle retreat uses E10420-step smoothstep around bias and+20max steps; budget only+20. No other control changes. Privileged geometric filter and route planning remain.',gate='Strict improvement over both controls, no lost reference success, nonincreasing collisions and no per-seed decline. Report all candidates including IK rejection coverage, all accepted tasks including no-route failures, DOF/seed/body breakdowns. Bootstrap paired success-rate difference by body with10000 fixed-seed resamples; descriptive interval, not independent placements/seeds.',videos='First accepted case seed0 all modes; every gentle gain/loss versus either control with all modes; append missing success/failure category. Outcome-selected illustrations, exact serialized replay/frame/hash checks.',scope='New prospective replication cohort, not the pre-existing reserved final test. No training, checkpoint selection, axis selection or policy-success screening. Same procedural generator and known axis orders limit scope.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));(ROOT/'axis_catalog.json').write_text(json.dumps(catalog,indent=2))
    for p in Path(__file__).parent.glob('*.py'):shutil.copy2(p,ROOT/'source'/p.name)
    validate(parents,students);print('9 E104 compatibility replays exact',flush=True)
    accepted=[];screen=[]
    for t in candidates:
        env=GeometryArm(Arm(**t['arm']),t['xy'],t['target']);q=t['ik_start'];poses=[];rng=np.random.default_rng(1051000+t['candidate'])
        for phase,goal,_,_ in targets(env.xy,env.target):
            q=ik(env,goal,rng,start=q,attempts=10)
            if q is None:break
            poses.append(q.tolist())
        ok=len(poses)==9;screen.append(dict(candidate=t['candidate'],body=t['body'],accepted=ok,endpoints=len(poses),failed_phase=None if ok else phase))
        if ok:accepted.append(dict(**t,home=poses[0]))
        print('screen',t['candidate'],ok,flush=True)
    (ROOT/'screening.json').write_text(json.dumps(screen,indent=2));(ROOT/'cases.json').write_text(json.dumps(accepted,indent=2))
    for case,t in enumerate(accepted):
        env=GeometryArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();c=ExactPose(env);calibrate(env,t['home'],c,t['calibration_seed']);env.controller=c;env.filter_enabled=True;env.filter_log=[]
        rollout(env,t['home'],c,reset=False,sequence=[next(iter(targets(env.xy,env.target)))])
        saved=dict(qpos=env.data.qpos.tolist(),qvel=env.data.qvel.tolist());start=env.observe()['q'];goal=list(targets(env.xy,env.target))[1][1]
        search=GeometryArm(Arm(**t['arm']),t['xy'],t['target']);search.prepare_geometry();rng=np.random.default_rng(1053000+t['candidate']);attempts=[]
        for attempt in range(20):
            search.data.qpos[:]=saved['qpos'];search.data.qvel[:]=0;mujoco.mj_forward(search.model,search.data)
            q=solve_endpoint(search,goal,start if attempt==0 else rng.uniform(-1.8,1.8,env.n));a=dict(attempt=attempt,endpoint_found=q is not None)
            if q is not None:a.update(endpoint=q.tolist(),distance=float(np.linalg.norm(q-start)),**sample_path(search,start,q))
            attempts.append(a)
        valid=[a for a in attempts if a['endpoint_found']];best=max(valid,key=lambda a:(a['minimum_clearance'],-a['distance'],-a['attempt'])) if valid else None
        (ROOT/'paths'/f'{case}.json').write_text(json.dumps(dict(case=case,state=saved,attempts=attempts,selected_attempt=None if best is None else best['attempt'])))
        print('planned',case,flush=True)
    (ROOT/'status.json').write_text(json.dumps(dict(status='prepared')))

def run():
    torch.set_num_threads(4);assert json.loads((ROOT/'status.json').read_text())['status']=='prepared'
    spec=json.loads((ROOT/'protocol.json').read_text());AXES.update(json.loads((ROOT/'axis_catalog.json').read_text()));tasks=json.loads((ROOT/'cases.json').read_text());parents,_=retained.previous.models();students,_=retained.load_students();rows=[]
    for case,t in enumerate(tasks):
        for seed in range(3):
            baseline=None
            for mode in MODES:
                r=evaluate(case,t,seed,mode,parents,students)
                if baseline is None:baseline=r
                else:
                    prefix=lambda x:[v for v in x['trace'] if v['phase'] not in ('retreat','settle')]
                    assert prefix(r)==prefix(baseline) and r['handoff']==baseline['handoff'] and r['calibration']==baseline['calibration']
                    if not r['route_used']:
                        assert all(r[k]==baseline[k] for k in r if k not in ('mode','affected_phase'))
                (ROOT/'trials'/f'{case}_{seed}_{r["affected_phase"]}_{mode}.json').write_text(json.dumps(r));rows.append({k:v for k,v in r.items() if k not in ('trace','handoff','calibration')})
        (ROOT/'records.json').write_text(json.dumps(rows,indent=2));print('case',case,'complete',len(rows),'rollouts',flush=True)
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);a=p.parse_args();globals()[a.stage]()
