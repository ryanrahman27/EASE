"""E110: broader action availability, copied physics only; no policy changes."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from . import early_physics_rollout as prior
from .geometry_filter import GeometryArm,allowed_pair
from .physics_prediction_audit import forbidden
from .benchmark import fingerprint

ROOT=Path('outputs/ease_goal_action_audit_v1')
REFERENCE=Path('outputs/ease_early_physics_v1')
PHASES=('descend','lift','transport','lower','retreat')

def audit_state(env,original,aperture,bias,entry,key):
    case,seed,step=key;q=env.observe()['q'];b=np.clip(bias,-.04,.04);fingers=env.data.qpos[env.n:env.n+2].copy()
    goal=np.r_[env.goal,[0.,0.,.15]]
    vector=lambda d:np.r_[d.site_xpos[env.site],.15*d.site_xmat[env.site].reshape(3,3)[:,2]]
    start=vector(env.data);before=float(np.sum((goal-start)**2));minimum_gain=max(1e-8,.01*before)
    data=mujoco.MjData(env.model);geometry=mujoco.MjData(env.geometry_model);cache={}
    def simulate(action):
        action=np.clip(action,-.04,.04);cache_key=action.tobytes()
        if cache_key in cache:return cache[cache_key]
        mujoco.mj_copyData(data,env.model,env.data);target=np.clip(q+action,-2.5,2.5);data.ctrl[:env.n]=target;data.ctrl[env.n:]=aperture
        static_clearance=min(env.clearance(q+f*(target-q),fingers+f*(aperture-fingers)) for f in (.5,1.))
        static_ok=static_clearance>=min(.002,entry['current_clearance'])-1e-6
        hits=set();clearance=.002;result=dict(action=action.tolist(),static_clearance=static_clearance,static_ok=static_ok)
        for k in range(1,81):
            mujoco.mj_step(env.model,data);hits.update(forbidden(env.model,data,env.gids))
            # Geometry-only copy includes the predicted object/finger state;
            # never integrates the expanded-margin model.
            mujoco.mj_copyData(geometry,env.geometry_model,data);mujoco.mj_forward(env.geometry_model,geometry)
            distances=[float(c.dist) for c in geometry.contact[:geometry.ncon] if not allowed_pair({int(c.geom1),int(c.geom2)},env.gids)]
            clearance=min([clearance,*distances])
            if k%20==0:mujoco.mj_forward(env.model,data)
            if k in (20,80):
                v=vector(data);error=float(np.sum((goal-v)**2));gain=before-error
                result[str(k*2)]=dict(error_sq=error,gain=gain,progress=gain>=minimum_gain,collision=bool(hits),pairs=sorted(hits),minimum_clearance=clearance,margin_2mm=clearance>=.002-1e-6,vector=v.tolist(),object=data.xpos[env.obj].tolist())
        cache[cache_key]=result;return result
    candidates=[]
    def add(name,group,action):candidates.append(dict(name=name,group=group,**simulate(action)))
    add('original','original',original)
    for name,a in [('half',b+.5*(original-b)),('quarter',b+.25*(original-b)),('bias',b),('hold',np.zeros(env.n)),('brake',b-.04*env.data.qvel[:env.n])]:add(name,'narrow',np.clip(a,-.04,.04))
    for amplitude in (.01,.02,.04):
        for j in range(env.n):
            for sign in (-1,1):
                a=b.copy();a[j]+=sign*amplitude;add(f'axis{j}_{sign}_{amplitude}','axis',np.clip(a,-.04,.04))
    rng=np.random.default_rng(11000000+case*100000+seed*10000+step)
    for i in range(64):
        direction=rng.normal(size=env.n);direction/=np.linalg.norm(direction)
        add(f'random{i}','random',np.clip(b+(.02 if i%2==0 else .04)*direction,-.04,.04))
    # Model-based finite-difference response, not analytic task Jacobians or
    # a learned dynamics prediction. Probes are already in the axis bank.
    v0=np.array(simulate(b)['40']['vector']);jac=np.zeros((6,env.n))
    for j in range(env.n):
        plus=b.copy();minus=b.copy();plus[j]+= .01;minus[j]-=.01;plus=np.clip(plus,-.04,.04);minus=np.clip(minus,-.04,.04)
        jac[:,j]=(np.array(simulate(plus)['40']['vector'])-np.array(simulate(minus)['40']['vector']))/(plus[j]-minus[j])
    correction=jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(6),.35*(goal-v0))
    for scale in (.25,.5,1.):add(f'directed{scale}','directed',np.clip(b+scale*correction,-.04,.04))
    banks={}
    for bank in ('narrow','broad'):
        cc=[c for c in candidates if c['group']=='narrow' or (bank=='broad' and c['group'] in ('axis','random','directed'))]
        admissible=[c for c in cc if c['static_ok'] and not c['40']['collision'] and c['40']['progress']]
        strict=[c for c in admissible if c['40']['margin_2mm']]
        best=min(admissible,key=lambda c:(c['40']['error_sq'],-c['40']['minimum_clearance'],c['name'])) if admissible else None
        banks[bank]=dict(candidates=len(cc),admissible40=len(admissible),with_2mm40=len(strict),best40=None if best is None else best['name'],selected_retains_160=bool(best and not best['160']['collision'] and best['160']['progress']),selected_retains_2mm160=bool(best and not best['160']['collision'] and best['160']['progress'] and best['160']['margin_2mm']))
    stem=f'{case}_{seed}_{step}';state_type=mujoco.mjtState.mjSTATE_INTEGRATION;state=np.empty(mujoco.mj_stateSize(env.model,state_type));mujoco.mj_getState(env.model,env.data,state,state_type)
    np.savez_compressed(ROOT/'states'/f'{stem}.npz',integration_state=state,state_type=int(state_type),goal=env.goal,bias=b,aperture=aperture,original=original)
    (ROOT/'states'/f'{stem}.xml').write_text(env.xml)
    return dict(case=case,seed=seed,step=step,body=env.arm.seed,dof=env.n,phase=env.phase,validation=case%3==2,before_error_sq=before,minimum_gain=minimum_gain,banks=banks,candidates=candidates,unique_physics_branches=len(cache))

def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False)
    for folder in ('states','trials','source','videos'):(ROOT/folder).mkdir()
    tasks=json.loads((prior.prior.TASKS/'cases.json').read_text());prior.AXES.update(json.loads((prior.prior.TASKS/'axis_catalog.json').read_text()));assert all(t['arm']['split']=='train' for t in tasks)
    baseline={};inputs={}
    for case in range(12):
        for seed in range(3):
            p=REFERENCE/'trials'/f'{case}_{seed}_None_baseline.json';inputs[str(p)]=fingerprint(p);baseline[(case,seed)]=json.loads(p.read_text())
    selections={}
    for pair,r in baseline.items():
        warnings=[a for a in r['physics_audit'] if a['clean'] and not a['prior_episode_collision'] and a['original_pairs']]
        if warnings:selections[(*pair,warnings[0]['step'])]='warning'
    for dof in (6,7):
        for seed in range(3):
            pair=next(k for k,r in sorted(baseline.items()) if k[1]==seed and r['dof']==dof and r['success'])
            for phase in PHASES:
                a=next(a for a in baseline[pair]['physics_audit'] if a['phase']==phase);selections[(*pair,a['step'])]='successful_control'
    parents,hashes=prior.prior.previous.models();students,hh=prior.prior.e99.load_students();hashes.update(hh)
    spec=dict(experiment='E110',inputs=inputs,checkpoints=hashes,selections=[dict(case=k[0],seed=k[1],step=k[2],kind=v) for k,v in sorted(selections.items())],design='First pre-collision clean warning from each E109 baseline pair with one; plus first step of five motion phases from first successful case in each DOF/seed stratum. All original training bodies; preserve train/held placement labels. Replay each selected baseline completely and exactly; never execute audited alternatives.',candidates='Same five E108 alternatives versus expanded bank including those five, each joint +/- .01/.02/.04 around clipped measured bias,64 deterministic isotropic directions alternating .02/.04 norm around bias, and three finite-difference goal-directed actions with scales .25/.5/1. All increments clipped +/- .04. Directional response fit uses simulated40ms end pose around bias with +/- .01 probes, damping .015 and gain .35. No task Jacobian access. Frozen RNG11000000+case*100000+seed*10000+step. No candidate/horizon tuning.',criteria='Primary40ms: original static admissibility, zero forbidden physical contact, and reduction of squared XYZ plus .15-scaled tool-axis error by at least max(1e-8,1% initial error). Report separate2mm minimum clearance over predicted2ms states using geometry-only expanded model with predicted object/fingers. Choose best40ms progress, tie larger clearance then name. Hold its absolute target to160ms and report continued contact-free progress, without reranking at160ms. This is not closed-loop replanning or full-task success.',limits='Privileged exact physics-state/model finite-difference candidate audit. Outcome/prediction-selected states and successful controls are not independent trial samples or new embodiments. Candidate simulations may contact/move objects; only copies evolve. No training or executed controller change, no reserved final test.',videos='Reuse original E104 baseline success/failure videos only for selected trajectory pairs, with exact hashes and explicit unchanged-policy labels.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);records=[];replays=[];select_fn=GeometryArm.select_action
    for case,seed in sorted({k[:2] for k in selections}):
        seen=[]
        def select(env,action,aperture,bias):
            original,entry=select_fn(env,action,aperture,bias);key=(case,seed,env.steps)
            if key in selections and getattr(env,'phase',None) in PHASES:
                assert key not in seen;seen.append(key)
                r=audit_state(env,original,aperture,bias,entry,key);r['kind']=selections[key]
                (ROOT/'states'/f'{case}_{seed}_{env.steps}.json').write_text(json.dumps(r));records.append({k:v for k,v in r.items() if k!='candidates'})
                print('state',key,r['kind'],r['phase'],'admissible',r['banks']['broad']['admissible40'],flush=True)
            return original,entry
        with patch.object(GeometryArm,'select_action',select):r=prior.evaluate(case,tasks[case],seed,'baseline',parents,students)
        old={k:v for k,v in baseline[(case,seed)].items() if k!='physics_audit'};assert json.loads(json.dumps(r))==old
        assert set(seen)=={k for k in selections if k[:2]==(case,seed)}
        (ROOT/'trials'/f'{case}_{seed}.json').write_text(json.dumps(r));replays.append(dict(case=case,seed=seed,exact=True,success=r['success']))
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(records,indent=2));(ROOT/'verification.json').write_text(json.dumps(dict(exact_replays=len(replays),replays=replays,states=len(records),unique_physics_branches=sum(r['unique_physics_branches'] for r in records),hashes_unchanged=True,no_policy_change=True,test_evaluated=False),indent=2))
    old=json.loads((prior.ROOT/'videos/manifest.json').read_text());videos=[]
    for v in old['videos']:
        if v['mode']!='baseline' or (v['case'],v['seed']) not in {k[:2] for k in selections}:continue
        assert v['exact_replay'] and fingerprint(prior.ROOT/'videos'/v['file'])==v['sha256'];shutil.copy2(prior.ROOT/'videos'/v['file'],ROOT/'videos'/v['file']);videos.append(v)
    (ROOT/'videos/manifest.json').write_text(json.dumps(dict(provenance='Unmodified E104 full-task baseline videos for replayed E110 trajectories. Counterfactual actions are NOT executed in these videos.',videos=videos),indent=2))
    html='<html><body><h1>E110 baseline state-source videos</h1><p>Unchanged verified baseline executions; candidate actions were simulated separately and are not shown as policy improvements.</p>'
    for v in videos:html+=f'<h2>{v["file"]}</h2><video controls width="640" src="{v["file"]}"></video>'
    (ROOT/'videos/index.html').write_text(html+'</body></html>')

if __name__=='__main__':run()
