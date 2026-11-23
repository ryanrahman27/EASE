"""E109: pre-first-collision physics selection across five motion phases."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from . import early_physics_rollout as prior
from .geometry_filter import GeometryArm
from .arm_pick_place import PickPlace
from .physics_prediction_audit import forbidden
from .benchmark import fingerprint

ROOT=Path('outputs/ease_early_physics_v1')
MODES=('baseline','physics')
PHASES=('descend','lift','transport','lower','retreat')

def evaluate(case,t,seed,mode,parents,students,record_step=None,affected_phase=None):
    assert mode in MODES
    select_fn=GeometryArm.select_action;step_fn=PickPlace.step;audit=[];pending={};scratch={}
    def predict(env,action,aperture):
        if 'data' not in scratch:scratch['data']=mujoco.MjData(env.model)
        d=scratch['data'];mujoco.mj_copyData(d,env.model,env.data)
        d.ctrl[:env.n]=np.clip(env.observe()['q']+np.clip(action,-.04,.04),-2.5,2.5);d.ctrl[env.n:]=aperture;hits=set()
        for _ in range(20):mujoco.mj_step(env.model,d);hits.update(forbidden(env.model,d,env.gids))
        mujoco.mj_forward(env.model,d)
        error=np.r_[env.goal-d.site_xpos[env.site],.15*(np.array([0.,0.,1.])-d.site_xmat[env.site].reshape(3,3)[:,2])]
        return dict(pairs=sorted(hits),qpos=d.qpos.copy(),qvel=d.qvel.copy(),error_sq=float(error@error))
    def select(env,action,aperture,bias):
        original,entry=select_fn(env,action,aperture,bias)
        if getattr(env,'phase',None) not in PHASES:return original,entry
        pred=predict(env,original,aperture);chosen=original;chosen_pred=pred;name='original';current=forbidden(env.model,env.data,env.gids);options=[]
        if mode=='physics' and pred['pairs'] and not current and not env.bad:
            q=env.observe()['q'];b=np.clip(bias,-.04,.04);fingers=env.data.qpos[env.n:env.n+2].copy()
            alternatives=[('half',np.clip(b+.5*(original-b),-.04,.04)),('quarter',np.clip(b+.25*(original-b),-.04,.04)),('bias',b),('hold',np.zeros(env.n)),('brake',np.clip(b-.04*env.data.qvel[:env.n],-.04,.04))]
            safe=[]
            for index,(label,a) in enumerate(alternatives):
                target=np.clip(q+a,-2.5,2.5)
                clearance=min(env.clearance(q+f*(target-q),fingers+f*(aperture-fingers)) for f in (.5,1.))
                static_ok=clearance>=min(.002,entry['current_clearance'])-1e-6
                candidate=predict(env,a,aperture)
                distance=float(np.linalg.norm(target-np.clip(q+original,-2.5,2.5)))
                options.append(dict(name=label,static_ok=static_ok,clearance=clearance,pairs=candidate['pairs'],distance=distance))
                if static_ok and not candidate['pairs']:safe.append((distance,index,label,a,candidate))
            if safe:
                _,_,name,chosen,chosen_pred=min(safe,key=lambda x:(x[0],x[1]))
        obs=env.observe();error=np.r_[env.goal-obs['position'],.15*(np.array([0.,0.,1.])-obs['rotation'][:,2])]
        row=dict(phase=env.phase,before_error_sq=float(error@error),original_after_error_sq=pred['error_sq'],selected_after_error_sq=chosen_pred['error_sq'],step=env.steps,clean=not current,prior_episode_collision=bool(env.bad),original_pairs=pred['pairs'],chosen_pairs=chosen_pred['pairs'],choice=name,changed=not np.array_equal(original,chosen),alternatives=options,original_action=original.tolist(),selected_action=chosen.tolist(),static_original=entry)
        audit.append(row);pending.update(prediction=chosen_pred,row=row)
        # Keep original static-filter metadata explicitly labelled in audit.
        return chosen,entry
    def step(env,action,aperture):
        result=step_fn(env,action,aperture)
        if getattr(env,'phase',None) in PHASES:
            pred=pending['prediction'];actual=sorted({tuple(sorted(p)) for p in result[2]['bad']});assert actual==pred['pairs']
            np.testing.assert_array_equal(env.data.qpos,pred['qpos']);np.testing.assert_array_equal(env.data.qvel,pred['qvel'])
            pending['row']['actual_pairs']=actual
        return result
    with patch.object(GeometryArm,'select_action',select),patch.object(PickPlace,'step',step):
        r=prior.evaluate(case,t,seed,mode,parents,students,record_step,affected_phase=None)
    old=json.loads((prior.ROOT/'trials'/f'{case}_{seed}_None_baseline.json').read_text())
    if mode=='baseline':assert json.loads(json.dumps(r))==old
    prefix=lambda x:[v for v in x['trace'] if v['phase']=='approach']
    assert prefix(r)==prefix(old)
    r['physics_audit']=audit
    return r

def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False)
    for f in ('trials','source'):(ROOT/f).mkdir()
    tasks=json.loads((prior.prior.TASKS/'cases.json').read_text());prior.AXES.update(json.loads((prior.prior.TASKS/'axis_catalog.json').read_text()))
    assert len(tasks)==12 and all(t['arm']['split']=='train' for t in tasks)
    parents,hashes=prior.prior.previous.models();students,hh=prior.prior.e99.load_students();hashes.update(hh)
    refs=[prior.ROOT/'trials'/f'{c}_{s}_None_baseline.json' for c in range(12) for s in range(3)]
    inputs={str(p):fingerprint(p) for p in [prior.prior.TASKS/'cases.json',prior.prior.TASKS/'axis_catalog.json',Path('outputs/ease_physics_prediction_v1/summary.json'),*refs]}
    spec=dict(experiment='E109',inputs=inputs,checkpoints=hashes,
        design='All36 original training-body pairs x baseline/physics=72 full tasks. Same24 train/12 held-placement split. Same E108 five alternatives,40ms horizon, static eligibility, nearest original target ranking, unchanged budgets/weights/apertures; no route planner. Extend intervention to descend/lift/transport/lower/retreat, only before ANY earlier forbidden contact. Calibration and approach remain exactly unchanged. Probe, close, release and settle unchanged. Altered descent can change subsequent history; this is not a fixed post-probe prefix.',
        rule='Preserve original static-selected command if prediction contact-free, current contact exists, or episode previously collided. Otherwise test E108 half/quarter/bias/hold/brake candidates; require same static threshold and zero predicted forbidden contact, choose nearest original joint target with fixed tie order, preserve original if none. No new progress objective in selection: isolate extending phase coverage and disabling post-collision interventions.',
        progress='Log phase completion, phase steps, grasp/lift, final object errors; for changed actions record same-state original versus selected40ms squared XYZ plus .15-scaled tool-axis error. Reduced one-step error is a diagnostic, not task success. Report collision-free task failures and stalled phases separately; preserve all paired gains/losses.',
        gate='Strict training success increase, preserve every baseline success on both train and held placements, no collision increase in either split, no training seed decline. No automatic transfer evaluation.',
        verification='Every selected prediction matches executed qpos/qvel/contact union;36 full baselines reproduce E104;72 approach prefixes identical. Hash frozen inputs and checkpoints.',
        videos='Case0 seed0 both modes, all paired task gains/losses, earliest changed-action pair and missing outcome category; additionally first collision-free physics failure with its baseline if present. Full-result replay verification.',
        limits='Exact simulator-state/model access remains privileged. Same four training bodies; held placements are not unseen embodiments. No E105 outcomes or reserved final test. New actions may change estimator histories and later phases. No horizon/candidate tuning after outcomes.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);shutil.copy2(Path(prior.__file__),ROOT/'source'/Path(prior.__file__).name);rows=[]
    for case,t in enumerate(tasks):
        for seed in range(3):
            for mode in MODES:
                r=evaluate(case,t,seed,mode,parents,students)
                (ROOT/'trials'/f'{case}_{seed}_None_{mode}.json').write_text(json.dumps(r));a=r['physics_audit']
                rows.append(dict(**{k:v for k,v in r.items() if k not in ('trace','contacts','calibration','physics_audit')},changed_actions=sum(x['changed'] for x in a),eligible_warnings=sum(x['clean'] and not x['prior_episode_collision'] and bool(x['original_pairs']) for x in a),no_safe_alternative=sum(bool(x['alternatives']) and x['choice']=='original' for x in a),changed_after_prior_collision=sum(x['changed'] and x['prior_episode_collision'] for x in a),prediction_count=sum(1+len(x['alternatives']) for x in a),executed_predictions=len(a)))
        (ROOT/'records.json').write_text(json.dumps(rows,indent=2));print('case',case,'complete',len(rows),'rollouts',flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'verification.json').write_text(json.dumps(dict(exact_baselines=36,exact_pre_descent_prefixes=72,exact_selected_predictions=sum(r['executed_predictions'] for r in rows),hashes_unchanged=True,no_test_evaluated=True),indent=2))

if __name__=='__main__':run()
