"""E108: fixed 40ms physics action selection on training bodies only."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from . import single_phase as prior
from .geometry_filter import GeometryArm
from .arm_pick_place import PickPlace
from .physics_prediction_audit import forbidden
from .benchmark import fingerprint

ROOT=Path('outputs/ease_physics_action_v1')
MODES=('baseline','physics')

def evaluate(case,t,seed,mode,parents,students,record_step=None,affected_phase=None):
    assert mode in MODES
    select_fn=GeometryArm.select_action;step_fn=PickPlace.step;audit=[];pending={};scratch={}
    def predict(env,action,aperture):
        if 'data' not in scratch:scratch['data']=mujoco.MjData(env.model)
        d=scratch['data'];mujoco.mj_copyData(d,env.model,env.data)
        d.ctrl[:env.n]=np.clip(env.observe()['q']+np.clip(action,-.04,.04),-2.5,2.5);d.ctrl[env.n:]=aperture;hits=set()
        for _ in range(20):mujoco.mj_step(env.model,d);hits.update(forbidden(env.model,d,env.gids))
        mujoco.mj_forward(env.model,d)
        return dict(pairs=sorted(hits),qpos=d.qpos.copy(),qvel=d.qvel.copy())
    def select(env,action,aperture,bias):
        original,entry=select_fn(env,action,aperture,bias)
        if getattr(env,'phase',None)!='retreat':return original,entry
        pred=predict(env,original,aperture);chosen=original;chosen_pred=pred;name='original';current=forbidden(env.model,env.data,env.gids);options=[]
        if mode=='physics' and pred['pairs'] and not current:
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
        row=dict(step=env.steps,clean=not current,prior_episode_collision=bool(env.bad),original_pairs=pred['pairs'],chosen_pairs=chosen_pred['pairs'],choice=name,changed=not np.array_equal(original,chosen),alternatives=options,original_action=original.tolist(),selected_action=chosen.tolist(),static_original=entry)
        audit.append(row);pending.update(prediction=chosen_pred,row=row)
        # Keep original static-filter metadata explicitly labelled in audit.
        return chosen,entry
    def step(env,action,aperture):
        result=step_fn(env,action,aperture)
        if getattr(env,'phase',None)=='retreat':
            pred=pending['prediction'];actual=sorted({tuple(sorted(p)) for p in result[2]['bad']});assert actual==pred['pairs']
            np.testing.assert_array_equal(env.data.qpos,pred['qpos']);np.testing.assert_array_equal(env.data.qvel,pred['qvel'])
            pending['row']['actual_pairs']=actual
        return result
    with patch.object(GeometryArm,'select_action',select),patch.object(PickPlace,'step',step):
        r=prior.evaluate(case,t,seed,mode,parents,students,record_step,affected_phase=None)
    old=json.loads((prior.ROOT/'trials'/f'{case}_{seed}_None_baseline.json').read_text())
    if mode=='baseline':assert json.loads(json.dumps(r))==old
    prefix=lambda x:[v for v in x['trace'] if v['phase'] not in ('retreat','settle')]
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
    spec=dict(experiment='E108',inputs=inputs,checkpoints=hashes,design='All36 original training-body task/seed pairs x baseline/physics=72 full tasks. Original E99 retained weights, E66 prefix/probe, budgets and success criteria. Retreat only. Same24 train/12 held-placement split; no route planner.',rule='Predict original static-filter-selected command in copied exact physical state for40ms. Preserve it bit-exactly if contact-free or current state already has forbidden contact. Otherwise evaluate five alternatives: measured-bias blends .5/.25/0, zero joint increment (hold target at current q), and clip(bias-.04*qvel) (brake heuristic). Every alternative must pass original midpoint/end nonworsening static-clearance criterion AND predict zero forbidden contacts. Choose smallest executed joint-target Euclidean deviation from original command; tie by listed order. If none qualifies preserve original. All alternatives fully evaluated, no early stopping. Preserve aperture, no extra control time. No ramp or weight change.',gate='Strict training success improvement, preserve every reference success in train and held placements, nonincreasing collision totals in each split, no training seed count decline. Report full paired gains/losses, DOF/seed/split totals, changed actions, no-safe-alternative triggers and previously collided trajectories. No automatic transfer or promotion based only on fewer contacts.',verification='Every selected40ms prediction must exactly match executed qpos/qvel and contact union. Every baseline full result must equal E104; both modes retain exact pre-retreat traces. Hash inputs/checkpoints.',videos='Case0 seed0 and all task-success-discordant pairs in both modes; add earliest changed-action pair even if task outcome unchanged, and any missing success/failure category. Full-result replay verified.',limits='Privileged exact-model action selector, not learned dynamics or proof of safety. Small candidate set and40ms horizon may stall progress or have no safe option. Same-body held placements are not unseen embodiments. No E105 data or reserved final test.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);rows=[]
    for case,t in enumerate(tasks):
        for seed in range(3):
            for mode in MODES:
                r=evaluate(case,t,seed,mode,parents,students)
                (ROOT/'trials'/f'{case}_{seed}_None_{mode}.json').write_text(json.dumps(r));a=r['physics_audit']
                rows.append(dict(**{k:v for k,v in r.items() if k not in ('trace','contacts','calibration','physics_audit')},changed_actions=sum(x['changed'] for x in a),eligible_warnings=sum(x['clean'] and bool(x['original_pairs']) for x in a),no_safe_alternative=sum(bool(x['alternatives']) and x['choice']=='original' for x in a),changed_after_prior_collision=sum(x['changed'] and x['prior_episode_collision'] for x in a),prediction_count=sum(1+len(x['alternatives']) for x in a),executed_predictions=len(a)))
        (ROOT/'records.json').write_text(json.dumps(rows,indent=2));print('case',case,'complete',len(rows),'rollouts',flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'verification.json').write_text(json.dumps(dict(exact_baselines=36,exact_pre_retreat_prefixes=72,exact_selected_predictions=sum(r['executed_predictions'] for r in rows),hashes_unchanged=True,no_test_evaluated=True),indent=2))

if __name__=='__main__':run()
