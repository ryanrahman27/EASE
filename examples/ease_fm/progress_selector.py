"""E111: frozen progress-aware broad action selector in full training episodes."""
import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from . import early_physics_rollout as prior
from .geometry_filter import GeometryArm,allowed_pair
from .arm_pick_place import PickPlace
from .physics_prediction_audit import forbidden
from .progress_action_bank import search_actions
from .benchmark import fingerprint

ROOT=Path('outputs/ease_progress_selector_v1')
MODES=('baseline','physics')
PHASES=('descend','lift','transport','lower','retreat')

def evaluate(case,t,seed,mode,parents,students,record_step=None,affected_phase=None,timings=None):
    assert mode in MODES
    select_fn=GeometryArm.select_action;step_fn=PickPlace.step;audit=[];pending={};scratch={}
    def predict(env,action,aperture):
        if 'data' not in scratch:scratch.update(data=mujoco.MjData(env.model),geometry=mujoco.MjData(env.geometry_model))
        d=scratch['data'];gd=scratch['geometry'];mujoco.mj_copyData(d,env.model,env.data);d.ctrl[:env.n]=np.clip(env.observe()['q']+np.clip(action,-.04,.04),-2.5,2.5);d.ctrl[env.n:]=aperture;hits=set();clearance=.002
        for _ in range(20):
            mujoco.mj_step(env.model,d);hits.update(forbidden(env.model,d,env.gids));mujoco.mj_copyData(gd,env.geometry_model,d);mujoco.mj_forward(env.geometry_model,gd)
            distances=[float(c.dist) for c in gd.contact[:gd.ncon] if not allowed_pair({int(c.geom1),int(c.geom2)},env.gids)];clearance=min([clearance,*distances])
        mujoco.mj_forward(env.model,d);error=np.r_[env.goal-d.site_xpos[env.site],.15*(np.array([0.,0.,1.])-d.site_xmat[env.site].reshape(3,3)[:,2])]
        return dict(pairs=[list(x) for x in sorted(hits)],qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),error_sq=float(error@error),minimum_clearance=clearance)
    def select(env,action,aperture,bias):
        original,entry=select_fn(env,action,aperture,bias)
        if getattr(env,'phase',None) not in PHASES:return original,entry
        started=time.perf_counter();pred=predict(env,original,aperture);chosen=original;chosen_pred=pred;name='original';current=forbidden(env.model,env.data,env.gids);options=[];branches=1
        unsafe=bool(pred['pairs']) or pred['minimum_clearance']<0
        if mode=='physics' and unsafe and not current and not env.bad:
            bank=search_actions(env,original,aperture,bias,entry,(case,seed,env.steps));branches+=bank['unique_physics_branches'];original_check=bank['candidates'][0]['40']
            for field in ('qpos','qvel','pairs','minimum_clearance'):assert original_check[field]==pred[field],field
            options=[c for c in bank['candidates'] if c['group']!='original']
            safe=[c for c in options if c['static_ok'] and not c['40']['collision'] and c['40']['minimum_clearance']>=0 and c['40']['progress']]
            if safe:
                best=min(safe,key=lambda c:(c['40']['error_sq'],-c['40']['minimum_clearance'],c['name']));name=best['name'];chosen=np.array(best['action']);chosen_pred=best['40']
        obs=env.observe();error=np.r_[env.goal-obs['position'],.15*(np.array([0.,0.,1.])-obs['rotation'][:,2])]
        row=dict(phase=env.phase,before_error_sq=float(error@error),original_after_error_sq=pred['error_sq'],selected_after_error_sq=chosen_pred['error_sq'],step=env.steps,clean=not current,prior_episode_collision=bool(env.bad),original_pairs=pred['pairs'],unsafe_original=unsafe,original_clearance=pred['minimum_clearance'],chosen_pairs=chosen_pred['pairs'],chosen_clearance=chosen_pred['minimum_clearance'],choice=name,changed=not np.array_equal(original,chosen),alternatives=options,original_action=original.tolist(),selected_action=chosen.tolist(),static_original=entry,prediction_count=branches)
        if row['changed']:assert not row['prior_episode_collision'] and not row['chosen_pairs'] and row['chosen_clearance']>=0 and row['before_error_sq']-row['selected_after_error_sq']>=max(1e-8,.01*row['before_error_sq'])
        audit.append(row);pending.update(prediction=chosen_pred,row=row)
        if timings is not None:timings.append(dict(case=case,seed=seed,mode=mode,phase=env.phase,step=env.steps,search=bool(options),branches=branches,ms=(time.perf_counter()-started)*1000))
        return chosen,entry
    def step(env,action,aperture):
        result=step_fn(env,action,aperture)
        if getattr(env,'phase',None) in PHASES:
            pred=pending['prediction'];actual=[list(x) for x in sorted({tuple(sorted(p)) for p in result[2]['bad']})];assert actual==pred['pairs']
            np.testing.assert_array_equal(env.data.qpos,pred['qpos']);np.testing.assert_array_equal(env.data.qvel,pred['qvel']);pending['row']['actual_pairs']=actual
        return result
    with patch.object(GeometryArm,'select_action',select),patch.object(PickPlace,'step',step):r=prior.evaluate(case,t,seed,mode,parents,students,record_step,affected_phase=None)
    old=json.loads((prior.ROOT/'trials'/f'{case}_{seed}_None_baseline.json').read_text())
    if mode=='baseline':assert json.loads(json.dumps(r))==old
    prefix=lambda x:[v for v in x['trace'] if v['phase']=='approach'];assert prefix(r)==prefix(old)
    r['physics_audit']=audit;return r

def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False)
    for f in ('trials','source'):(ROOT/f).mkdir()
    tasks=json.loads((prior.prior.TASKS/'cases.json').read_text());prior.AXES.update(json.loads((prior.prior.TASKS/'axis_catalog.json').read_text()));assert len(tasks)==12 and all(t['arm']['split']=='train' for t in tasks)
    parents,hashes=prior.prior.previous.models();students,hh=prior.prior.e99.load_students();hashes.update(hh)
    refs=[prior.ROOT/'trials'/f'{c}_{s}_None_baseline.json' for c in range(12) for s in range(3)]
    inputs={str(p):fingerprint(p) for p in [prior.prior.TASKS/'cases.json',prior.prior.TASKS/'axis_catalog.json',Path('outputs/ease_goal_action_audit_v1/summary.json'),*refs]}
    spec=dict(experiment='E111',inputs=inputs,checkpoints=hashes,
        design='All36 original training-body task/seed pairs x baseline/physics=72 full episodes. Same24 train/12 held-placement split; no route planner; unchanged model weights, apertures, phase budgets, success criteria. Before first collision only, in descend/lift/transport/lower/retreat. Calibration and approach identical. Preserve original command when40ms prediction has neither forbidden contact nor negative geometric clearance, or current forbidden contact/earlier collision exists.',
        bank='Frozen E110 five narrow candidates, per-joint +/- .01/.02/.04 around measured bias,64 seeded directions (.02/.04 norm), three model-based finite-difference directed candidates(.25/.5/1,damping.015,gain.35). Same seed11000000+case*100000+seed*10000+step; no tuning. Only40ms branches are simulated; no160ms selection or persistence gate.',
        selection='Trigger search on predicted forbidden contact OR minimum geometric clearance<0. Require original static admissibility, zero physical contacts, nonnegative geometric clearance across20 substeps and reduction of squared XYZ plus .15-tool-axis error by max(1e-8,1% initial). Choose minimum40ms error, tie larger clearance then name. If none qualifies keep original. Nonnegative margin is distinct from2mm; log both. No additional control steps or rescue after first collision.',
        gate='Strict training task success increase; retain every baseline success and nonincreasing collision counts in both train/held placements; no training seed decline. Report all task changes and phase progress. No automatic transfer test.',
        verification='All baseline tasks and approach prefixes exact. All executed selected next states/contact unions equal predictions. Bank original prediction matches lightweight evaluator bit-exactly. All changed actions meet progress/nonpenetration criteria. Record action-selection wall time separately from deterministic results.',
        videos='Case0 seed0, every task gain/loss, earliest changed pair, first collision-free failure and first changed pair per earlier phase, both modes. Outcome-selected illustrations; exact full-result replay.',
        limits='Privileged exact physics state/model and finite-difference model response. Compute cost may exceed40ms; this is not a real-time/history-only learned policy. Held placements share training bodies. No E105 or reserved final test.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('progress_selector.py','progress_action_bank.py','early_physics_rollout.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    rows=[];timings=[]
    for case,t in enumerate(tasks):
        for seed in range(3):
            for mode in MODES:
                r=evaluate(case,t,seed,mode,parents,students,timings=timings);(ROOT/'trials'/f'{case}_{seed}_None_{mode}.json').write_text(json.dumps(r));a=r['physics_audit']
                rows.append(dict(**{k:v for k,v in r.items() if k not in ('trace','contacts','calibration','physics_audit')},changed_actions=sum(x['changed'] for x in a),eligible_warnings=sum(x['clean'] and not x['prior_episode_collision'] and x['unsafe_original'] for x in a),no_safe_alternative=sum(bool(x['alternatives']) and x['choice']=='original' for x in a),changed_after_prior_collision=sum(x['changed'] and x['prior_episode_collision'] for x in a),prediction_count=sum(x['prediction_count'] for x in a),executed_predictions=len(a),changed_with_2mm=sum(x['changed'] and x['chosen_clearance']>=.002-1e-6 for x in a)))
        (ROOT/'records.json').write_text(json.dumps(rows,indent=2));(ROOT/'timings.json').write_text(json.dumps(timings));print('case',case,'complete',len(rows),'rollouts',flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'verification.json').write_text(json.dumps(dict(exact_baselines=36,exact_pre_descent_prefixes=72,exact_selected_predictions=sum(r['executed_predictions'] for r in rows),hashes_unchanged=True,no_test_evaluated=True),indent=2))

if __name__=='__main__':run()
