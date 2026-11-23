"""E112 adapter: E111 selection and physical prediction logic unchanged."""
import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from . import unseen_progress_rollout as prior
from .geometry_filter import GeometryArm,allowed_pair
from .arm_pick_place import PickPlace
from .physics_prediction_audit import forbidden
from .progress_action_bank import search_actions
from .benchmark import fingerprint

ROOT=Path('outputs/ease_unseen_progress_v1')
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
    r['physics_audit']=audit;return r
