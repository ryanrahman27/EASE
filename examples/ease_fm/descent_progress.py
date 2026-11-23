"""E171: PRIVILEGED probe/gripper coverage ablations, frozen E120 and release fix."""
from pathlib import Path
from unittest.mock import patch
import shutil
import numpy as np
import mujoco
from . import link_clearance_diagnostic as prior
from .abstaining_task_value_gate import read,save,fingerprint
from .arm_pick_place import PickPlace
old=prior.old
BASE=prior.BASE
REFERENCE=Path('outputs/ease_descent_execution_v1/history_allowance')
ROOT=Path('outputs/ease_descent_progress_v1')
CONDITIONS=('direct_task','limit_gated')
REPLAYS=(0,26,62,67)

def geometry_ids(model,n,gripper):
    links,table=prior.geometry_ids(model,n)
    if gripper:
        links += [mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name) for name in ('gripper_palm','gripper_left_pad','gripper_right_pad')]
    assert len(set(links))==len(links) and all(g>=0 for g in links)
    return links,table

def distances(model,data,ids,table):
    values=prior.distances(model,data,ids[:6],table).tolist()
    if len(ids)>6:
        # Conservative infinite tabletop plane; exact lowest vertex of each box.
        table_rotation=data.geom_xmat[table].reshape(3,3)
        np.testing.assert_allclose(table_rotation,np.eye(3),atol=1e-12,rtol=0)
        top=data.geom_xpos[table,2]+model.geom_size[table,2]
        for g in ids[6:]:
            assert model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX
            rotation=data.geom_xmat[g].reshape(3,3)
            values.append(float(data.geom_xpos[g,2]-np.abs(rotation[2])@model.geom_size[g]-top))
    return np.asarray(values)

def distance_gradient(model,data,scratch,ids,table,n):
    scratch.qpos[:]=data.qpos;scratch.qvel[:]=data.qvel;scratch.ctrl[:]=data.ctrl;gradient=np.zeros((len(ids),n))
    for k in range(n):
        original=data.qpos[k];scratch.qpos[k]=original+prior.FD;mujoco.mj_forward(model,scratch);plus=distances(model,scratch,ids,table)
        scratch.qpos[k]=original-prior.FD;mujoco.mj_forward(model,scratch);minus=distances(model,scratch,ids,table);gradient[:,k]=(plus-minus)/(2*prior.FD);scratch.qpos[k]=original
    return gradient

def prepare():
    assert (ROOT/'diagnosis.json').exists();(ROOT/'source').mkdir();shutil.copy2(BASE/'cases.json',ROOT/'cases.json')
    save(ROOT/'protocol.json',dict(experiment='E171',scope='Fitting-only privileged control diagnostic; two variants x48 six-joint cases and4 exact E170 history-allowance replays. E170 reference44/48 with0 collisions. Original E120+release37/48 remains baseline. No held execution or training.',diagnosis='All four failures saturate joint5 near+2.5rad during descent, with active clearance constraints and little or negative late progress. Nominal damped least-squares action is independently clipped before metric projection, potentially distorting its task direction.',intervention='direct_task: on every descent step, minimize the original E120-estimated task residual directly subject to all original bounds/clearance constraints. limit_gated: do this only if any measured absolute arm joint coordinate>=2.4rad. Equivalent QP target is the UNCLIPPED damped least-squares command J.T solve(J J.T+.015^2 I,.35 error)+frozen_bias, with metric J.T J+.015^2 I. Original projection minimizes distance to the CLIPPED command. No new Jacobian or goal source. All thresholds and objectives frozen before outcomes.',preserved='History-based descent error allowance from E170, arm/palm/pad constraints, probe protection/schedule/aperture, frozen E120 checkpoints and bias, release265mm, phase budgets and strict scoring. No extra steps or outcome-based fallback. Same solver and logged hold fallback; no safety guarantee.',screen='More than44 successes, zero collisions, zero lost E170 successes. Retain and report every outcome; no automatic baseline replacement.',inputs={str(p):fingerprint(p) for p in (REFERENCE/'records.json',BASE/'records.json',ROOT/'diagnosis.json',Path(__file__).with_name('descent_execution.py'),Path(prior.__file__))}))
    for name in (*CONDITIONS,'replay'):
        root=ROOT/name;root.mkdir()
        for folder in ('trials','states','actions','histories','clearance'):(root/folder).mkdir()
    for name in ('descent_progress.py','descent_execution.py','probe_gripper_clearance.py','link_clearance_diagnostic.py','build_descent_progress.py','descent_progress_diagnosis.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py')]})

def run(condition):
    assert condition in (*CONDITIONS,'replay');root=ROOT/condition;protect_probe=True;gripper=True
    if condition!='replay':assert read(ROOT/'replay/verification.json')['exact_replays']==4
    _,models,hashes=old.broad.setup();core=old.broad.previous.core;core.ROOT=root;old.broad.previous.ROOT=root;original_targets=core.targets;original_rollout=core.probe_rollout;tasks=read(ROOT/'cases.json');choices=read(BASE/'reference_choices.json');folds=read(BASE/'folds.json');path=root/'records.json';rows=read(path) if path.exists() else [];done={r['case'] for r in rows}
    for case,t in enumerate(tasks):
        if t['validation'] or len(t['home'])!=6 or case in done or (condition=='replay' and case not in REPLAYS):continue
        logs=[];error_history=[]
        class ProtectedArm(PickPlace):
            def contacts(self):
                result=super().contacts()
                if getattr(self,'clearance_recording',False):self.substep_min=np.minimum(self.substep_min,distances(self.model,self.data,self.geom_ids,self.table_id))
                return result
            def step(self,action,aperture):
                c=getattr(self,'clearance_controller',None)
                if c is None:return super().step(action,aperture)
                before=self.observe();nominal=np.asarray(action).copy();qpos=self.data.qpos.copy();qvel=self.data.qvel.copy();d=distances(self.model,self.data,self.geom_ids,self.table_id);self.substep_min=d.copy();self.clearance_recording=True;g=None;rhs=None;metric=None;solve=dict(attempted=False,solver_success=True,fallback=False,status=0,iterations=0)
                allowance=np.zeros(len(d));effective=d.copy()
                if self.phase=='descend':
                    allowance[6:]=.001+min(.020,max(error_history[-8:],default=0.))
                    effective-=allowance/prior.ALPHA
                direct=bool(self.phase=='descend' and condition!='replay' and (condition=='direct_task' or np.max(np.abs(before['q']))>=2.4))
                optimization_target=nominal.copy();eligible=bool(direct or np.min(effective)<=prior.ACTIVATION)
                if eligible:
                    g=distance_gradient(self.model,self.data,self.scratch,self.geom_ids,self.table_id,self.n);np.testing.assert_array_equal(self.data.qpos,qpos);np.testing.assert_array_equal(self.data.qvel,qvel);j,_=c.predict();j=np.asarray(j,dtype=np.float64);metric=j.T@j+.015**2*np.eye(self.n)
                    if direct:
                        error=np.r_[self.goal-before['position'],.15*(np.array([0.,0.,1.])-before['rotation'][:,2])]
                        optimization_target=j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*error)+c.bias
                    action,solve,rhs=prior.project(optimization_target,c.bias,before['q'],effective,g,metric)
                result=super().step(action,aperture);post=distances(self.model,self.data,self.geom_ids,self.table_id);self.substep_min=np.minimum(self.substep_min,post);self.clearance_recording=False
                prediction=None if g is None else d+g@(np.asarray(action)-c.bias)
                if self.phase=='descend' and prediction is not None:error_history.append(float(max(0.,np.max(prediction[6:]-post[6:]))))
                logs.append(dict(direct_task=direct,optimization_target=optimization_target.tolist(),estimated_jacobian=None if not eligible else j.tolist(),allowance=allowance.tolist(),effective_distance=effective.tolist(),predicted_distance=None if prediction is None else prediction.tolist(),full_qpos_before=qpos.tolist(),full_qvel_before=qvel.tolist(),phase=self.phase,goal=self.goal.tolist(),q_before=before['q'].tolist(),bias=c.bias.tolist(),nominal_action=nominal.tolist(),final_action=np.asarray(action).tolist(),executed_command=result[1].tolist(),aperture=float(aperture),distance_before=d.tolist(),distance_after=post.tolist(),substep_min=self.substep_min.tolist(),eligible=eligible,gradient=None if g is None else g.tolist(),rhs=None if rhs is None else rhs.tolist(),metric=None if metric is None else metric.tolist(),solve=solve,probe_anchor=None if self.phase!='probe' else self.anchor.tolist(),probe_index=None if self.phase!='probe' else int(self.probe_step-1)))
                return result
        def targets(xy,target):
            for phase,goal,aperture,budget in original_targets(xy,target):
                goal=goal.copy()
                if phase in ('lower','release'):goal[2]=.265
                yield phase,goal,aperture,budget
        def rollout(env,home,c,**kwargs):
            env.clearance_controller=c;env.geom_ids,env.table_id=geometry_ids(env.model,env.n,gripper);env.scratch=mujoco.MjData(env.model);env.clearance_recording=False
            return original_rollout(env,home,c,**kwargs)
        with patch.object(old,'ROOT',root),patch.object(core,'PickPlace',ProtectedArm),patch.object(core,'targets',targets),patch.object(core,'probe_rollout',rollout):r=old.evaluate(case,t,'e120_ensemble',models,None,choices[folds[str(case)]])
        stem=f'{case}_e120_ensemble_frozen_bias';save(root/'clearance'/f'{stem}.json',logs);trial=read(root/'trials'/f'{stem}.json');base=read(BASE/'trials'/f'{stem}.json');assert trial['calibration']==base['calibration'] and trial['initial_estimator']==base['initial_estimator'];assert trial['initial_estimator']['bias']==trial['final_estimator']['bias'];assert len(logs)==len(trial['trace'])
        probe=[l for l in logs if l['phase']=='probe'];assert len(probe)==32
        for i,l in enumerate(probe):
            expected=np.clip(np.array(l['probe_anchor'])+core.probe_offset(i,6)-l['q_before']+np.array(l['bias']),-.04,.04);np.testing.assert_array_equal(expected,l['nominal_action']);assert l['probe_index']==i and l['aperture']==0.
            if not protect_probe:np.testing.assert_array_equal(l['nominal_action'],l['final_action'])
        for l in logs:
            np.testing.assert_array_equal(np.clip(np.array(l['q_before'])+np.clip(l['final_action'],-.04,.04),-2.5,2.5)-l['q_before'],l['executed_command'])
            if l['eligible'] and l['solve']['solver_success']:assert np.min(np.array(l['gradient'])@l['final_action']-l['rhs'])>=-1e-7
        reference=read(REFERENCE/'trials'/f'{stem}.json');states=np.load(root/'states'/f'{stem}.npz')['states']
        if condition=='replay':
            for key in ('trace','phases','success','collision','final_estimator'):assert trial[key]==reference[key],key
            np.testing.assert_array_equal(states,np.load(REFERENCE/'states'/f'{stem}.npz')['states'])
        r.update(condition=condition,constraint_steps=sum(l['solve']['attempted'] for l in logs),fallback_steps=sum(l['solve']['fallback'] for l in logs),changed_probe_steps=sum(np.max(np.abs(np.array(l['nominal_action'])-l['final_action']))>1e-10 for l in probe));r['changed_probe_steps']=int(r['changed_probe_steps']);rows.append(r);save(path,rows);print('E171 PRIVILEGED',condition,case,r['success'],'probe changes',r['changed_probe_steps'],'fallback',r['fallback_steps'],flush=True)
    assert len(rows)==(4 if condition=='replay' else 48)
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h
    save(root/'verification.json',dict(episodes=len(rows),exact_replays=4 if condition=='replay' else 0,calibration_identical=True,frozen_bias=True,release_fix_preserved=True,nominal_probe_schedule_verified=True,probe_aperture_unchanged=True,actual_commands_verified=True,checkpoint_hashes_unchanged=True,no_held_execution=True,privileged_geometry=True))

if __name__=='__main__':
    import sys
    if sys.argv[1]=='run':run(sys.argv[2])
    else:prepare()
