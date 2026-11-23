"""E170: PRIVILEGED probe/gripper coverage ablations, frozen E120 and release fix."""
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
REFERENCE=Path('outputs/ease_probe_gripper_clearance_v1/both')
ROOT=Path('outputs/ease_descent_execution_v1')
CONDITIONS=('fixed_allowance','history_allowance')
REPLAYS=(0,1,13,27)

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
    ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir();shutil.copy2(BASE/'cases.json',ROOT/'cases.json')
    save(ROOT/'protocol.json',dict(experiment='E170',scope='Privileged fitting diagnostic: two fixed variants x48 six-joint cases. E120+release baseline37/48; E169 both/probe-protected reference41/48 with4 collisions. Four exact E169 both replays before interventions. No held execution or training.',intervention='Keep E169 both geometry/probe protection. Only descent palm/pad rows receive a one-step clearance execution-error allowance. fixed_allowance:3mm; history_allowance:1mm plus maximum positive predicted-minus-actual clearance error over previous8 eligible descent steps, capped at20mm before the1mm floor. This uses past measured errors, never future outcomes. Same fixed parameters for every case. Convert allowance to effective-distance subtraction allowance/alpha, leaving actual geometry and reporting unchanged. Initial history empty; reset each episode. No outcome-dependent tuning.',preserved='E120 weights, frozen estimator bias, calibration, probe schedule/aperture, all phase budgets/goals and265mm release fix, strict contact/success scoring. Same linear QP, activation and fallback. No safety guarantee. Full gripper transforms privileged; gripper plane conservative outside table footprint.',screen='More than41 successes, fewer than4 collision episodes and zero lost reference successes. Report all failures and stalls even if screen fails.',inputs={str(p):fingerprint(p) for p in (REFERENCE/'records.json',BASE/'records.json',Path(__file__).with_name('probe_gripper_clearance.py'),Path(prior.__file__))}))
    for name in (*CONDITIONS,'replay'):
        root=ROOT/name;root.mkdir()
        for folder in ('trials','states','actions','histories','clearance'):(root/folder).mkdir()
    for name in ('descent_execution.py','probe_gripper_clearance.py','link_clearance_diagnostic.py','build_descent_execution.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
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
                if self.phase=='descend' and condition!='replay':
                    allowance[6:]=.003 if condition=='fixed_allowance' else .001+min(.020,max(error_history[-8:],default=0.))
                    effective-=allowance/prior.ALPHA
                eligible=bool(np.min(effective)<=prior.ACTIVATION)
                if eligible:
                    g=distance_gradient(self.model,self.data,self.scratch,self.geom_ids,self.table_id,self.n);np.testing.assert_array_equal(self.data.qpos,qpos);np.testing.assert_array_equal(self.data.qvel,qvel);j,_=c.predict();j=np.asarray(j,dtype=np.float64);metric=j.T@j+.015**2*np.eye(self.n);action,solve,rhs=prior.project(nominal,c.bias,before['q'],effective,g,metric)
                result=super().step(action,aperture);post=distances(self.model,self.data,self.geom_ids,self.table_id);self.substep_min=np.minimum(self.substep_min,post);self.clearance_recording=False
                prediction=None if g is None else d+g@(np.asarray(action)-c.bias)
                if self.phase=='descend' and prediction is not None:error_history.append(float(max(0.,np.max(prediction[6:]-post[6:]))))
                logs.append(dict(allowance=allowance.tolist(),effective_distance=effective.tolist(),predicted_distance=None if prediction is None else prediction.tolist(),full_qpos_before=qpos.tolist(),full_qvel_before=qvel.tolist(),phase=self.phase,goal=self.goal.tolist(),q_before=before['q'].tolist(),bias=c.bias.tolist(),nominal_action=nominal.tolist(),final_action=np.asarray(action).tolist(),executed_command=result[1].tolist(),aperture=float(aperture),distance_before=d.tolist(),distance_after=post.tolist(),substep_min=self.substep_min.tolist(),eligible=eligible,gradient=None if g is None else g.tolist(),rhs=None if rhs is None else rhs.tolist(),metric=None if metric is None else metric.tolist(),solve=solve,probe_anchor=None if self.phase!='probe' else self.anchor.tolist(),probe_index=None if self.phase!='probe' else int(self.probe_step-1)))
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
        r.update(condition=condition,constraint_steps=sum(l['solve']['attempted'] for l in logs),fallback_steps=sum(l['solve']['fallback'] for l in logs),changed_probe_steps=sum(np.max(np.abs(np.array(l['nominal_action'])-l['final_action']))>1e-10 for l in probe));r['changed_probe_steps']=int(r['changed_probe_steps']);rows.append(r);save(path,rows);print('E170 PRIVILEGED',condition,case,r['success'],'probe changes',r['changed_probe_steps'],'fallback',r['fallback_steps'],flush=True)
    assert len(rows)==(4 if condition=='replay' else 48)
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h
    save(root/'verification.json',dict(episodes=len(rows),exact_replays=4 if condition=='replay' else 0,calibration_identical=True,frozen_bias=True,release_fix_preserved=True,nominal_probe_schedule_verified=True,probe_aperture_unchanged=True,actual_commands_verified=True,checkpoint_hashes_unchanged=True,no_held_execution=True,privileged_geometry=True))

if __name__=='__main__':
    import sys
    if sys.argv[1]=='run':run(sys.argv[2])
    else:prepare()
