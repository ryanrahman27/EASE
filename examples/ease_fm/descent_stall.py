"""E172: PRIVILEGED probe/gripper coverage ablations, frozen E120 and release fix."""
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
ROOT=Path('outputs/ease_descent_stall_v1')
CONDITIONS=('stall_recovery',)
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
    assert (ROOT/'diagnosis_60_local.json').exists();(ROOT/'source').mkdir();shutil.copy2(BASE/'cases.json',ROOT/'cases.json')
    save(ROOT/'protocol.json',dict(experiment='E172',scope='One fixed fitting candidate x48 six-joint cases, plus4 exact E170 replays. Adaptive fitting development after E171 regressions; no held evidence. E17044/48 zero collisions remains protected reference; E120+release primary baseline unchanged.',gate='Measured local late descent stall: at least60 descent observations,25mm<position error<75mm, less than1mm error reduction over previous19 intervals, any absolute joint>=2.4rad. Latch for rest of descent. All thresholds frozen before E172 outcomes. Read-only gate diagnosis on E170 triggers only cases26,62,65,67; no case IDs or body identities enter control. Earlier broader stall rules also triggered successful case43 and are archived.',recovery='When active, direct E120 task objective plus joint-limit posture regularization: H=J.T J+.03^2 I, preferred relative command -0.02*sign(q) on joints abs(q)>2.35, zero elsewhere. Unconstrained center solve(H,J.T*.35*error+.03^2*posture)+frozen_bias, projected with same bounds and clearance constraints. Outside active descent use exact E170 action/protection. No extra steps, new task Jacobian, phase goal or aperture change.',preserved='E120 weights/frozen bias, calibration, history-based descent clearance allowance, probe protection/schedule,265mm release, original task budgets and strict scoring. Same QP solver and logged fallback. Privileged geometry diagnostic; not guaranteed safety or learned-only transfer.',screen='More than44 successes, zero collisions, no lost E170 successes. Also report all four descent errors and exact trace/state retention for every nonactivated episode.',inputs={str(p):fingerprint(p) for p in (REFERENCE/'records.json',ROOT/'diagnosis_60_local.json',Path(__file__).with_name('descent_progress.py'))}))
    for name in (*CONDITIONS,'replay'):
        root=ROOT/name;root.mkdir()
        for folder in ('trials','states','actions','histories','clearance'):(root/folder).mkdir()
    for name in ('descent_stall.py','descent_progress.py','descent_execution.py','probe_gripper_clearance.py','link_clearance_diagnostic.py','build_descent_stall.py','descent_stall_diagnosis.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
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
                if self.phase=='descend':
                    pe=float(np.linalg.norm(self.goal-before['position']));self.descent_errors.append(pe)
                    if condition!='replay' and len(self.descent_errors)>=60 and .025<pe<.075 and self.descent_errors[-20]-pe<.001 and np.max(np.abs(before['q']))>=2.4:self.recovery_active=True
                direct=bool(self.phase=='descend' and self.recovery_active);posture=np.zeros(self.n)
                optimization_target=nominal.copy();eligible=bool(direct or np.min(effective)<=prior.ACTIVATION)
                if eligible:
                    g=distance_gradient(self.model,self.data,self.scratch,self.geom_ids,self.table_id,self.n);np.testing.assert_array_equal(self.data.qpos,qpos);np.testing.assert_array_equal(self.data.qvel,qvel);j,_=c.predict();j=np.asarray(j,dtype=np.float64);metric=j.T@j+.015**2*np.eye(self.n)
                    if direct:
                        error=np.r_[self.goal-before['position'],.15*(np.array([0.,0.,1.])-before['rotation'][:,2])]
                        posture=np.where(np.abs(before['q'])>2.35,-.02*np.sign(before['q']),0.);metric=j.T@j+.03**2*np.eye(self.n)
                        optimization_target=np.linalg.solve(metric,j.T@(.35*error)+.03**2*posture)+c.bias
                    action,solve,rhs=prior.project(optimization_target,c.bias,before['q'],effective,g,metric)
                result=super().step(action,aperture);post=distances(self.model,self.data,self.geom_ids,self.table_id);self.substep_min=np.minimum(self.substep_min,post);self.clearance_recording=False
                prediction=None if g is None else d+g@(np.asarray(action)-c.bias)
                if self.phase=='descend' and prediction is not None:error_history.append(float(max(0.,np.max(prediction[6:]-post[6:]))))
                logs.append(dict(posture=posture.tolist(),descent_observations=len(self.descent_errors),direct_task=direct,optimization_target=optimization_target.tolist(),estimated_jacobian=None if not eligible else j.tolist(),allowance=allowance.tolist(),effective_distance=effective.tolist(),predicted_distance=None if prediction is None else prediction.tolist(),full_qpos_before=qpos.tolist(),full_qvel_before=qvel.tolist(),phase=self.phase,goal=self.goal.tolist(),q_before=before['q'].tolist(),bias=c.bias.tolist(),nominal_action=nominal.tolist(),final_action=np.asarray(action).tolist(),executed_command=result[1].tolist(),aperture=float(aperture),distance_before=d.tolist(),distance_after=post.tolist(),substep_min=self.substep_min.tolist(),eligible=eligible,gradient=None if g is None else g.tolist(),rhs=None if rhs is None else rhs.tolist(),metric=None if metric is None else metric.tolist(),solve=solve,probe_anchor=None if self.phase!='probe' else self.anchor.tolist(),probe_index=None if self.phase!='probe' else int(self.probe_step-1)))
                return result
        def targets(xy,target):
            for phase,goal,aperture,budget in original_targets(xy,target):
                goal=goal.copy()
                if phase in ('lower','release'):goal[2]=.265
                yield phase,goal,aperture,budget
        def rollout(env,home,c,**kwargs):
            env.descent_errors=[];env.recovery_active=False;env.clearance_controller=c;env.geom_ids,env.table_id=geometry_ids(env.model,env.n,gripper);env.scratch=mujoco.MjData(env.model);env.clearance_recording=False
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
        r.update(condition=condition,constraint_steps=sum(l['solve']['attempted'] for l in logs),fallback_steps=sum(l['solve']['fallback'] for l in logs),changed_probe_steps=sum(np.max(np.abs(np.array(l['nominal_action'])-l['final_action']))>1e-10 for l in probe));r['changed_probe_steps']=int(r['changed_probe_steps']);rows.append(r);save(path,rows);print('E172 PRIVILEGED',condition,case,r['success'],'probe changes',r['changed_probe_steps'],'fallback',r['fallback_steps'],flush=True)
    assert len(rows)==(4 if condition=='replay' else 48)
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h
    save(root/'verification.json',dict(episodes=len(rows),exact_replays=4 if condition=='replay' else 0,calibration_identical=True,frozen_bias=True,release_fix_preserved=True,nominal_probe_schedule_verified=True,probe_aperture_unchanged=True,actual_commands_verified=True,checkpoint_hashes_unchanged=True,no_held_execution=True,privileged_geometry=True))

if __name__=='__main__':
    import sys
    if sys.argv[1]=='run':run(sys.argv[2])
    else:prepare()
