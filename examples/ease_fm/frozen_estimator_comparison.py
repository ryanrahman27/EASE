"""E176: PRIVILEGED probe/gripper coverage ablations, frozen E120 and release fix."""
from pathlib import Path
from unittest.mock import patch
import shutil
import numpy as np
import mujoco
from . import link_clearance_diagnostic as prior
from .abstaining_task_value_gate import read,save,fingerprint
from .arm_pick_place import PickPlace,ROT_SCALE,rotation_error,targets
from .object_preserving_plan import plan
from .early_object_plan import early_gate,observation
old=prior.old
BASE=prior.BASE
REFERENCE=Path('outputs/ease_early_object_recovery_v1/early_guard')
ROOT=Path('outputs/ease_frozen_estimator_comparison_v1')
CONDITIONS=('e120_ensemble','analytical')
REPLAYS=(0,26,43,62,67)

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
    _,_,hashes=old.broad.setup()
    save(ROOT/'protocol.json',dict(experiment='E176',scope='48 six-joint fitting cases x2 estimators=96 full episodes;5 exact E175 preflight replays. E120 full48 must also exactly reproduce E175. No held bodies or training.',methods='E120 unchanged three-model reference/correction ensembles and bounded residual versus original analytical LocalPose: solve(gram,cross).T, forgetting.98, ridge floor1e-6; accumulated history before probe, last32 transitions from probe onward. Both update motion statistics online and freeze identical calibrated bias. Analytical has no neural correction.',controller='Exact E175 early risk and late stall gates, strict object-safe grasp-first planner, route tracking, probe/table/gripper protection, causal clearance allowance, guarded close/lift,265mm release, task goals and scoring. Same simulator geometry/object/contact privilege. No estimator-specific retuning or case-ID controller rules.',budget='Both:32 calibration steps,32 pre-close probe steps, same action bounds/dt. Approach80, descend100 or300 only after accepted route, close40,lift100,transport100,lower100,release40,retreat80,settle40. Same reach early-exit rule. Maximum944 control steps including calibration when routed,744 otherwise. Actual phase lengths may differ; report them, no forced padding or extra retries.',analysis='Report paired successes/losses, collisions, grasp/lift, trigger and abort counts, per-family results, total/median actual interactions and prediction errors. One deterministic paired rollout per fitting case; placements are not independent morphology samples. No superiority claim from a tie or a fitting-only result.',videos='Save all traces/states/actions/histories. Render both methods for cases0,26,43,62,65,67 and every discordant success/collision case, plus any missing success/failure category. Include archived E174 case26 failure for context, labeled historical and excluded from paired counts.',checkpoints=hashes,reference_sha256=fingerprint(REFERENCE/'records.json'),privileged=True))
    for name in (*CONDITIONS,'replay'):
        root=ROOT/name;root.mkdir()
        for folder in ('trials','states','actions','histories','clearance','planning'):(root/folder).mkdir()
    names=('frozen_estimator_comparison.py','build_frozen_estimator_comparison.py','early_object.py','early_object_plan.py','early_object_diagnosis.py','object_preserving_plan.py','object_preserving_planner.py','history_estimator_control.py','bounded_history_correction.py','training_estimator_core.py','local_history.py','pick_place_control.py','retention_fresh.py','arm_pick_place.py','local_probe.py','link_clearance_diagnostic.py')
    for name in names:shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py')]})
    save(ROOT/'status.json',dict(status='prepared',held_evaluated=False))

def run(condition):
    assert condition in (*CONDITIONS,'replay');root=ROOT/condition;method='e120_ensemble' if condition=='replay' else condition;protect_probe=True;gripper=True
    if condition!='replay':assert read(ROOT/'replay/verification.json')['exact_replays']==5
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
                before=self.observe();estimator_jacobian,_=c.predict();estimator_gram=c.gram.copy();estimator_cross=c.cross.copy();nominal=np.asarray(action).copy();qpos=self.data.qpos.copy();qvel=self.data.qvel.copy();d=distances(self.model,self.data,self.geom_ids,self.table_id);self.substep_min=d.copy();self.clearance_recording=True;g=None;rhs=None;metric=None;solve=dict(attempted=False,solver_success=True,fallback=False,status=0,iterations=0)
                allowance=np.zeros(len(d));effective=d.copy()
                if self.phase=='descend':
                    allowance[6:]=.001+min(.020,max(error_history[-8:],default=0.))
                    effective-=allowance/prior.ALPHA
                if self.phase=='descend':
                    pe=float(np.linalg.norm(self.goal-before['position']));self.descent_errors.append(pe)
                    self.object_history.append(observation(self.model,self.data,self.goal))
                    early=early_gate(self.object_history)
                    late=len(self.descent_errors)>=60 and .025<pe<.075 and self.descent_errors[-20]-pe<.001 and np.max(np.abs(before['q']))>=2.4
                    if not self.planning_attempted and (early or late):
                        self.trigger_reason='early_object_tilt' if early else 'late_stall'
                        self.planning_attempted=True;ctrl_before=self.data.ctrl.copy();time_before=self.data.time
                        planned=plan(self.model,qpos,qvel,self.goal);save(root/'planning'/f'{case}.json',planned)
                        np.testing.assert_array_equal(self.data.qpos,qpos);np.testing.assert_array_equal(self.data.qvel,qvel);np.testing.assert_array_equal(self.data.ctrl,ctrl_before);assert self.data.time==time_before
                        if planned['available']:self.route=np.array(planned['dense_path']);self.route_index=0;self.pick_goal=np.array(planned['pick_goal'])
                route_active=bool(self.phase=='descend' and self.route is not None);waypoint=None;optimization_target=nominal.copy()
                if route_active:
                    nearest=self.route_index+int(np.argmin(np.linalg.norm(self.route[self.route_index:]-before['q'],axis=1)));self.route_index=max(self.route_index,nearest);waypoint=self.route[min(self.route_index+3,len(self.route)-1)]
                    optimization_target=np.clip(waypoint-before['q']+c.bias,-.04,.04)
                requested_aperture=float(aperture);handoff_control=False
                if self.route is not None:
                    if self.phase=='close' and self.last_phase!='close':
                        self.close_anchor=before['q'].copy();gap=before['position']-self.pick_goal;drift=np.linalg.norm(self.data.xpos[self.obj,:2]-self.pick_goal[:2]);axis=np.linalg.norm(before['rotation'][:,2]-[0,0,1])
                        self.close_ready=bool(np.linalg.norm(gap[:2])<=.008 and abs(gap[2])<=.012 and axis<.08 and drift<=.008)
                    if self.phase=='lift' and self.last_phase=='close' and (not self.close_ready or self.close_good<5):self.aborted=True;self.abort_anchor=before['q'].copy()
                    if self.phase=='close':
                        optimization_target=np.clip(self.close_anchor-before['q']+c.bias,-.04,.04);aperture=aperture if self.close_ready else 0.;handoff_control=True
                    if self.aborted:optimization_target=np.clip(self.abort_anchor-before['q']+c.bias,-.04,.04);aperture=0.;handoff_control=True
                eligible=bool(route_active or handoff_control or np.min(effective)<=prior.ACTIVATION)
                if eligible:
                    g=distance_gradient(self.model,self.data,self.scratch,self.geom_ids,self.table_id,self.n);np.testing.assert_array_equal(self.data.qpos,qpos);np.testing.assert_array_equal(self.data.qvel,qvel);j,_=c.predict();j=np.asarray(j,dtype=np.float64);metric=j.T@j+.015**2*np.eye(self.n)
                    action,solve,rhs=prior.project(optimization_target,c.bias,before['q'],effective,g,metric)
                result=super().step(action,aperture);post=distances(self.model,self.data,self.geom_ids,self.table_id);self.substep_min=np.minimum(self.substep_min,post);self.clearance_recording=False
                prediction=None if g is None else d+g@(np.asarray(action)-c.bias)
                if self.phase=='descend' and prediction is not None:error_history.append(float(max(0.,np.max(prediction[6:]-post[6:]))))
                end_pads=len(super().contacts()[0])
                if self.phase=='close':self.close_good=self.close_good+1 if end_pads==2 else 0
                logs.append(dict(estimator_jacobian=estimator_jacobian.tolist(),estimator_gram=estimator_gram.tolist(),estimator_cross=estimator_cross.tolist(),estimator_local=bool(c.local),trigger_reason=self.trigger_reason,object_guard_observation=self.object_history[-1] if self.phase=='descend' else None,requested_aperture=requested_aperture,handoff_control=handoff_control,close_ready=self.close_ready,close_good=self.close_good,end_pads=end_pads,aborted=self.aborted,close_anchor=None if self.close_anchor is None else self.close_anchor.tolist(),abort_anchor=None if self.abort_anchor is None else self.abort_anchor.tolist(),pick_goal=None if self.pick_goal is None else self.pick_goal.tolist(),route_active=route_active,route_index=self.route_index,waypoint=None if waypoint is None else waypoint.tolist(),planning_attempted=self.planning_attempted,descent_observations=len(self.descent_errors),optimization_target=optimization_target.tolist(),estimated_jacobian=None if not eligible else j.tolist(),allowance=allowance.tolist(),effective_distance=effective.tolist(),predicted_distance=None if prediction is None else prediction.tolist(),full_qpos_before=qpos.tolist(),full_qvel_before=qvel.tolist(),phase=self.phase,goal=self.goal.tolist(),q_before=before['q'].tolist(),bias=c.bias.tolist(),nominal_action=nominal.tolist(),final_action=np.asarray(action).tolist(),executed_command=result[1].tolist(),aperture=float(aperture),distance_before=d.tolist(),distance_after=post.tolist(),substep_min=self.substep_min.tolist(),eligible=eligible,gradient=None if g is None else g.tolist(),rhs=None if rhs is None else rhs.tolist(),metric=None if metric is None else metric.tolist(),solve=solve,probe_anchor=None if self.phase!='probe' else self.anchor.tolist(),probe_index=None if self.phase!='probe' else int(self.probe_step-1)))
                self.last_phase=self.phase
                return result
        def targets(xy,target):
            for phase,goal,aperture,budget in original_targets(xy,target):
                goal=goal.copy()
                if phase in ('lower','release'):goal[2]=.265
                yield phase,goal,aperture,budget
        def rollout(env,home,c,**kwargs):
            env.object_history=[];env.trigger_reason=None;env.descent_errors=[];env.route=None;env.route_index=0;env.planning_attempted=False;env.extended_route=True;env.pick_goal=None;env.close_anchor=None;env.close_ready=False;env.close_good=0;env.last_phase=None;env.aborted=False;env.abort_anchor=None;env.clearance_controller=c;env.geom_ids,env.table_id=geometry_ids(env.model,env.n,gripper);env.scratch=mujoco.MjData(env.model);env.clearance_recording=False
            return route_rollout(env,home,c,**kwargs)
        with patch.object(old,'ROOT',root),patch.object(core,'PickPlace',ProtectedArm),patch.object(core,'targets',targets),patch.object(core,'probe_rollout',rollout):r=old.evaluate(case,t,method,models,None,choices[folds[str(case)]])
        stem=f'{case}_{method}_frozen_bias';reference_stem=f'{case}_e120_ensemble_frozen_bias';save(root/'clearance'/f'{stem}.json',logs);trial=read(root/'trials'/f'{stem}.json');base=read(BASE/'trials'/f'{reference_stem}.json');assert trial['calibration']==base['calibration'] and trial['initial_estimator']==base['initial_estimator'];assert trial['initial_estimator']['bias']==trial['final_estimator']['bias'];assert len(logs)==len(trial['trace'])
        probe=[l for l in logs if l['phase']=='probe'];assert len(probe)==32
        for i,l in enumerate(probe):
            expected=np.clip(np.array(l['probe_anchor'])+core.probe_offset(i,6)-l['q_before']+np.array(l['bias']),-.04,.04);np.testing.assert_array_equal(expected,l['nominal_action']);assert l['probe_index']==i and l['aperture']==0.
            if not protect_probe:np.testing.assert_array_equal(l['nominal_action'],l['final_action'])
        for l in logs:
            np.testing.assert_array_equal(np.clip(np.array(l['q_before'])+np.clip(l['final_action'],-.04,.04),-2.5,2.5)-l['q_before'],l['executed_command'])
            if l['eligible'] and l['solve']['solver_success']:assert np.min(np.array(l['gradient'])@l['final_action']-l['rhs'])>=-1e-7
        reference=read(REFERENCE/'trials'/f'{reference_stem}.json');states=np.load(root/'states'/f'{stem}.npz')['states']
        if method=='e120_ensemble':
            for key in ('trace','phases','success','collision','final_estimator'):assert trial[key]==reference[key],key
            np.testing.assert_array_equal(states,np.load(REFERENCE/'states'/f'{reference_stem}.npz')['states'])
        r.update(condition=condition,constraint_steps=sum(l['solve']['attempted'] for l in logs),fallback_steps=sum(l['solve']['fallback'] for l in logs),changed_probe_steps=sum(np.max(np.abs(np.array(l['nominal_action'])-l['final_action']))>1e-10 for l in probe));r['changed_probe_steps']=int(r['changed_probe_steps']);rows.append(r);save(path,rows);print('E176 PRIVILEGED',condition,case,r['success'],'probe changes',r['changed_probe_steps'],'fallback',r['fallback_steps'],flush=True)
    assert len(rows)==(5 if condition=='replay' else 48)
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h
    save(root/'verification.json',dict(episodes=len(rows),exact_replays=len(rows) if method=='e120_ensemble' else 0,calibration_identical=True,frozen_bias=True,release_fix_preserved=True,nominal_probe_schedule_verified=True,probe_aperture_unchanged=True,actual_commands_verified=True,checkpoint_hashes_unchanged=True,no_held_execution=True,privileged_geometry=True))

def route_rollout(env,home,controller=None,render=None,reset=True,sequence=None):
    obs=env.reset(home) if reset else env.observe();trace=[];phases=[];renderer=None
    if render:
        from PIL import Image
        Path(render).mkdir(parents=True,exist_ok=True);renderer=mujoco.Renderer(env.model,height=480,width=640)
        camera=mujoco.MjvCamera();camera.lookat[:]=[.25,0,.35];camera.distance=1.5;camera.azimuth=130;camera.elevation=-25
    for phase,goal,aperture,budget in (targets(env.xy,env.target) if sequence is None else sequence):
        held=0;reached=False
        maximum=300 if phase=='descend' and getattr(env,'extended_route',False) else budget
        for t in range(maximum):
            if t>=budget and getattr(env,'route',None) is None:
                t-=1;break
            if env.pick_goal is not None and phase in ('descend','close','lift'):
                goal=goal.copy();goal[:2]=env.pick_goal[:2];env.goal=goal.copy()
            error=np.r_[goal-obs['position'],ROT_SCALE*rotation_error(np.eye(3),obs['rotation'])]
            if controller is None:
                jac=env.jacobian()
                bias=env.privileged_bias() if hasattr(env,'privileged_bias') else env.data.qfrc_bias[:env.n]/(800*env.arm.strength)
            else:jac,bias=controller.predict()
            action=np.clip(jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(6),.35*error)+bias,-.04,.04)
            before=obs;obs,command,info=env.step(action,aperture)
            info['bad']=[list(pair) for pair in sorted({tuple(sorted(p)) for p in info['bad']})]
            if controller is not None:controller.update(before,obs,command)
            pe=float(np.linalg.norm(goal-obs['position']));re=float(np.linalg.norm(rotation_error(np.eye(3),obs['rotation'])))
            trace.append(dict(phase=phase,position=obs['position'].tolist(),object=env.data.xpos[env.obj].tolist(),position_error=pe,rotation_error=re,bad=info['bad'],pads=info['pads']))
            held=held+1 if pe<.008 and re<.08 else 0
            if held>=5 and phase not in ('close','release','settle','probe','hold'):
                reached=True;break
        phases.append(dict(phase=phase,reached=reached or (pe<.008 and re<.08),steps=t+1))
        if renderer:
            renderer.update_scene(env.data,camera=camera);Image.fromarray(renderer.render()).save(Path(render)/f'{phase}.png')
    if renderer:renderer.close()
    obj=env.data.xpos[env.obj].copy();addr=env.model.jnt_dofadr[env.oj];velocity=env.data.qvel[addr:addr+6]
    success=bool(env.grasped and env.lifted and not env.bad and np.linalg.norm(obj[:2]-env.target)<.025 and abs(obj[2]-.215)<.005 and np.linalg.norm(velocity)<.02 and not env.contacts()[0])
    return dict(success=success,grasped=bool(env.grasped),lifted=bool(env.lifted),collision=bool(env.bad),steps=env.steps,final_object=obj.tolist(),phases=phases,trace=trace)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='run':run(sys.argv[2])
    else:prepare()
