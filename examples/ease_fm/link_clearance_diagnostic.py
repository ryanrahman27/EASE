"""E168: PRIVILEGED link/table constraints around frozen E120, fitting only."""
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
from scipy.optimize import minimize
from . import history_estimator_control as old
from .release_clearance_control import ROOT as BASE
from .arm_pick_place import PickPlace
from .abstaining_task_value_gate import read,save,fingerprint

ROOT=Path('outputs/ease_link_clearance_diagnostic_v1')
CONDITIONS=('joint_projection','task_projection')
REPLAY_CASES=(0,1,3,4)
MARGIN=.003
ACTIVATION=.030
ALPHA=.5
FD=1e-5

def geometry_ids(model,n):
    links=[]
    for k in range(n):
        body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f'link{k}');g=int(model.body_geomadr[body]);assert model.geom_bodyid[g]==body;links.append(g)
    table=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'table');return links,table

def distances(model,data,links,table):
    return np.array([mujoco.mj_geomDistance(model,data,g,table,1.,None) for g in links])

def distance_gradient(model,data,scratch,links,table,n):
    # Scratch forward kinematics only. Never advances or changes live state.
    scratch.qpos[:]=data.qpos;scratch.qvel[:]=data.qvel;scratch.ctrl[:]=data.ctrl;gradient=np.zeros((len(links),n))
    for k in range(n):
        original=data.qpos[k];scratch.qpos[k]=original+FD;mujoco.mj_forward(model,scratch);plus=distances(model,scratch,links,table)
        scratch.qpos[k]=original-FD;mujoco.mj_forward(model,scratch);minus=distances(model,scratch,links,table);gradient[:,k]=(plus-minus)/(2*FD);scratch.qpos[k]=original
    return gradient

def project(action,bias,q,d,g,metric):
    # Linearized barrier: d + G*(command - calibrated_bias) >= d - alpha*(d-margin).
    lower=np.maximum(-.04,-2.5-q);upper=np.minimum(.04,2.5-q);rhs=g@bias-ALPHA*(d-MARGIN)
    nominal_slack=g@action-rhs
    if np.min(nominal_slack)>=0 and np.all(action>=lower) and np.all(action<=upper):return action.copy(),dict(attempted=False,solver_success=True,fallback=False,status=0,iterations=0,linear_slack=nominal_slack.tolist()),rhs
    h=metric/(np.trace(metric)/len(action));fun=lambda u:.5*(u-action)@h@(u-action);jac=lambda u:h@(u-action)
    result=minimize(fun,np.clip(action,lower,upper),jac=jac,bounds=list(zip(lower,upper)),constraints=[dict(type='ineq',fun=lambda u:g@u-rhs,jac=lambda u:g)],method='SLSQP',options=dict(maxiter=60,ftol=1e-12,disp=False))
    good=bool(result.success and np.isfinite(result.x).all() and np.min(g@result.x-rhs)>=-1e-7 and np.all(result.x>=lower-1e-10) and np.all(result.x<=upper+1e-10))
    command=np.clip(result.x,lower,upper) if good else np.clip(bias,lower,upper)
    return command,dict(attempted=True,solver_success=good,fallback=not good,status=int(result.status),iterations=int(result.nit),linear_slack=(g@command-rhs).tolist(),message=str(result.message)),rhs

def prepare():
    ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir();shutil.copy2(BASE/'cases.json',ROOT/'cases.json')
    save(ROOT/'protocol.json',dict(experiment='E168',scope='PRIVILEGED GEOMETRY DIAGNOSTIC ONLY. All48 six-joint fitting placements x2 constraint metrics=96 new intervention episodes. Four E161 six-joint ancestry roots replayed exactly with constraint layer disabled before interventions. No held cases, seven-joint intervention or training data/weights changed.',
      baseline='Frozen E120 three-reference/three-correction ensemble with original bounded residual and E161265mm lower/release height. No higher approach, centering, descent cap or global rate limiter. Same calibration, fixed close probe, frozen measured bias, task waypoints, phase budgets and original strict success/contact scoring.',
      privileged_inputs='Exact simulator link-capsule/table signed distances and their current-q finite-difference gradients using scratch forward kinematics, h=1e-5rad. All six arm capsules constrained against table; gripper, object, ground and self-contact are still scored but not constrained. No future dynamics rollouts, object-pose controller input or collision outcome filtering. This is not history-only learned adaptation or a safety guarantee.',
      constraint='After original full E120 action a, solve closest-command convex QP subject to G*u >= G*frozen_bias - .5*(distance-.003m), original +/- .04 command bounds and actual joint target bounds +/-2.5rad. The model assumes measured joint increment approximately command-bias; inertia, model error, nonlinear geometry and within-step motion can violate real clearance. Activate gradient/optimization only when minimum current link/table distance<=.030m. If nominal action already satisfies all constraints exactly, retain it bit-for-bit. No post-selection parameter sweep.',
      metrics='joint_projection minimizes squared joint-action deviation. task_projection uses H=J_E120.T@J_E120+.015**2*I, retaining E120 estimated tool Jacobian, to favor preservation of task motion. H is normalized by mean diagonal for numerical conditioning. Identical constraints and parameters in both variants.',
      solver='SLSQP analytic objective and linear-constraint gradients,60iterations,ftol1e-12. Accept only successful finite solution satisfying linear inequalities within1e-7m and command bounds; otherwise use clipped frozen-bias hold and log constraint slack/fallback. No silent relaxation; fallback is not guaranteed safe or progressive.',
      isolation='Layer starts only after calibration and is disabled during the unchanged fixed probe. All non-probe task phases receive the same constraints. Record actual substep link/table distances separately from original forbidden-contact labels, commands, constraint matrices/slacks, QP status and full states/histories.',
      evaluation='Report success, all collision episodes, early approach/descent first contacts, actual minimum clearance, constraint violations, fallbacks, phase noncompletion and every paired rescue/regression versus release-fixed E12037/48. A positive fitting screen requires higher success, fewer early first contacts, no higher collisions and no lost baseline success. A collision reduction without progress is not task improvement. No automatic baseline replacement or held test.',
      margin_m=MARGIN,activation_m=ACTIVATION,barrier_alpha=ALPHA,finite_difference_rad=FD,replay_cases=list(REPLAY_CASES),inputs={str(BASE/p):fingerprint(BASE/p) for p in ('records.json','cases.json','verification.json','protocol.json')}))
    for condition in (*CONDITIONS,'replay'):
        root=ROOT/condition;root.mkdir()
        for f in ('trials','states','actions','histories','clearance'):(root/f).mkdir()
    for f in ('link_clearance_diagnostic.py','history_estimator_control.py','training_estimator_core.py','bounded_history_correction.py','local_probe.py','arm_pick_place.py'):shutil.copy2(Path(__file__).with_name(f),ROOT/'source'/f)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py')]})

def run(condition):
    assert condition in (*CONDITIONS,'replay');root=ROOT/condition
    if condition!='replay':assert read(ROOT/'replay/verification.json')['exact_baseline_replays']==4
    _,models,hashes=old.broad.setup();core=old.broad.previous.core;core.ROOT=root;old.broad.previous.ROOT=root;original_targets=core.targets;original_rollout=core.probe_rollout;tasks=read(ROOT/'cases.json');choices=read(BASE/'reference_choices.json');folds=read(BASE/'folds.json');path=root/'records.json';rows=read(path) if path.exists() else [];done={r['case'] for r in rows}
    for case,t in enumerate(tasks):
        if t['validation'] or len(t['home'])!=6 or (condition=='replay' and case not in REPLAY_CASES) or case in done:continue
        logs=[]
        class ClearanceArm(PickPlace):
            def contacts(self):
                result=super().contacts()
                if getattr(self,'clearance_recording',False):self.substep_min=np.minimum(self.substep_min,distances(self.model,self.data,self.link_ids,self.table_id))
                return result
            def step(self,action,aperture):
                c=getattr(self,'clearance_controller',None)
                if c is None:return super().step(action,aperture)
                before=self.observe();original=np.asarray(action).copy();live_qpos=self.data.qpos.copy();live_qvel=self.data.qvel.copy();d=distances(self.model,self.data,self.link_ids,self.table_id);self.substep_min=d.copy();self.clearance_recording=True;info={};g=None;rhs=None;metric=None;solve=dict(attempted=False,solver_success=True,fallback=False,status=0,iterations=0)
                eligible=bool(condition!='replay' and self.phase!='probe' and np.min(d)<=ACTIVATION)
                if eligible:
                    g=distance_gradient(self.model,self.data,self.scratch,self.link_ids,self.table_id,self.n);np.testing.assert_array_equal(self.data.qpos,live_qpos);np.testing.assert_array_equal(self.data.qvel,live_qvel)
                    if condition=='task_projection':j,_=c.predict();j=np.asarray(j,dtype=np.float64);metric=j.T@j+.015**2*np.eye(self.n)
                    else:metric=np.eye(self.n)
                    action,solve,rhs=project(original,c.bias,before['q'],d,g,metric)
                result=super().step(action,aperture);post=distances(self.model,self.data,self.link_ids,self.table_id);self.substep_min=np.minimum(self.substep_min,post);self.clearance_recording=False
                logs.append(dict(phase=self.phase,goal=self.goal.tolist(),q_before=before['q'].tolist(),bias=c.bias.tolist(),nominal_action=original.tolist(),final_action=np.asarray(action).tolist(),executed_command=result[1].tolist(),distance_before=d.tolist(),distance_after=post.tolist(),substep_min=self.substep_min.tolist(),eligible=eligible,gradient=None if g is None else g.tolist(),rhs=None if rhs is None else rhs.tolist(),metric=None if metric is None else metric.tolist(),solve=solve))
                return result
        def targets(xy,target):
            for name,goal,aperture,budget in original_targets(xy,target):
                goal=goal.copy()
                if name in ('lower','release'):goal[2]=.265
                yield name,goal,aperture,budget
        def rollout(env,home,c,**kwargs):
            env.clearance_controller=c;env.link_ids,env.table_id=geometry_ids(env.model,env.n);env.scratch=mujoco.MjData(env.model);env.clearance_recording=False
            return original_rollout(env,home,c,**kwargs)
        with patch.object(old,'ROOT',root),patch.object(core,'PickPlace',ClearanceArm),patch.object(core,'targets',targets),patch.object(core,'probe_rollout',rollout):r=old.evaluate(case,t,'e120_ensemble',models,None,choices[folds[str(case)]])
        stem=f'{case}_e120_ensemble_frozen_bias';save(root/'clearance'/f'{stem}.json',logs);trial=read(root/'trials'/f'{stem}.json');base=read(BASE/'trials'/f'{stem}.json');assert len(logs)==len(trial['trace']);assert trial['calibration']==base['calibration'] and trial['initial_estimator']==base['initial_estimator'];assert trial['initial_estimator']['bias']==trial['final_estimator']['bias']
        for l in logs:
            if condition=='replay' or l['phase']=='probe':np.testing.assert_array_equal(l['nominal_action'],l['final_action'])
            if l['eligible']:
                g=np.array(l['gradient']);rhs=np.array(l['rhs']);np.testing.assert_allclose(rhs,g@np.array(l['bias'])-ALPHA*(np.array(l['distance_before'])-MARGIN),rtol=0,atol=1e-14)
                if l['solve']['attempted'] and l['solve']['solver_success']:assert np.min(g@np.array(l['final_action'])-rhs)>=-1e-7
            expected=np.clip(np.array(l['q_before'])+np.clip(l['final_action'],-.04,.04),-2.5,2.5)-l['q_before'];np.testing.assert_array_equal(expected,l['executed_command'])
        if condition=='replay':
            for k in ('trace','phases','success','collision','final_estimator'):assert trial[k]==base[k],k
            np.testing.assert_array_equal(np.load(root/'states'/f'{stem}.npz')['states'],np.load(BASE/'states'/f'{stem}.npz')['states'])
        r['condition']=condition;r['constraint_steps']=sum(l['solve']['attempted'] for l in logs);r['fallback_steps']=sum(l['solve']['fallback'] for l in logs);rows.append(r);save(path,rows);print('E168 PRIVILEGED',condition,case,r['success'],'QP',r['constraint_steps'],'fallback',r['fallback_steps'],flush=True)
    assert len(rows)==(4 if condition=='replay' else 48)
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h
    save(root/'verification.json',dict(episodes=len(rows),exact_baseline_replays=4 if condition=='replay' else 0,calibration_identical=True,frozen_bias=True,release_fix_preserved=True,probe_commands_unchanged=True,actual_commands_and_linear_constraints_verified=True,scratch_did_not_mutate_live_joint_state=True,E120_checkpoint_hashes_unchanged=True,no_held_execution=True,privileged_geometry=True))

if __name__=='__main__':
    import sys
    if sys.argv[1]=='run':run(sys.argv[2])
    else:prepare()
