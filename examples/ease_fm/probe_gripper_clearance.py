"""E169: PRIVILEGED probe/gripper coverage ablations, frozen E120 and release fix."""
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
REFERENCE=prior.ROOT/'task_projection'
ROOT=Path('outputs/ease_probe_gripper_clearance_v1')
CONDITIONS=('probe_only','gripper_only','both')
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
    save(ROOT/'protocol.json',dict(experiment='E169',scope='PRIVILEGED GEOMETRY DIAGNOSTIC. Three coverage ablations x48 six-joint fitting placements=144 new episodes; four exact E168 task-projection replays first. E120 plus release remains primary unchanged baseline37/48; E168 task-projection39/48 is the matched privileged reference. No held evaluation or training changes.',
      conditions='probe_only: existing six arm/table constraints also apply to fixed local probe commands. gripper_only: add palm and both finger pads against table, leaving probe unprotected as before. both: add gripper geometry and protect probes. All use unchanged E168 task-projection objective,3mm margin,30mm activation,.5 barrier gain,1e-5 finite difference, SLSQP settings and explicit hold fallback. No condition/threshold search after outcomes.',
      probe='Initial32-step calibration is unchanged and unfiltered. Local close-phase probe still schedules the same32 anchor/offset commands and apertures; probe-protected variants can modify the executed arm commands with the same QP and update estimator history from actual measured motion. No extra probe, nominal amplitude change or weight update. Report probe interventions and resulting excitation separately.',
      gripper='Privileged current palm/pad/table distances and arm-joint gradients; gripper aperture commands remain unchanged. Gradient holds jaw positions fixed, so actual jaw motion/dynamics can violate predicted clearance. Object contact remains allowed under original scoring. Ground, self-contact and object interactions are scored but not constrained. No future physics rollout.',
      isolation='Original E120 reference/correction ensemble and bounded residual remain frozen;265mm release fix, all other goals/budgets, strict success/contact criteria and frozen measured bias unchanged. Only the two coverage flags change. This is not history-only learned transfer or guaranteed safety.',
      verification='Exact replay of E168 task_projection cases0,1,13,27. Probe-only trajectory prefix before first probe must equal E168. Both-prefix before probe must equal gripper-only. Validate actual commands, geometry gradients/distances, probe nominal schedule and unchanged apertures. Preserve all failures, stalls, fallback events and complete trajectories.',
      screen='Report comparisons to both E120+release and E168 task projection, all rescues/losses, collision categories, collision-free failures, probe excitation and fallbacks. A strict improvement over E168 requires>39 successes, no more than7 collisions and no lost E168 success. No automatic baseline replacement or held test.',
      inputs={str(p):fingerprint(p) for p in (prior.ROOT/'summary.json',prior.ROOT/'verification.json',REFERENCE/'records.json',BASE/'records.json',Path(prior.__file__))}))
    for name in (*CONDITIONS,'replay'):
        root=ROOT/name;root.mkdir()
        for folder in ('trials','states','actions','histories','clearance'):(root/folder).mkdir()
    for name in ('probe_gripper_clearance.py','link_clearance_diagnostic.py','history_estimator_control.py','training_estimator_core.py','bounded_history_correction.py','local_probe.py','arm_pick_place.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py'),Path(prior.__file__)]})

def run(condition):
    assert condition in (*CONDITIONS,'replay');root=ROOT/condition;protect_probe=condition in ('probe_only','both');gripper=condition in ('gripper_only','both')
    if condition!='replay':assert read(ROOT/'replay/verification.json')['exact_replays']==4
    _,models,hashes=old.broad.setup();core=old.broad.previous.core;core.ROOT=root;old.broad.previous.ROOT=root;original_targets=core.targets;original_rollout=core.probe_rollout;tasks=read(ROOT/'cases.json');choices=read(BASE/'reference_choices.json');folds=read(BASE/'folds.json');path=root/'records.json';rows=read(path) if path.exists() else [];done={r['case'] for r in rows}
    for case,t in enumerate(tasks):
        if t['validation'] or len(t['home'])!=6 or case in done or (condition=='replay' and case not in REPLAYS):continue
        logs=[]
        class ProtectedArm(PickPlace):
            def contacts(self):
                result=super().contacts()
                if getattr(self,'clearance_recording',False):self.substep_min=np.minimum(self.substep_min,distances(self.model,self.data,self.geom_ids,self.table_id))
                return result
            def step(self,action,aperture):
                c=getattr(self,'clearance_controller',None)
                if c is None:return super().step(action,aperture)
                before=self.observe();nominal=np.asarray(action).copy();qpos=self.data.qpos.copy();qvel=self.data.qvel.copy();d=distances(self.model,self.data,self.geom_ids,self.table_id);self.substep_min=d.copy();self.clearance_recording=True;g=None;rhs=None;metric=None;solve=dict(attempted=False,solver_success=True,fallback=False,status=0,iterations=0)
                eligible=bool((self.phase!='probe' or protect_probe) and np.min(d)<=prior.ACTIVATION)
                if eligible:
                    g=distance_gradient(self.model,self.data,self.scratch,self.geom_ids,self.table_id,self.n);np.testing.assert_array_equal(self.data.qpos,qpos);np.testing.assert_array_equal(self.data.qvel,qvel);j,_=c.predict();j=np.asarray(j,dtype=np.float64);metric=j.T@j+.015**2*np.eye(self.n);action,solve,rhs=prior.project(nominal,c.bias,before['q'],d,g,metric)
                result=super().step(action,aperture);post=distances(self.model,self.data,self.geom_ids,self.table_id);self.substep_min=np.minimum(self.substep_min,post);self.clearance_recording=False
                logs.append(dict(phase=self.phase,goal=self.goal.tolist(),q_before=before['q'].tolist(),bias=c.bias.tolist(),nominal_action=nominal.tolist(),final_action=np.asarray(action).tolist(),executed_command=result[1].tolist(),aperture=float(aperture),distance_before=d.tolist(),distance_after=post.tolist(),substep_min=self.substep_min.tolist(),eligible=eligible,gradient=None if g is None else g.tolist(),rhs=None if rhs is None else rhs.tolist(),metric=None if metric is None else metric.tolist(),solve=solve,probe_anchor=None if self.phase!='probe' else self.anchor.tolist(),probe_index=None if self.phase!='probe' else int(self.probe_step-1)))
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
        elif condition=='probe_only':
            index=next(i for i,x in enumerate(trial['trace']) if x['phase']=='probe');assert index==next(i for i,x in enumerate(reference['trace']) if x['phase']=='probe');assert trial['trace'][:index]==reference['trace'][:index];np.testing.assert_array_equal(states[:index],np.load(REFERENCE/'states'/f'{stem}.npz')['states'][:index])
        r.update(condition=condition,constraint_steps=sum(l['solve']['attempted'] for l in logs),fallback_steps=sum(l['solve']['fallback'] for l in logs),changed_probe_steps=sum(np.max(np.abs(np.array(l['nominal_action'])-l['final_action']))>1e-10 for l in probe));r['changed_probe_steps']=int(r['changed_probe_steps']);rows.append(r);save(path,rows);print('E169 PRIVILEGED',condition,case,r['success'],'probe changes',r['changed_probe_steps'],'fallback',r['fallback_steps'],flush=True)
    assert len(rows)==(4 if condition=='replay' else 48)
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h
    save(root/'verification.json',dict(episodes=len(rows),exact_replays=4 if condition=='replay' else 0,calibration_identical=True,frozen_bias=True,release_fix_preserved=True,nominal_probe_schedule_verified=True,probe_aperture_unchanged=True,actual_commands_verified=True,checkpoint_hashes_unchanged=True,no_held_execution=True,privileged_geometry=True))

if __name__=='__main__':
    import sys
    if sys.argv[1]=='run':run(sys.argv[2])
    else:prepare()
