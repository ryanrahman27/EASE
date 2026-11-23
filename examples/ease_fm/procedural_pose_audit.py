"""E55: position/tool-Z action ranking on procedural training-arm histories."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .benchmark import fingerprint
from .paired_data import register
from .mixed_sim import Arm,MixedReach
from .motion_transformer import MotionController
from .real_arm_reach import load_models
from .native_reliability import bounded_error
from .action_objective_report import agreement


from .action_objective_audit import operator, copy_controller as copied
from .procedural_action_audit import clone
from .retention_fresh import GuardedPose
from .arm_pick_place import direction_jacobian


def observe(env):
    return dict(q=env.data.qpos.copy(),position=env.data.site_xpos[0].copy(),rotation=env.data.site_xmat[0].reshape(3,3).copy())


def vector(obs):return np.r_[obs['position'],.15*obs['rotation'][:,2]]


def reference_step(reference,goals,step):
    goal=np.asarray(goals[(step//50)%2]);result=reference.copy()
    result[:3]+=bounded_error(goal[:3],result[:3],.002)
    z=result[3:]/.15;desired=goal[3:]/.15
    angle=np.arccos(np.clip(z@desired,-1,1));fraction=min(1.,.01/max(angle,1e-12))
    z=(1-fraction)*z+fraction*desired;z/=np.linalg.norm(z)
    result[3:]=.15*z
    return result


def targets(env,body):
    home=env.data.qpos.copy();rng=np.random.default_rng(55000+body);result=[]
    for index in range(2):
        q=np.clip(home+rng.uniform(-.15,.15,len(home)),-2.5,2.5)
        with clone(env):
            env.data.qpos[:]=q;env.data.qvel[:]=0;env.data.ctrl[:]=q;mujoco.mj_forward(env.model,env.data)
            result.append(dict(q=q.tolist(),pose=vector(observe(env)).tolist(),endpoint_contact=bool(env.data.ncon)))
    return result


def execute(env,c,reference,goals,step):
    reference=reference_step(reference,goals,step);env.goal=reference[:3].copy()
    obs=observe(env);error=reference-vector(obs);jac,bias=c.predict()
    motion=operator(jac)@(.35*error);action=np.clip(motion+bias,-.04,.04)
    _,_,info=env.step(action);after=observe(env);c.update(obs,after,info['executed_action'])
    residual=reference-vector(after)
    return reference,dict(error_squared=float(residual@residual),position_error=float(np.linalg.norm(residual[:3])),
        direction_error=float(np.linalg.norm(residual[3:])/.15),collision=bool(info['collision']),
        command=info['executed_action'].tolist(),position=after['position'].tolist(),q=after['q'].tolist())


def branch(env,c,reference,goals,step):
    with clone(env):
        result={}
        for offset in range(5):
            reference,info=execute(env,c,reference,goals,step+offset)
            if offset in [0,4]:result[str(offset+1)]=info
        return result


def run(data,parents,training,output):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    manifest=register(data);arms={a['seed']:Arm(**a) for a in manifest['arms'] if a['split']=='train'}
    tasks=[]
    for body,arm in arms.items():
        env=MixedReach(arm,horizon=1000,synchronize=True);env.reset(0,[0,0,.9],spread=0)
        endpoints=targets(env,body)
        tasks.append(dict(body=body,goals=[x['pose'] for x in endpoints],endpoints=endpoints))
    assert len(tasks)==8
    models={'analytical':None};hashes={}
    for condition,folder in [('original',parents),('prediction',str(Path(training)/'prediction')),('control',str(Path(training)/'control'))]:
        mm,hh=load_models(folder);hashes.update(hh)
        for seed in range(3):models[f'{condition}_{seed}']=mm[f'neural_{seed}']
    protocol=dict(experiment='E55',checkpoints=hashes,manifest_sha256=fingerprint(Path(data)/'manifest.json'),test_evaluated=False,tasks=tasks,
        scope='Training-split procedural arms only. XYZ plus scaled tool-Z tracking with free yaw; no gripper or object. Frozen models; no training, selection or native data.',
        drivers=list(models),horizon=150,sample_steps=[9,29,49,50,79,99,100,129,149],
        targets='Two forward-kinematic endpoints per body: home plus seeded uniform +/- .15 rad offsets, no outcome/contact filtering. Alternate every 50 steps with .002 m positional and approximately .01 rad unit-direction ramps. Cold start, no probes; each of ten drivers produces its own online history. Stop driver at first collision; retain missing later states and failures. Ignore legacy static-goal success termination because task is scheduled tracking.',
        branches='Ten candidates per fixed state, same driver Gram/cross/bias; five steps of continuing scheduled moving reference, online updates and native procedural clipping/limits. Record contact, one/five-step errors. Copy all simulator data and Python state; exact driver replay without branching required.',
        proxies='E53 operator, local unclipped and local executed-command scores for XYZ/tool-Z, scale .15, gain .35 and limits +/-2.5. Privileged geometric Jacobian diagnostic-only; forbidden in candidate inference.',
        statistics='Case-macro pairwise ranking, stratified by DoF and driver family; eight bodies are clusters. Record collision-free pairs separately. No independent task-success claims.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'source').mkdir();(root/'trials').mkdir()
    for name in ['procedural_pose_audit.py','procedural_action_audit.py','action_objective_audit.py','mixed_sim.py','pick_place_control.py','retention_fresh.py','action_objective_report.py']:
        shutil.copy2(Path(__file__).with_name(name),root/'source'/name)
    rows=[];drivers=[];exact_branches=0
    real_jac=mujoco.mj_jacSite
    for task in tasks:
        arm=arms[task['body']]
        for driver_name,driver_model in models.items():
            env=MixedReach(arm,horizon=1000,synchronize=True);env.reset(0,task['goals'][0][:3],spread=0)
            reference=vector(observe(env));controller=GuardedPose(arm.dof,driver_model);trace=[]
            with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
                for step in range(150):
                    obs=observe(env);target=reference_step(reference,task['goals'],step)
                    error=target-vector(obs)
                    selected=step in protocol['sample_steps'];driver_branch=None
                    if selected:
                        jp=np.zeros((3,arm.dof));jr=np.zeros_like(jp);real_jac(env.model,env.data,jp,jr,0)
                        truth=np.concatenate([jp,.15*direction_jacobian(obs['rotation'],jr)])
                        ideal=truth@operator(truth);candidates=[]
                        for name,model in models.items():
                            c=copied(controller,model);jac,_=c.predict();op=operator(jac);motion=op@(.35*error)
                            command=np.clip(env.data.qpos+np.clip(motion+c.bias,-.04,.04),-2.5,2.5)-env.data.qpos
                            clipped=int(np.sum(np.abs(motion+c.bias)>.04))
                            scores=dict(operator_mse=float(np.mean((truth@op-ideal)**2)),local_unclipped=float(np.sum((error-truth@motion)**2)),local_executed=float(np.sum((error-truth@(command-c.bias))**2)))
                            result=branch(env,c,reference.copy(),task['goals'],step)
                            if name==driver_name:driver_branch=result['1']
                            candidates.append(dict(method=name,scores=scores,branch=result,clipped_joints=clipped))
                        rows.append(dict(case=arm.seed,robot=f'{arm.dof}dof',driver=driver_name,phase='tracking',index=step,initial_error_squared=float(error@error),candidates=candidates))
                    reference,info=execute(env,controller,reference,task['goals'],step);trace.append(info)
                    if selected:assert driver_branch==info;exact_branches+=1
                    if info['collision']:break
                # A completely fresh no-branch replay must match every driver step.
                replay=MixedReach(arm,horizon=1000,synchronize=True);replay.reset(0,task['goals'][0][:3],spread=0)
                replay_ref=vector(observe(replay));c=GuardedPose(arm.dof,driver_model)
                for step,expected in enumerate(trace):
                    replay_ref,actual=execute(replay,c,replay_ref,task['goals'],step);assert actual==expected
            record=dict(body=arm.seed,dof=arm.dof,driver=driver_name,steps=len(trace),collision=trace[-1]['collision'],exact_replay=True,samples=sum(r['case']==arm.seed and r['driver']==driver_name for r in rows),trace=trace)
            drivers.append(record);(root/'trials'/f'{arm.seed}_{driver_name}.json').write_text(json.dumps(dict(driver=record,samples=[r for r in rows if r['case']==arm.seed and r['driver']==driver_name])))
            print(arm.seed,driver_name,'steps',len(trace),'collision',record['collision'],'exact',flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'records.json').write_text(json.dumps(rows));(root/'drivers.json').write_text(json.dumps(drivers))
    (root/'audit.json').write_text(json.dumps(dict(driver_replays=len(drivers),exact_first_steps=exact_branches,states=len(rows),branches=10*len(rows),unchanged_weights=True,test_evaluated=False),indent=2))
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['data','parents','training','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))
