"""E41: bounded Cartesian targets and gated native-gripper manipulation."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .real_arm_pick_place import NativePickPlace, endpoint_ik
from .real_arm_reach import load_models
from .real_arms import verify_assets
from .pick_place_control import PoseController, calibrate
from .arm_pick_place import ROT_SCALE
from .benchmark import fingerprint


def sequence(env):
    grasp=.24 if env.name=='piper' else .222
    return [(phase, np.r_[xy,z], aperture, budget) for phase,xy,z,aperture,budget in [
        ('approach',env.xy,.30,0.,220),('descend',env.xy,grasp,0.,260),
        ('close',env.xy,grasp,.035,60),('lift',env.xy,.32,.035,260),
        ('transport',env.target,.32,.035,260),('lower',env.target,grasp+.008,.035,260),
        ('release',env.target,grasp+.008,0.,60),('retreat',env.target,.32,0.,260),
        ('settle',env.target,.32,0.,60)]]


def bounded_error(goal,position,limit=.004):
    error=goal-position
    return error*min(1.,limit/max(float(np.linalg.norm(error)),1e-12))


def rollout(env,controller=None):
    obs=env.observe();trace=[];phases=[];failure=None
    for phase,goal,aperture,budget in sequence(env):
        # The cube can settle below the nominal tool site inside native fingers.
        # Use allowed object-position feedback to avoid driving it into the table.
        if phase in ('transport','lower'):
            offset=obs['position']-env.data.xpos[env.obj]
            goal=np.r_[env.target+offset[:2], .32 if phase=='transport' else .222+offset[2]]
        if phase=='release':goal=release_goal.copy()
        held=0;reached=False;reference=obs['position'].copy()
        for t in range(budget):
            reference+=bounded_error(goal,reference,.002)
            error=np.r_[reference-obs['position'],ROT_SCALE*(np.array([0,0,1])-obs['rotation'][:,2])]
            if controller is None:jac,bias=env.jacobian(),env.privileged_bias()
            else:jac,bias=controller.predict()
            action=np.clip(jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(6),.35*error)+bias,-.04,.04)
            before=obs;obs,command,info=env.step(action,aperture)
            if controller is not None:controller.update(before,obs,command)
            pe=float(np.linalg.norm(goal-obs['position']));re=float(np.linalg.norm(np.array([0,0,1])-obs['rotation'][:,2]))
            pads,_=env.contacts()
            trace.append(dict(phase=phase,position=obs['position'].tolist(),object=env.data.xpos[env.obj].tolist(),
                position_error=pe,rotation_error=re,pads=len(pads),bad=info['bad']))
            held=held+1 if pe<.006 and re<.08 else 0
            if env.bad:failure='forbidden_contact';break
            if held>=8 and phase not in ('close','release','settle'):reached=True;break
        if phase in ('close','release','settle'):reached=held>=8
        phases.append(dict(phase=phase,reached=reached,steps=t+1,goal=goal.tolist()))
        if phase=='lower':release_goal=goal.copy()
        if failure:break
        if not reached:failure='phase_timeout:'+phase;break
        if phase=='close' and len(env.contacts()[0])!=2:failure='no_bilateral_grasp';break
        if phase in ('lift','transport') and len(env.contacts()[0])!=2:failure='lost_grasp:'+phase;break
    obj=env.data.xpos[env.obj].copy();addr=env.model.jnt_dofadr[env.oj];velocity=env.data.qvel[addr:addr+6]
    success=bool(failure is None and env.grasped and env.lifted and not env.bad and
        np.linalg.norm(obj[:2]-env.target)<.025 and abs(obj[2]-.215)<.005 and
        np.linalg.norm(velocity)<.02 and not env.contacts()[0])
    return dict(success=success,failure=failure if failure else (None if success else 'final_placement'),
        grasped=bool(env.grasped),lifted=bool(env.lifted),collision=bool(env.bad),steps=env.steps,
        final_object=obj.tolist(),phases=phases,trace=trace)


def run(assets,models,output,expanded=False,methods='privileged,analytical,neural_0,neural_1,neural_2'):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    manifest=verify_assets(assets);loaded,hashes=load_models(models)
    # Fixed before screening: broader lateral placements, distinct from the pilot.
    candidates={name:([(x,y) for x in xs for y in [-.03,.03,.06]] if expanded else pilot)
        for name,xs,pilot in [('piper',[.205,.23,.255],[(.2,0),(.225,0),(.25,0)]),
            ('panda',[.31,.36,.41],[(.3,0),(.35,-.06),(.4,.04)])]}
    protocol=dict(experiment='E41',revision=3,expanded=expanded,candidates=candidates,transfer_y=.04,base_height=.2,
        checkpoints=hashes,commit=manifest['commit'],methods=methods,seed=41000,
        control='Frozen E35 weights; measured 32-step calibration; Cartesian reference ramps .002 m per step, tracking error uncapped; .35 gain/.015 damping/.04 joint clip. Native mechanics unchanged. Revision 1 bounded-error pilot preserved separately.',
        gates='8 holds within .006 m and .08 tool direction. Stop on any forbidden contact, phase timeout, missing bilateral close grasp, or lost grasp after lift/transport.',
        geometry='Both bases .2 m; approach .30/lift .32 m; grasp PiPER .24/Panda .222; lower grasp+.008. Shared 4 cm transfer.',
        placement='Revision 3: transport/lower targets compensate measured tool-minus-object XYZ offset at phase entry. Lower targets object center .222 m (7 mm above resting height); release holds that goal. Object pose is an allowed simulated state input. Reuses observed development layouts; no independent generalization claim.',
        screening='All nine endpoint IK poses; no policy-success filtering. Every rejection and failure retained.',
        scope='Development controller/fixture change, not weight learning or independent final-test validation.',test_evaluated=False)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'source').mkdir()
    for name in ['native_reliability.py','real_arm_pick_place.py','real_arms.py','pick_place_control.py','motion_transformer.py']:
        shutil.copyfile(Path(__file__).with_name(name),root/'source'/name)
    tasks=[];screening=[];rng=np.random.default_rng(41000)
    for name,points in candidates.items():
        for xy in points:
            target=[xy[0],xy[1]+.04];env=NativePickPlace(assets,name,xy,target,.2,True)
            q=env.observe()['q'];poses=[]
            for _,goal,_,_ in sequence(env):
                q=endpoint_ik(env,goal,rng,q)
                if q is None:break
                poses.append(q.tolist())
            screening.append(dict(robot=name,xy=xy,endpoint_count=len(poses),accepted=len(poses)==9))
            if len(poses)==9:tasks.append(dict(robot=name,xy=xy,target=target,home=poses[0]))
            print('IK',name,xy,len(poses),flush=True)
            (root/'screening.json').write_text(json.dumps(screening,indent=2))
    (root/'cases.json').write_text(json.dumps(tasks,indent=2));rows=[]
    for method in methods.split(','):
        for i,task in enumerate(tasks):
            env=NativePickPlace(assets,task['robot'],task['xy'],task['target'],.2,True)
            controller=PoseController(env.n,loaded[method])
            guard=nullcontext() if method=='privileged' else patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden'))
            with guard:
                probe=calibrate(env,task['home'],controller,41000+i)
                result=rollout(env,None if method=='privileged' else controller)
            rows.append(dict(method=method,case=i,robot=env.name,dof=env.n,probe=probe,**result))
            (root/'records.json').write_text(json.dumps(rows,indent=2))
            print(method,env.name,i,result['success'],result['failure'],result['steps'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','models','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--expanded',action='store_true');p.add_argument('--methods',default='privileged,analytical,neural_0,neural_1,neural_2')
    run(**vars(p.parse_args()))
