"""E40 native PiPER/Panda manipulation pilot with frozen measured-pose control."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from .real_arms import RealArm,verify_assets
from .real_arm_reach import load_models
from .arm_pick_place import ROT_SCALE,rollout
from .pick_place_control import PoseController,calibrate
from .benchmark import fingerprint


class NativePickPlace(RealArm):
    def __init__(self,assets,name,xy,target,base_height=0.,compact=False):
        self.compact=compact
        self.xy=np.array(xy);self.target=np.array(target)
        super().__init__(assets,name,xy,target,base_height)
        self.obj=self.id(mujoco.mjtObj.mjOBJ_BODY,'ease_object');self.oj=self.id(mujoco.mjtObj.mjOBJ_JOINT,'ease_object_joint')
        self.cube=self.id(mujoco.mjtObj.mjOBJ_GEOM,'ease_cube');self.table=self.id(mujoco.mjtObj.mjOBJ_GEOM,'ease_table')
        self.finger_bodies=[self.id(mujoco.mjtObj.mjOBJ_BODY,b) for b in self.spec['finger_bodies']]

    def reset(self,q=None,openness=1.):
        obs=super().reset(q,openness);self.bad=False;self.grasped=False;self.lifted=False
        return obs

    def classify(self,pairs):
        pads=set();bad=[]
        for pair in pairs:
            if self.cube in pair:
                other=next(g for g in pair if g!=self.cube)
                if other==self.table:continue
                body=int(self.model.geom_bodyid[other])
                if body in self.finger_bodies:pads.add(body);continue
            bad.append([mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,g) or str(g) for g in pair])
        return pads,bad

    def contacts(self):
        return self.classify([tuple(sorted([int(c.geom1),int(c.geom2)])) for c in self.data.contact[:self.data.ncon]])

    def step(self,action,aperture):
        # Legacy sequencer 0=open, .035=closed; native adapter uses normalized openness.
        obs,command,pairs=super().step(action,1-np.clip(aperture/.035,0,1))
        pads,bad=self.classify(pairs);self.bad |= bool(bad)
        current,_=self.contacts();self.grasped |= len(current)==2
        self.lifted |= len(current)==2 and self.data.xpos[self.obj,2]>.275
        return obs,command,dict(pads=len(pads),bad=bad)

    def privileged_bias(self):
        return (self.data.qfrc_bias[self.vids]-self.data.qfrc_passive[self.vids])/self.model.actuator_gainprm[self.aids,0]

    def sequence(self):
        # Geometry-calibrated tool centers: finger/table clearance differs by gripper.
        grasp=.24 if self.name=='piper' else .22
        approach=.30 if self.compact else .33;lift=.32 if self.compact else .35
        return [('approach',np.r_[self.xy,approach],0.,100),('descend',np.r_[self.xy,grasp],0.,120),
            ('close',np.r_[self.xy,grasp],.035,60),('lift',np.r_[self.xy,lift],.035,120),
            ('transport',np.r_[self.target,lift],.035,120),('lower',np.r_[self.target,grasp+.005],.035,120),
            ('release',np.r_[self.target,grasp+.005],0.,60),('retreat',np.r_[self.target,lift],0.,100),
            ('settle',np.r_[self.target,lift],0.,60)]


def endpoint_ik(env,goal,rng,start):
    for attempt in range(16):
        q=start.copy() if attempt==0 else rng.uniform(env.limits[:,0]+.03,env.limits[:,1]-.03)
        for _ in range(200):
            obs=env.reset(q);error=np.r_[goal-obs['position'],ROT_SCALE*(np.array([0,0,1])-obs['rotation'][:,2])]
            if np.linalg.norm(error[:3])<.001 and np.linalg.norm(error[3:])<.003:
                if not env.contacts()[1]:return q
                break
            jac=env.jacobian();delta=jac.T@np.linalg.solve(jac@jac.T+.005**2*np.eye(6),error)
            q=np.clip(q+np.clip(delta,-.1,.1),env.limits[:,0]+.001,env.limits[:,1]-.001)
    return None


def run(assets,models,output,base_height=0.,compact=False,robots='piper,panda'):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    manifest=verify_assets(assets);loaded,hashes=load_models(models);rng=np.random.default_rng(40000)
    candidates=[(.2,0),(.225,0),(.25,0)] if compact else [(.3,0),(.35,-.06),(.4,.04)]
    transfer=.04 if compact else .06
    (root/'protocol.json').write_text(json.dumps(dict(experiment='E40',commit=manifest['commit'],checkpoints=hashes,base_height=base_height,compact=compact,robots=robots,
        candidates=candidates,transfer_y=transfer,screening='Endpoint IK only, retain failures; no policy-success filtering. Compact variant is post-workspace-diagnostic development, approach .30/lift .32; not comparable as the same task to earlier pilots.',
        native='Upstream model mechanics/actuators/compensation preserved. Tool center piper link6 z=.105; panda hand z=.1034, tool +Z away from fingers. Native grippers map normalized openness.',
        control='Same frozen E35 estimators and common scripted sequencer as E38. Grasp-center height .24 for PiPER, .22 for Panda based on gripper depth; lower+.005. Same 32-step bounded probe for every method; no reset thereafter.',
        scope='First development integration with real robot designs in simulation; not hardware or broad transfer validation.',test_evaluated=False),indent=2))
    (root/'source').mkdir()
    for name in ['real_arms.py','real_arm_pick_place.py','arm_pick_place.py','pick_place_control.py']:shutil.copyfile(Path(__file__).with_name(name),root/'source'/name)
    tasks=[];screening=[]
    for name in robots.split(','):
        for xy in candidates:
            target=(xy[0],xy[1]+transfer);env=NativePickPlace(assets,name,xy,target,base_height,compact);q=env.observe()['q'];poses=[]
            for phase,goal,_,_ in env.sequence():
                q=endpoint_ik(env,goal,rng,q)
                if q is None:break
                poses.append(q.tolist())
            screening.append(dict(robot=name,xy=xy,endpoint_count=len(poses),accepted=len(poses)==9))
            if len(poses)==9:tasks.append(dict(robot=name,xy=xy,target=target,home=poses[0]))
            print('IK',name,xy,len(poses),flush=True)
    (root/'screening.json').write_text(json.dumps(screening,indent=2));(root/'cases.json').write_text(json.dumps(tasks,indent=2))
    rows=[]
    for method,model in loaded.items():
        for i,task in enumerate(tasks):
            env=NativePickPlace(assets,task['robot'],task['xy'],task['target'],base_height,compact);controller=PoseController(env.n,model)
            guard=nullcontext() if method=='privileged' else patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden'))
            with guard:
                probe=calibrate(env,task['home'],controller,40000+i)
                result=rollout(env,task['home'],None if method=='privileged' else controller,reset=False,sequence=env.sequence())
            rows.append(dict(method=method,case=i,robot=env.name,dof=env.n,probe=probe,**result))
            (root/'records.json').write_text(json.dumps(rows,indent=2))
            print(env.name,method,i,result['success'],'grasp',result['grasped'],'lift',result['lifted'],'collision',result['collision'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','models','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--base-height',type=float,default=0.)
    p.add_argument('--compact',action='store_true');p.add_argument('--robots',default='piper,panda')
    run(**vars(p.parse_args()))
