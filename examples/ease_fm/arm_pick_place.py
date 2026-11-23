"""E37: physical arm-driven manipulation pilot and matched pose controllers."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .mixed_sim import Arm
from .paired_data import register
from .pick_place_fixture import arm_fixture_xml

ROT_SCALE=.15


def rotation_error(target,current):
    # Downward fingers: local +Z points up. Yaw is unconstrained.
    return target[:,2]-current[:,2]


def direction_jacobian(rotation,angular):
    z=rotation[:,2]
    skew=np.array([[0,-z[2],z[1]],[z[2],0,-z[0]],[-z[1],z[0],0]])
    return -skew@angular


def scene(arm,xy,target):
    root=ET.fromstring(arm_fixture_xml(arm,xy,target))
    # Fingers must extend away from the last link, not back along its capsule.
    root.find(".//body[@name='gripper_mount']").set('quat','0 1 0 0')
    world=root.find('worldbody')
    ET.SubElement(world,'light',pos='0 -.5 1.5',dir='0 0 -1',diffuse='1 1 1',ambient='.3 .3 .3')
    root.find(".//geom[@name='cube']").set('rgba','1 .45 .1 1')
    return ET.tostring(root,encoding='unicode')


class PickPlace:
    def __init__(self,arm,xy=(.35,0),target=(.35,.08)):
        self.arm=arm;self.n=arm.dof;self.xy=np.array(xy);self.target=np.array(target)
        self.xml=scene(arm,xy,target);self.model=mujoco.MjModel.from_xml_string(self.xml);self.data=mujoco.MjData(self.model)
        self.site=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,'grasp_center')
        self.obj=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_BODY,'object')
        self.gids={name:mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,name) for name in ['cube','table','gripper_left_pad','gripper_right_pad']}
        self.oj=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_JOINT,'object_free')
        self.bad=False;self.lifted=False;self.grasped=False;self.steps=0

    def observe(self):
        return dict(q=self.data.qpos[:self.n].copy(),position=self.data.site_xpos[self.site].copy(),
            rotation=self.data.site_xmat[self.site].reshape(3,3).copy())

    def jacobian(self):
        p=np.zeros((3,self.model.nv));r=np.zeros_like(p)
        mujoco.mj_jacSite(self.model,self.data,p,r,self.site)
        rotation=self.data.site_xmat[self.site].reshape(3,3)
        return np.concatenate([p[:,:self.n],ROT_SCALE*direction_jacobian(rotation,r[:,:self.n])])

    def contacts(self):
        pads=set();bad=[]
        for c in self.data.contact[:self.data.ncon]:
            pair={int(c.geom1),int(c.geom2)}
            if self.gids['cube'] in pair:
                if self.gids['table'] in pair:continue
                for pad in ['gripper_left_pad','gripper_right_pad']:
                    if self.gids[pad] in pair:pads.add(pad)
                if pair.intersection({self.gids['gripper_left_pad'],self.gids['gripper_right_pad']}):continue
            bad.append([mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,g) or str(g) for g in pair])
        return pads,bad

    def reset(self,q):
        mujoco.mj_resetData(self.model,self.data);self.data.qpos[:self.n]=q;self.data.ctrl[:self.n]=q
        mujoco.mj_forward(self.model,self.data);self.bad=False;self.lifted=False;self.grasped=False;self.steps=0
        return self.observe()

    def step(self,action,aperture):
        before=self.observe();target=np.clip(before['q']+np.clip(action,-.04,.04),-2.5,2.5)
        executed=target-before['q'];self.data.ctrl[:self.n]=target;self.data.ctrl[self.n:]=aperture
        pads=set();bad=[]
        for _ in range(20):
            mujoco.mj_step(self.model,self.data)
            touched,invalid=self.contacts();pads|=touched;bad+=invalid
            self.grasped |= len(touched)==2
            self.lifted |= len(touched)==2 and self.data.xpos[self.obj,2]>.275
        mujoco.mj_forward(self.model,self.data);self.steps+=1;self.bad |= bool(bad)
        assert np.isfinite(self.data.qpos).all()
        return self.observe(),executed,dict(pads=len(pads),bad=bad)


def ik(env,position,rng,start=None,attempts=30):
    best=None
    for attempt in range(attempts):
        q=np.array(start) if attempt==0 and start is not None else rng.uniform(-1.8,1.8,env.n)
        for _ in range(220):
            env.reset(q);obs=env.observe()
            error=np.r_[position-obs['position'],ROT_SCALE*rotation_error(np.eye(3),obs['rotation'])]
            if np.linalg.norm(error[:3])<.001 and np.linalg.norm(error[3:])<.003:
                if not env.contacts()[1]:return q
                break
            jac=env.jacobian();delta=jac.T@np.linalg.solve(jac@jac.T+.005**2*np.eye(6),error)
            q=np.clip(q+np.clip(delta,-.15,.15),-2.4,2.4)
            best=float(np.linalg.norm(error))
    return None


def targets(xy,target):
    return [('approach',np.r_[xy,.33],0.,80),('descend',np.r_[xy,.24],0.,100),
            ('close',np.r_[xy,.24],.035,40),('lift',np.r_[xy,.35],.035,100),
            ('transport',np.r_[target,.35],.035,100),('lower',np.r_[target,.245],.035,100),
            ('release',np.r_[target,.245],0.,40),('retreat',np.r_[target,.35],0.,80),
            ('settle',np.r_[target,.35],0.,40)]


def rollout(env,home,controller=None,render=None,reset=True,sequence=None):
    obs=env.reset(home) if reset else env.observe();trace=[];phases=[];renderer=None
    if render:
        from PIL import Image
        Path(render).mkdir(parents=True,exist_ok=True);renderer=mujoco.Renderer(env.model,height=480,width=640)
        camera=mujoco.MjvCamera();camera.lookat[:]=[.25,0,.35];camera.distance=1.5;camera.azimuth=130;camera.elevation=-25
    for phase,goal,aperture,budget in (targets(env.xy,env.target) if sequence is None else sequence):
        held=0;reached=False
        for t in range(budget):
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
            if held>=5 and phase not in ('close','release','settle'):
                reached=True;break
        phases.append(dict(phase=phase,reached=reached or (pe<.008 and re<.08),steps=t+1))
        if renderer:
            renderer.update_scene(env.data,camera=camera);Image.fromarray(renderer.render()).save(Path(render)/f'{phase}.png')
    if renderer:renderer.close()
    obj=env.data.xpos[env.obj].copy();addr=env.model.jnt_dofadr[env.oj];velocity=env.data.qvel[addr:addr+6]
    success=bool(env.grasped and env.lifted and not env.bad and np.linalg.norm(obj[:2]-env.target)<.025 and abs(obj[2]-.215)<.005 and np.linalg.norm(velocity)<.02 and not env.contacts()[0])
    return dict(success=success,grasped=bool(env.grasped),lifted=bool(env.lifted),collision=bool(env.bad),steps=env.steps,final_object=obj.tolist(),phases=phases,trace=trace)


def build(data,output,split='train',families_per_dof=2):
    manifest=register(data);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    (root/'protocol.json').write_text(json.dumps(dict(experiment='E37',scope='Development feasibility; three fixed cube positions, 6 cm lateral transfer, endpoint IK filtering only; all rollout outcomes retained',split=split,families_per_dof=families_per_dof,selection='First distinct families in manifest order per joint count; first body per family, no success selection',seed=37000,orientation='Downward local Z direction; yaw unconstrained',test_evaluated=False),indent=2))
    shutil.copyfile(Path(__file__),root/'source.py');rng=np.random.default_rng(37000);records=[];cases=[]
    # One body per distinct training family, kept independent of E33 validation.
    seen=set();counts={6:0,7:0}
    for raw in manifest['arms']:
        if raw['split']!=split or raw['family'] in seen:continue
        arm=Arm(**raw)
        if counts[arm.dof]>=families_per_dof:continue
        seen.add(raw['family']);counts[arm.dof]+=1
        for xy in [(.35,0),(.3,.08),(.4,-.08)]:
            target=(xy[0],xy[1]+.06);env=PickPlace(arm,xy,target)
            poses=[];start=None
            for _,position,_,_ in targets(xy,target):
                q=ik(env,position,rng,start)
                if q is None:break
                poses.append(q.tolist());start=q
            entry=dict(arm=raw,xy=xy,target=target,ik_count=len(poses))
            if len(poses)==9:
                entry['home']=poses[0];result=rollout(env,poses[0]);entry['result']=result
                cases.append(dict(arm=raw,xy=xy,target=target,home=poses[0]))
                print(arm.family,xy,'success',result['success'],'grasp',result['grasped'],'lift',result['lifted'],'collision',result['collision'],flush=True)
            else:print(arm.family,xy,'IK failed',len(poses),flush=True)
            records.append(entry);(root/'records.json').write_text(json.dumps(records,indent=2))
    (root/'cases.json').write_text(json.dumps(cases,indent=2));(root/'axis_catalog.json').write_text(json.dumps(manifest['axis_catalog'],indent=2))
    (root/'status.json').write_text(json.dumps(dict(status='complete',scope='Development feasibility pilot; endpoint IK screening; all rollout failures retained',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split',default='train');p.add_argument('--families-per-dof',type=int,default=2);build(**vars(p.parse_args()))
