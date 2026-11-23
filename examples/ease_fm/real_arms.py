"""Pinned real-design arm adapters; preserve upstream mechanics and actuators."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .arm_pick_place import ROT_SCALE,direction_jacobian

SPECS={
    'piper':dict(folder='agilex_piper',xml='piper.xml',n=6,actuators=[f'joint{i}' for i in range(1,7)],gripper='gripper',
        fingers=['joint7','joint8'],finger_bodies=['link7','link8'],mount='link6',tip='0 0 .105',opening=.07),
    'panda':dict(folder='franka_emika_panda',xml='panda.xml',n=7,actuators=[f'actuator{i}' for i in range(1,8)],gripper='actuator8',
        fingers=['finger_joint1','finger_joint2'],finger_bodies=['left_finger','right_finger'],mount='hand',tip='0 0 .1034',opening=.08)}


def verify_assets(assets):
    root=Path(assets);manifest=json.loads((root/'MANIFEST.json').read_text())
    for entry in manifest['files']:
        assert hashlib.sha256((root/entry['path']).read_bytes()).hexdigest()==entry['sha256'],entry['path']
    return manifest


@lru_cache(maxsize=16)
def load_model(assets,name,object_xy=None,target_xy=None,base_height=0.):
    spec=SPECS[name];folder=Path(assets).resolve()/spec['folder'];root=ET.parse(folder/spec['xml']).getroot()
    root.find('compiler').set('meshdir',str(folder/'assets'))
    world=root.find('worldbody')
    if base_height:
        base=root.find(f".//body[@name='{'base_link' if name=='piper' else 'link0'}']")
        base.set('pos',f'0 0 {base_height}')
    ET.SubElement(world,'geom',name='ease_floor',type='plane',pos='0 0 -.005',size='2 2 .1',rgba='.4 .45 .5 1')
    mount=root.find(f".//body[@name='{spec['mount']}']")
    ET.SubElement(mount,'site',name='ease_tool',pos=spec['tip'],quat='0 1 0 0',size='.005',rgba='0 1 0 1')
    if object_xy is not None:
        ET.SubElement(world,'geom',name='ease_table',type='box',pos='.275 0 .175' if base_height else '.4 0 .175',size='.375 .25 .025' if base_height else '.22 .25 .025',rgba='.6 .65 .7 1')
        obj=ET.SubElement(world,'body',name='ease_object',pos=f'{object_xy[0]} {object_xy[1]} .2152')
        ET.SubElement(obj,'freejoint',name='ease_object_joint')
        ET.SubElement(obj,'geom',name='ease_cube',type='box',size='.015 .015 .015',mass='.04',friction='1 .01 .001',condim='4',rgba='1 .4 .1 1')
        ET.SubElement(world,'site',name='ease_target',type='cylinder',pos=f'{target_xy[0]} {target_xy[1]} .201',size='.025 .001',rgba='0 1 0 .5')
        # Extend original keyframes for the added free object, preserving robot values.
        for key in root.findall('keyframe/key'):
            key.set('qpos',key.get('qpos')+f' {object_xy[0]} {object_xy[1]} .2152 1 0 0 0')
    xml=ET.tostring(root,encoding='unicode')
    return mujoco.MjModel.from_xml_string(xml),xml


class RealArm:
    def __init__(self,assets,name,object_xy=None,target_xy=None,base_height=0.):
        self.name=name;self.spec=SPECS[name];self.n=self.spec['n']
        self.model,self.xml=load_model(str(Path(assets).resolve()),name,
            None if object_xy is None else tuple(object_xy),None if target_xy is None else tuple(target_xy),base_height)
        self.data=mujoco.MjData(self.model)
        self.joints=np.array([self.id(mujoco.mjtObj.mjOBJ_JOINT,f'joint{i}') for i in range(1,self.n+1)])
        self.qids=self.model.jnt_qposadr[self.joints];self.vids=self.model.jnt_dofadr[self.joints]
        self.aids=np.array([self.id(mujoco.mjtObj.mjOBJ_ACTUATOR,a) for a in self.spec['actuators']])
        self.gripper=self.id(mujoco.mjtObj.mjOBJ_ACTUATOR,self.spec['gripper'])
        self.site=self.id(mujoco.mjtObj.mjOBJ_SITE,'ease_tool')
        self.limits=np.stack([np.maximum(self.model.jnt_range[self.joints,0],self.model.actuator_ctrlrange[self.aids,0]),
            np.minimum(self.model.jnt_range[self.joints,1],self.model.actuator_ctrlrange[self.aids,1])],-1)
        self.substeps=round(.04/self.model.opt.timestep);assert abs(self.substeps*self.model.opt.timestep-.04)<1e-8
        self.reset()

    def id(self,kind,name):
        result=mujoco.mj_name2id(self.model,kind,name)
        if result<0:raise ValueError(name)
        return result

    def reset(self,q=None,openness=1.):
        key=self.id(mujoco.mjtObj.mjOBJ_KEY,'home');mujoco.mj_resetDataKeyframe(self.model,self.data,key)
        if q is not None:self.data.qpos[self.qids]=np.clip(q,self.limits[:,0],self.limits[:,1])
        self.data.ctrl[self.aids]=self.data.qpos[self.qids]
        # Reset gripper positions consistently with its coupling, not by actuator count.
        for fname in self.spec['fingers']:
            jid=self.id(mujoco.mjtObj.mjOBJ_JOINT,fname);lo,hi=self.model.jnt_range[jid]
            self.data.qpos[self.model.jnt_qposadr[jid]]=(lo if hi==0 else hi)*openness
        lo,hi=self.model.actuator_ctrlrange[self.gripper];self.data.ctrl[self.gripper]=lo+openness*(hi-lo)
        mujoco.mj_forward(self.model,self.data);self.steps=0
        return self.observe()

    def observe(self):
        return dict(q=self.data.qpos[self.qids].copy(),position=self.data.site_xpos[self.site].copy(),
            rotation=self.data.site_xmat[self.site].reshape(3,3).copy())

    def step(self,delta,openness=1.):
        delta=np.asarray(delta);assert delta.shape==(self.n,) and np.isfinite(delta).all()
        q=self.data.qpos[self.qids].copy();target=np.clip(q+np.clip(delta,-.04,.04),self.limits[:,0],self.limits[:,1])
        self.data.ctrl[self.aids]=target
        lo,hi=self.model.actuator_ctrlrange[self.gripper];self.data.ctrl[self.gripper]=lo+np.clip(openness,0,1)*(hi-lo)
        contacts=set()
        for _ in range(self.substeps):
            mujoco.mj_step(self.model,self.data)
            contacts.update(tuple(sorted([int(c.geom1),int(c.geom2)])) for c in self.data.contact[:self.data.ncon])
        mujoco.mj_forward(self.model,self.data);self.steps+=1
        assert np.isfinite(self.data.qpos).all()
        return self.observe(),target-q,contacts

    def jacobian(self):
        jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp);mujoco.mj_jacSite(self.model,self.data,jp,jr,self.site)
        return np.concatenate([jp[:,self.vids],ROT_SCALE*direction_jacobian(self.observe()['rotation'],jr[:,self.vids])])

    def opening(self):
        return sum(abs(self.data.qpos[self.model.jnt_qposadr[self.id(mujoco.mjtObj.mjOBJ_JOINT,j)]]) for j in self.spec['fingers'])
