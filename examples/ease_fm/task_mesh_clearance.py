"""Audit rendered gripper vertices against the task table and floor."""
import json
from pathlib import Path
import numpy as np
import mujoco

def audit(path,states_path):
    m=mujoco.MjModel.from_xml_path(str(path.resolve()));d=mujoco.MjData(m);states=np.load(states_path)['states'];robot='piperx' if 'piperx' in path.name else 'openarm'
    mount=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'Link6' if robot=='piperx' else 'openarm_link7')
    def descendant(b):
        while b:
            if b==mount:return True
            b=int(m.body_parentid[b])
        return False
    meshes=[]
    for g in range(m.ngeom):
        if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH or m.geom_group[g]==3 or not descendant(int(m.geom_bodyid[g])):continue
        mid=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid]+m.mesh_vertnum[mid]].copy();meshes.append((g,v))
    worst=dict(clearance=100.);minfloor=100.;bygeom={}
    for i,state in enumerate(states):
        d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_forward(m,d)
        for g,v in meshes:
            world=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g];minfloor=min(minfloor,float(world[:,2].min()+.005))
            mask=(world[:,0]>=0)&(world[:,0]<=.6)&(np.abs(world[:,1])<=.24)
            if not mask.any():continue
            clearance=float(world[mask,2].min()-.20);name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))
            bygeom[name]=min(bygeom.get(name,100.),clearance)
            if clearance<worst['clearance']:worst=dict(clearance=clearance,step=i,body=name)
    return dict(robot=robot,minimum_table_clearance=worst,minimum_floor_clearance=minfloor,by_body=bygeom,states=len(states))

if __name__=='__main__':
    root=Path('outputs/ease_known_arm_task_demo_v1')
    for name,i in [('piperx',3),('openarm',2)]:print(json.dumps(audit(root/f'{name}_attempt{i}.xml',root/f'{name}_attempt{i}_states.npz'),indent=2))
