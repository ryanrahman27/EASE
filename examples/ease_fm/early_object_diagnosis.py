"""E175 fitting-only pre-intervention state inspection."""
from pathlib import Path
import numpy as np
import mujoco
from .abstaining_task_value_gate import read,save
from .object_preserving_plan import forbidden,plan,geometry_ids
ROOT=Path('outputs/ease_early_object_recovery_v1')
REFERENCE=Path('outputs/ease_object_preserving_v1/guarded_handoff')

def object_clearance(model,data):
    cube=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'cube')
    robot,_=geometry_ids(model,6,True)
    assert len(robot)>=9
    return min(float(mujoco.mj_geomDistance(model,data,i,cube,1.,None)) for i in robot)

def pad_separation(model,data):
    """SAT separating-axis gap, a conservative distance lower bound for boxes."""
    cube=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'cube');b=data.geom_xmat[cube].reshape(3,3);values=[]
    for name in ('gripper_left_pad','gripper_right_pad'):
        g=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name);a=data.geom_xmat[g].reshape(3,3);axes=[*a.T,*b.T]
        axes += [np.cross(x,y) for x in a.T for y in b.T];delta=data.geom_xpos[g]-data.geom_xpos[cube];gaps=[]
        for axis in axes:
            norm=np.linalg.norm(axis)
            if norm<1e-9:continue
            axis=axis/norm;gaps.append(abs(delta@axis)-np.abs(a.T@axis)@model.geom_size[g]-np.abs(b.T@axis)@model.geom_size[cube])
        values.append(max(gaps))
    return float(min(values))

def run():
    ROOT.mkdir(exist_ok=True);rows=[];details=[]
    for r in read(REFERENCE/'records.json'):
        case=r['case'];stem=f'{case}_e120_ensemble_frozen_bias';logs=read(REFERENCE/'clearance'/f'{stem}.json');m=mujoco.MjModel.from_xml_path(str(REFERENCE/'states'/f'{stem}.xml'));d=mujoco.MjData(m);es=[];series=[]
        for i,l in enumerate(logs):
            if l['phase']!='descend':continue
            d.qpos[:]=l['full_qpos_before'];d.qvel[:]=l['full_qvel_before'];mujoco.mj_forward(m,d);site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'object');pe=float(np.linalg.norm(np.array(l['goal'])-d.site_xpos[site]));es.append(pe)
            x=dict(step=i,descent=len(es),error=pe,axis=float(np.linalg.norm(d.site_xmat[site].reshape(3,3)[:,2]-[0,0,1])),pad_gap=pad_separation(m,d),qmax=float(np.max(np.abs(d.qpos[:6]))),clearance=object_clearance(m,d),contacts=forbidden(m,d),object=d.xpos[obj].tolist(),progress10=None if len(es)<10 else es[-10]-pe,route=l['route_active']);series.append(x)
        rows.append(dict(case=case,success=r['success'],first_contact=next((x for x in series if x['contacts']),None),first_route=next((x for x in series if x['route']),None)))
        details.append(dict(case=case,series=series))
        if case==26:print([x for x in series[:65] if x['descent']%5==0 or x['contacts']],flush=True)
    save(ROOT/'reference_diagnosis.json',rows);save(ROOT/'reference_descent_states.json',details)

if __name__=='__main__':run()
