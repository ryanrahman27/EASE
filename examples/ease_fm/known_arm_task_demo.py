"""Development physical task demo; native visual/kinematic models, simplified pads/servos."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
from .pick_place_control import PoseController
from .benchmark import fingerprint

ROOT=Path('outputs/ease_known_arm_task_demo_v1')
SOURCE=Path('outputs/ease_known_arm_demo_v2/models')

def build(name):
    root=ET.parse(SOURCE/f'{name}.xml').getroot();n=6 if name=='piperx' else 7
    for tag in ('actuator','keyframe','equality','contact'):
        for e in root.findall(tag):root.remove(e)
    opt=root.find('option')
    if opt is None:opt=ET.SubElement(root,'option')
    opt.set('timestep','.002');opt.set('integrator','implicitfast')
    for b in root.findall('.//body'):b.set('gravcomp','1')
    for g in root.findall('.//geom'):g.set('contype','0');g.set('conaffinity','0')
    arm_names=[f'joint{i}' if name=='piperx' else f'openarm_joint{i}' for i in range(1,n+1)]
    fingers=['joint7','joint8'] if name=='piperx' else ['openarm_finger_joint1','openarm_finger_joint2']
    act=ET.SubElement(root,'actuator')
    for j in arm_names:
        ET.SubElement(act,'position',name='servo_'+j,joint=j,kp='600',kv='35',forcerange='-200 200',forcelimited='true')
        elem=root.find(f'.//joint[@name="{j}"]');elem.set('damping','2');elem.set('armature','.02')
    for j in fingers:
        ET.SubElement(act,'position',name='servo_'+j,joint=j,kp='1500',kv='30',forcerange='-30 30',forcelimited='true')
        if name=='piperx':root.find(f'.//joint[@name="{j}"]').set('armature','.1')
    base=root.find('.//body[@name="base_link"]' if name=='piperx' else './/body[@name="openarm_link0"]');base.set('pos','0 0 .2')
    mount=root.find('.//body[@name="Link6"]' if name=='piperx' else './/body[@name="openarm_hand"]')
    ET.SubElement(mount,'site',name='task_tcp',pos='0 0 .08' if name=='piperx' else '0 0 .065',quat='0 1 0 0',size='.002',rgba='0 1 0 0')
    for i,bname in enumerate(['Link7','Link8'] if name=='piperx' else ['openarm_right_finger','openarm_left_finger']):
        b=root.find(f'.//body[@name="{bname}"]');pos=('0 -.055 -.004' if i==0 else '0 .055 -.004') if name=='piperx' else '0 .002 .05'
        ET.SubElement(b,'geom',name=f'pad{i}',type='box',pos=pos,size='.012 .012 .004' if name=='piperx' else '.012 .004 .012',contype='1',conaffinity='1',friction='1.5 .01 .001',condim='4',rgba='.12 .15 .1 1',solref='.008 1',solimp='.95 .99 .001')
    world=root.find('worldbody');xy=[.25,-.04] if name=='piperx' else [.28,-.04];target=[xy[0],xy[1]+.10]
    ET.SubElement(world,'geom',name='table',type='box',pos='.30 0 .185',size='.30 .24 .015',rgba='.10 .12 .10 1',contype='1',conaffinity='1',friction='1 .01 .001')
    ET.SubElement(world,'geom',name='pedestal',type='cylinder',pos='0 0 .10',size='.075 .10',rgba='.07 .08 .07 1',contype='0',conaffinity='0')
    cube=ET.SubElement(world,'body',name='cube',pos=f'{xy[0]} {xy[1]} .216')
    ET.SubElement(cube,'freejoint',name='object_joint');ET.SubElement(cube,'geom',name='object',type='box',size='.015 .015 .015',mass='.035',rgba='.58 .86 .05 1',contype='1',conaffinity='1',friction='1.5 .01 .001',condim='4')
    ET.SubElement(world,'site',name='goal',type='cylinder',pos=f'{target[0]} {target[1]} .201',size='.03 .001',rgba='.6 .86 .1 .5')
    ET.SubElement(world,'geom',name='floor',type='plane',size='3 3 .1',pos='0 0 -.005',rgba='.025 .03 .025 1',contype='0',conaffinity='0')
    ET.SubElement(world,'light',pos='1 -1 3',dir='-1 1 -3',diffuse='.7 .7 .7')
    visual=root.find('visual')
    if visual is None:visual=ET.SubElement(root,'visual')
    ET.SubElement(visual,'global',offwidth='640',offheight='480')
    path=ROOT/f'{name}.xml';ET.ElementTree(root).write(path)
    return path,n,arm_names,fingers,xy,target

def run(name,method='analytical'):
    ROOT.mkdir(exist_ok=True);path,n,arm_names,fingers,xy,target=build(name);m=mujoco.MjModel.from_xml_path(str(path.resolve()));d=mujoco.MjData(m)
    ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in arm_names]);qi=m.jnt_qposadr[ids];vi=m.jnt_dofadr[ids];limits=m.jnt_range[ids];site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'task_tcp');obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'cube')
    fi=[m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,j)] for j in fingers]
    def obs():return dict(q=d.qpos[qi].copy(),position=d.site_xpos[site].copy(),rotation=d.site_xmat[site].reshape(3,3).copy())
    def jac():
        jp=np.zeros((3,m.nv));jr=np.zeros_like(jp);mujoco.mj_jacSite(m,d,jp,jr,site);z=obs()['rotation'][:,2]
        return np.concatenate([jp[:,vi],.15*np.cross(jr[:,vi].T,z).T])
    def err(goal):
        o=obs();return np.r_[goal-o['position'],.15*(np.array([0.,0.,1.])-o['rotation'][:,2])]
    rng=np.random.default_rng(11401);home=None
    for attempt in range(40):
        q=np.array([0,1.,-1.15,0,.3,0]) if name=='piperx' and attempt==0 else (np.array([0,0,0,1.57,0,0,0]) if attempt==0 else rng.uniform(limits[:,0],limits[:,1]))
        for _ in range(350):
            d.qpos[qi]=q;mujoco.mj_forward(m,d);e=err(np.r_[xy,.34])
            if np.linalg.norm(e)<.001:home=q.copy();break
            j=jac();delta=j.T@np.linalg.solve(j@j.T+.01**2*np.eye(6),e);q=np.clip(q+np.clip(delta,-.1,.1),limits[:,0]+.005,limits[:,1]-.005)
        if home is not None:break
    assert home is not None,'no IK home'
    mujoco.mj_resetData(m,d);d.qpos[qi]=home;d.qpos[fi]=[.035,-.035] if name=='piperx' else [.035,.035];d.ctrl[:n]=home;d.ctrl[n:]=d.qpos[fi];mujoco.mj_forward(m,d)
    # Geometry prior at initialization only. Online estimator thereafter uses measured transitions.
    model=None;checkpoint=None
    if method=='neural':
        import torch
        from .unseen_progress import models
        torch.set_num_threads(4);_,students,_=models();model=students[('retained',0)]
        checkpoint='outputs/ease_task_retention_training_v1/retained/seed_0/policy.pt'
    c=PoseController(n,model);c.gram=np.eye(n)*.0002;c.cross=c.gram@(.8*jac()).T
    initial_j=c.predict()[0].copy()
    records=[];states=[];grasp=False;lift=False;phases=[]
    plan=[('calibrate',np.r_[xy,.34],True,48),('descend',np.r_[xy,.217],True,180),('close',np.r_[xy,.217],False,35),('lift',np.r_[xy,.34],False,180),('transfer',np.r_[target,.34],False,180),('lower',np.r_[target,.219],False,180),('release',np.r_[target,.219],True,35),('retreat',np.r_[target,.34],True,180),('settle',np.r_[target,.34],True,40)]
    for phase,goal,opened,budget in plan:
        o=obs();reference=o['position'].copy();hold=o['q'].copy();held=0
        if phase in ('transfer','lower'):
            offset=o['position']-d.xpos[obj];goal=np.r_[np.array(target)+offset[:2],.34 if phase=='transfer' else .217+offset[2]]
        if phase=='release':goal=release_goal.copy()
        for k in range(budget):
            o=obs();j,bias=c.predict();before_j=j.copy();e=err(goal)
            if phase=='calibrate':
                if k%3==0:probe=rng.uniform(-.012,.012,n)
                action=np.clip(home+probe-o['q'],-.03,.03)
            elif phase in ('close','release','settle'):action=np.clip(hold-o['q'],-.03,.03)
            else:
                diff=goal-reference;reference+=diff*min(1,.0025/max(np.linalg.norm(diff),1e-12));e=err(reference)
                action=np.clip(j.T@np.linalg.solve(j@j.T+.025**2*np.eye(6),.5*e)+bias,-.035,.035)
            targetq=np.clip(o['q']+action,limits[:,0],limits[:,1]);d.ctrl[:n]=targetq
            d.ctrl[n:]=([.035,-.035] if name=='piperx' else [.035,.035]) if opened else [0,0]
            for _ in range(20):mujoco.mj_step(m,d)
            mujoco.mj_forward(m,d);after=obs();dq=after['q']-o['q'];dx=np.r_[after['position']-o['position'],.15*(after['rotation'][:,2]-o['rotation'][:,2])]
            prediction_error=float(np.linalg.norm(dx-before_j@dq));c.update(o,after,targetq-o['q']);new_j,_=c.predict()
            pads=set();pairs=[]
            for con in d.contact[:d.ncon]:
                names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in (con.geom1,con.geom2)]
                pairs.append(names)
                if 'object' in names:
                    pads.update(x for x in names if x in ('pad0','pad1'))
            grasp|=len(pads)==2;lift|=len(pads)==2 and d.xpos[obj,2]>.28
            records.append(dict(phase=phase,time=len(records)*.04,tool=after['position'].tolist(),object=d.xpos[obj].tolist(),pads=len(pads),pairs=pairs,history=len(records)+1,prediction_error=prediction_error,position_prediction_error=float(np.linalg.norm(dx[:3]-(before_j@dq)[:3])),initial_position_prediction_error=float(np.linalg.norm(dx[:3]-(initial_j@dq)[:3])),estimate_update=float(np.linalg.norm(new_j-before_j)),goal=goal.tolist()))
            states.append(np.r_[d.qpos.copy(),d.qvel.copy()]);held=held+1 if np.linalg.norm(goal-after['position'])<.004 and np.linalg.norm(err(goal)[3:])<.012 else 0
            if held>=6 and phase not in ('calibrate','close','release','settle'):break
        phases.append(dict(phase=phase,steps=k+1,error=float(np.linalg.norm(goal-obs()['position'])),pads=len(pads),object=d.xpos[obj].tolist()))
        print(name,phase,phases[-1],flush=True)
        if phase=='lower':release_goal=goal.copy()
    oj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'object_joint');v=d.qvel[m.jnt_dofadr[oj]:m.jnt_dofadr[oj]+6]
    success=bool(grasp and lift and np.linalg.norm(d.xpos[obj,:2]-target)<.025 and abs(d.xpos[obj,2]-.215)<.008 and np.linalg.norm(v)<.03 and not pads)
    result=dict(robot=name,success=success,grasped=bool(grasp),lifted=bool(lift),final_object=d.xpos[obj].tolist(),target=target,phases=phases,trace=records,controller='Online analytical response estimator with0.8x model Jacobian prior at home; measured updates thereafter. Same algorithm for both arms. Not the learned policy benchmark.',scene='Native visual/kinematic models; demonstration-specific gravcomp position servos and simplified contact pads; physical free object, no welding or scripted object pose.',model_sha256=fingerprint(path))
    result.update(method=method,initialization='0.8x exact home Jacobian encoded into initial statistics; no further model Jacobian queries during execution.')
    if method=='neural':
        result['controller']='Frozen E99 retained seed0 neural response estimator, measured history updates, shared demo servo/task sequence; no weight updates. This is a separately configured demonstration, not E111-E113 evaluation.'
        result.update(checkpoint=checkpoint,checkpoint_sha256=fingerprint(checkpoint))
    index=len(list(ROOT.glob(f'{name}_attempt*.json')));(ROOT/f'{name}_attempt{index}.json').write_text(json.dumps(result));np.savez_compressed(ROOT/f'{name}_attempt{index}_states.npz',states=np.array(states),nq=m.nq,nv=m.nv)
    shutil.copy2(path,ROOT/f'{name}_attempt{index}.xml');print('RESULT',name,success,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('robot',choices=['piperx','openarm']);a=p.parse_args();run(a.robot)
