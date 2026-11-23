"""Physical grasp/lift/place fixture; scripted Cartesian rig, not an arm policy."""
import argparse
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from .mixed_sim import arm_xml


def add_gripper(parent, actuators, prefix='gripper'):
    """Attach a physical parallel jaw gripper along local -Z to an XML body."""
    ET.SubElement(parent,'geom',name=prefix+'_palm',type='box',size='.035 .055 .01',mass='.15',rgba='.25 .45 .75 1')
    for name,y,axis in [('left',.045,'0 -1 0'),('right',-.045,'0 1 0')]:
        finger=ET.SubElement(parent,'body',name=prefix+'_'+name,pos=f'0 {y} -.045')
        joint=prefix+'_'+name+'_slide'
        ET.SubElement(finger,'joint',name=joint,type='slide',axis=axis,range='0 .035',damping='2')
        ET.SubElement(finger,'geom',name=prefix+'_'+name+'_pad',type='box',size='.025 .008 .035',mass='.04',friction='1 .01 .001',condim='4',rgba='.25 .45 .75 1')
        ET.SubElement(actuators,'position',joint=joint,kp='200',kv='5',ctrlrange='0 .035',forcerange='-10 10')


def fixture_xml(object_xy=(0.,0.)):
    root=ET.Element('mujoco',model='ease_physical_pick_place_fixture')
    ET.SubElement(root,'compiler',angle='radian')
    ET.SubElement(root,'option',timestep='.002',integrator='implicitfast')
    world=ET.SubElement(root,'worldbody');act=ET.SubElement(root,'actuator')
    ET.SubElement(world,'light',pos='0 -.3 .8',dir='0 0 -1',diffuse='1 1 1',ambient='.4 .4 .4')
    ET.SubElement(world,'geom',name='table',type='box',pos='0 0 -.025',size='.4 .3 .025',friction='1 .01 .001',rgba='.65 .68 .7 1')
    carriage=ET.SubElement(world,'body',name='carriage',pos='0 0 .1',gravcomp='1')
    for axis,vector,limits in [('x','1 0 0','-.3 .3'),('y','0 1 0','-.3 .3'),('z','0 0 1','-.025 .3')]:
        ET.SubElement(carriage,'joint',name=axis,type='slide',axis=vector,range=limits,damping='5')
        ET.SubElement(act,'position',joint=axis,kp='2000',kv='80',ctrlrange=limits,forcerange='-500 500')
    add_gripper(carriage,act)
    obj=ET.SubElement(world,'body',name='object',pos=f'{object_xy[0]} {object_xy[1]} .0152')
    ET.SubElement(obj,'freejoint',name='object_free')
    ET.SubElement(obj,'geom',name='cube',type='box',size='.015 .015 .015',mass='.04',friction='1 .01 .001',condim='4',rgba='1 .45 .1 1')
    return ET.tostring(root,encoding='unicode')


def arm_fixture_xml(arm, object_xy=(.3,0.), target_xy=(.3,.12)):
    """Serial-arm scene asset; n arm actuators followed by two coupled jaw targets."""
    root=ET.fromstring(arm_xml(arm));world=root.find('worldbody')
    tip=root.find(f".//body[@name='link{arm.dof-1}']")
    mount=ET.SubElement(tip,'body',name='gripper_mount',pos=f'0 0 {arm.lengths[-1]}')
    add_gripper(mount,root.find('actuator'))
    ET.SubElement(mount,'site',name='grasp_center',pos='0 0 -.055',size='.005')
    ET.SubElement(world,'geom',name='table',type='box',pos='.35 0 .175',size='.2 .25 .025')
    obj=ET.SubElement(world,'body',name='object',pos=f'{object_xy[0]} {object_xy[1]} .2152')
    ET.SubElement(obj,'freejoint',name='object_free')
    ET.SubElement(obj,'geom',name='cube',type='box',size='.015 .015 .015',mass='.04',friction='1 .01 .001',condim='4')
    ET.SubElement(world,'site',name='place_target',pos=f'{target_xy[0]} {target_xy[1]} .201',type='cylinder',size='.025 .001',rgba='0 1 0 .4')
    return ET.tostring(root,encoding='unicode')


def run_case(object_xy=(0.,0.),target_xy=(.12,0.),render_directory=None):
    xml=fixture_xml(object_xy);model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    oid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'object')
    names=['cube','gripper_left_pad','gripper_right_pad','table','gripper_palm']
    ids={name:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name) for name in names}
    stages=[('approach',30,[*object_xy,-.015,0,0]),('close',40,[*object_xy,-.015,.035,.035]),
            ('lift',50,[*object_xy,.1,.035,.035]),('transport',60,[*target_xy,.1,.035,.035]),
            ('lower',50,[*target_xy,-.01,.035,.035]),('release',40,[*target_xy,-.01,0,0]),
            ('retreat',40,[*target_xy,.1,0,0]),('settle',60,[*target_xy,.1,0,0])]
    trace=[];bilateral=False;lifted=False;bad_contact=False
    renderer=None
    if render_directory is not None:
        from PIL import Image
        render_directory=Path(render_directory);render_directory.mkdir(parents=True,exist_ok=True)
        renderer=mujoco.Renderer(model,height=360,width=480)
        camera=mujoco.MjvCamera();camera.lookat[:]=[.06,0,.07];camera.distance=.55;camera.azimuth=135;camera.elevation=-25
    for stage,length,target in stages:
        start=data.ctrl.copy();target=np.array(target)
        for step in range(length):
            data.ctrl[:]=start+(target-start)*min(1,(step+1)/max(1,length//2))
            touched=set()
            for _ in range(20):
                mujoco.mj_step(model,data)
                for contact in data.contact[:data.ncon]:
                    pair={contact.geom1,contact.geom2}
                    for pad in ('gripper_left_pad','gripper_right_pad'):
                        if pair=={ids['cube'],ids[pad]}:touched.add(pad)
                    if ids['table'] in pair and pair.intersection({ids['gripper_left_pad'],ids['gripper_right_pad'],ids['gripper_palm']}):bad_contact=True
            mujoco.mj_forward(model,data)
            pos=data.xpos[oid].copy();bilateral |= len(touched)==2;lifted |= pos[2]>.075 and len(touched)==2
            trace.append(dict(stage=stage,object=pos.tolist(),bilateral=len(touched)==2,contacts=len(touched)))
        if renderer is not None:
            renderer.update_scene(data,camera=camera)
            Image.fromarray(renderer.render()).save(render_directory/f'{stage}.png')
    if renderer is not None:renderer.close()
    joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'object_free');addr=model.jnt_dofadr[joint]
    speed=float(np.linalg.norm(data.qvel[addr:addr+6]));pos=data.xpos[oid].copy()
    success=bool(bilateral and lifted and not bad_contact and np.linalg.norm(pos[:2]-target_xy)<.025 and abs(pos[2]-.015)<.005 and speed<.02 and trace[-1]['contacts']==0)
    return dict(success=success,lifted=bool(lifted),bilateral_contact=bool(bilateral),forbidden_contact=bool(bad_contact),
        final_object=pos.tolist(),final_speed=speed,steps=len(trace),trace=trace),xml


def run(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(Path(__file__),root/'fixture_source.py')
    protocol=dict(experiment='E36_fixture',scope='Scripted Cartesian carriage validates frictional grasp mechanics only. No arm policy, IK, object weld, attachment, or teleportation. Not a cross-embodiment result.',
        cases=[dict(object_xy=[0,0],target_xy=[.12,0]),dict(object_xy=[.01,.005],target_xy=[.1,.04]),dict(object_xy=[-.01,-.005],target_xy=[.1,-.04])])
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));results=[]
    for i,case in enumerate(protocol['cases']):
        result,xml=run_case(**case);results.append(result);(root/f'fixture_{i}.xml').write_text(xml)
        print('fixture',i,{k:v for k,v in result.items() if k!='trace'},flush=True)
    (root/'results.json').write_text(json.dumps(results,indent=2))
    (root/'status.json').write_text(json.dumps(dict(status='complete',passed=all(r['success'] for r in results))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(**vars(p.parse_args()))
