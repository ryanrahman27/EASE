"""Short concept/progress film with actual PiperX and OpenArm visual models."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse
import json
import math
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from .benchmark import fingerprint

ROOT=Path('outputs/ease_known_arm_demo_v2');W,H,FPS=1280,720,24
GREEN=(156,222,25);WHITE=(246,247,245);GRAY=(158,165,156)
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
def txt(d,xy,s,size=24,color=WHITE,bold=False):d.text(xy,s,font=ImageFont.truetype(BOLD if bold else FONT,size),fill=color)
def center(d,y,s,size,color=WHITE):
    f=ImageFont.truetype(BOLD,size);length=d.textlength(s,font=f);d.text(((W-length)/2,y),s,font=f,fill=color)

def prepare():
    ROOT.mkdir(exist_ok=True);(ROOT/'models').mkdir(exist_ok=True)
    source=Path('.tmp-piperx-latency/assets/piper_x_description').resolve();p=source/'urdf/piper_x_description.urdf';root=ET.parse(p).getroot();assets=[]
    for link in root.findall('link'):
        for c in link.findall('collision'):link.remove(c)
    for mesh in root.findall('.//mesh'):
        name=Path(mesh.get('filename')).name
        actual=source/'meshes/dae'/name
        if not actual.exists():actual=source/'meshes/dae'/name.lower()
        dest=(ROOT/'models'/f'{actual.stem}.obj').resolve()
        if not dest.exists():
            scene=trimesh.load(actual,force='scene');geo=scene.to_geometry();geo.export(dest)
        mesh.set('filename',str(dest));assets.append(dict(path=str(actual),sha256=fingerprint(actual)))
    mj=ET.SubElement(root,'mujoco');ET.SubElement(mj,'compiler',discardvisual='false',fusestatic='false',balanceinertia='true',boundmass='.001',boundinertia='.000001')
    urdf=ROOT/'models/piperx.urdf';ET.ElementTree(root).write(urdf)
    model=mujoco.MjModel.from_xml_path(str(urdf.resolve()));mujoco.mj_saveLastXML(str(ROOT/'models/piperx.xml'),model)
    op=Path('.tmp-openarm-mujoco/v1/openarm.xml').resolve();tree=ET.parse(op);tree.getroot().find('compiler').set('meshdir',str(op.parent/'meshes'));tree.write(ROOT/'models/openarm.xml')
    for name in ('piperx','openarm'):
        tree=ET.parse(ROOT/'models'/f'{name}.xml');r=tree.getroot();visual=r.find('visual')
        if visual is None:visual=ET.SubElement(r,'visual')
        ET.SubElement(visual,'global',offwidth='640',offheight='540')
        ET.SubElement(visual,'headlight',ambient='.35 .35 .35',diffuse='.45 .45 .45',specular='.3 .3 .3')
        world=r.find('worldbody');ET.SubElement(world,'light',pos='1 -2 3',dir='-1 2 -3',diffuse='.6 .65 .7',castshadow='true')
        ET.SubElement(world,'light',pos='-1 1 2',dir='1 -1 -2',diffuse='.45 .6 .3')
        ET.SubElement(world,'geom',name='demo_floor',type='plane',size='3 3 .1',pos='0 0 -.006',rgba='.045 .05 .04 1',contype='0',conaffinity='0')
        tree.write(ROOT/'models'/f'{name}_scene.xml')
    manifest=dict(style_reference='https://research.nvidia.com/labs/gear/robottt/',visual_direction='Black, white, acid green; large typography, sparse captions, prominent robot footage. Original graphics; no reference footage/logos reused.',
      models=[dict(name='PiperX',arm_dof=6,source=str(p),sha256=fingerprint(p),repository='https://github.com/axiboai/piperx_lerobot_setup',meshes=assets),dict(name='OpenArm v1',arm_dof=7,source=str(op),sha256=fingerprint(op),repository='https://github.com/enactic/openarm_mujoco',license='Apache-2.0')],
      motion='Articulated concept visualization using original joint hierarchy/limits and visual meshes. Visual-only finger inertias bounded for compilation; smooth scripted joint motion with mj_forward; not a trained-policy rollout or task-success demonstration.',
      metrics='E111 original training-body suite22/36->31/36; E112 prospective body parameter suite28/48->31/48; E113 reused six-joint cohort14/24->16/24. These are physics-assisted procedural simulation results, not evaluations of the pictured PiperX/OpenArm models.',duration_seconds=32,fps=FPS,resolution=[W,H],audio='Silent/captioned')
    (ROOT/'provenance.json').write_text(json.dumps(manifest,indent=2))
    license=Path('.tmp-openarm-mujoco/LICENSE')
    if license.exists():shutil.copy2(license,ROOT/'OpenArm_LICENSE.txt')

def rigs():
    result=[]
    for name,n in [('piperx',6),('openarm',7)]:
        m=mujoco.MjModel.from_xml_path(str((ROOT/'models'/f'{name}_scene.xml').resolve()));d=mujoco.MjData(m)
        if name=='openarm':mujoco.mj_resetDataKeyframe(m,d,0)
        else:d.qpos[:8]=[0,1.0,-1.15,0,.3,0,.025,-.025]
        q=d.qpos.copy();mujoco.mj_forward(m,d)
        camera=mujoco.MjvCamera();camera.lookat[:]=[.1,0,.28];camera.distance=1.22 if name=='piperx' else 1.18;camera.azimuth=135;camera.elevation=-17
        opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.sitegroup[:]=0
        renderer=mujoco.Renderer(m,height=540,width=640)
        print(name,'joints',m.njnt,'qpos',q.tolist(),flush=True)
        result.append((name,n,m,d,q,camera,opt,renderer))
    return result

def frame(t,rs):
    im=Image.new('RGB',(W,H),'black');d=ImageDraw.Draw(im)
    for i,(name,n,m,data,q,camera,opt,renderer) in enumerate(rs):
        data.qpos[:]=q
        for j in range(n):
            offset=.15*math.sin(t*.6+j*.9)*math.sin(math.pi*min(t/2,1)/2)
            data.qpos[j]=np.clip(q[j]+offset,m.jnt_range[j,0]+.01,m.jnt_range[j,1]-.01)
        mujoco.mj_forward(m,data);camera.azimuth=135+7*math.sin(t*.14+i*.4)
        renderer.update_scene(data,camera=camera,scene_option=opt);a=Image.fromarray(renderer.render());im.paste(a,(640*i,140))
    d=ImageDraw.Draw(im)
    # Consistent captions identify the known-arm footage as concept visualization.
    txt(d,(48,26),'EASE',26,WHITE,True);txt(d,(1080,31),'RESEARCH',16,GREEN,True)
    if t<6:
        center(d,81,'Different bodies. One goal.',48)
        txt(d,(78,596),'PiperX',34,WHITE,True);txt(d,(80,643),'6 arm joints',21,GREEN)
        txt(d,(720,596),'OpenArm',34,WHITE,True);txt(d,(722,643),'7 arm joints',21,GREEN)
    elif t<13:
        center(d,81,'Let experience become context.',43)
        # Animated context strip; schematic only, not internal model telemetry.
        for i in range(18):
            x=220+i*47;alpha=int(45+160*(.5+.5*math.sin(t*2-i*.6)))
            d.rounded_rectangle((x,582,x+31,623),radius=5,fill=(int(alpha*.65),alpha,12))
        center(d,641,'ACT  /  OBSERVE  /  ADAPT',23,GREEN)
    elif t<20:
        center(d,81,'From motion history to control.',44)
        d.rounded_rectangle((345,580,935,640),radius=30,fill=(18,26,7),outline=GREEN,width=2)
        center(d,594,'History-conditioned adaptation',26,GREEN)
    elif t<27:
        # Separate results from concept models so measurements retain their scope.
        im=Image.new('RGB',(W,H),'black');d=ImageDraw.Draw(im);txt(d,(48,26),'EASE',26,WHITE,True)
        center(d,112,'Progress, measured.',57)
        labels=['TRAINING BODIES','UNSEEN BODIES','6-JOINT RELEASE']
        numbers=['22 → 31','28 → 31','14 → 16'];denoms=['/ 36 episodes','/ 48 episodes','/ 24 episodes']
        for i in range(3):
            x=52+i*416;txt(d,(x,288),labels[i],20,GRAY,True);txt(d,(x,348),numbers[i],58,GREEN,True);txt(d,(x,433),denoms[i],23,WHITE)
        center(d,539,'More completed tasks. Every prior success retained.',27)
        center(d,633,'Physics-assisted simulation benchmarks  ·  E111 / E112 / E113',19,GRAY)
    else:
        center(d,80,'Different kinematics.',46)
        center(d,143,'Shared intelligence.',46,GREEN)
        center(d,609,'Adapting control through experience.',29)
    if not 20<=t<27:txt(d,(48,693),'CONCEPT VISUALIZATION  ·  PiperX + OpenArm models',13,GRAY)
    d.rectangle((0,716,int(W*t/32),720),fill=GREEN)
    # Short fade between chapters; keep graphics and native meshes crisp.
    edges=[0,6,13,20,27,32];dist=min(abs(t-e) for e in edges)
    if dist<.2:im=Image.blend(Image.new('RGB',im.size,'black'),im,max(.15,dist/.2))
    return im

def render(preview=False):
    rs=rigs()
    if preview:
        for t in (3,9,16,23,29):frame(t,rs).save(ROOT/f'preview_{t}.png')
    else:
        writer=imageio.get_writer(ROOT/'ease_short_demo.mp4',fps=FPS,codec='libx264',quality=9,macro_block_size=16)
        for f in range(32*FPS):
            t=f/FPS;im=frame(t,rs);writer.append_data(np.asarray(im))
            if f in (72,216,384,552,696):im.save(ROOT/f'preview_{int(t)}.png')
            if f%192==0:print('rendered seconds',int(t),flush=True)
        writer.close();reader=imageio.get_reader(ROOT/'ease_short_demo.mp4');assert reader.count_frames()==768;reader.close()
        shutil.copy2(__file__,ROOT/'known_arm_demo.py')
        (ROOT/'verification.json').write_text(json.dumps(dict(frames=768,seconds=32,sha256=fingerprint(ROOT/'ease_short_demo.mp4')),indent=2))
        (ROOT/'index.html').write_text('<html><body style="background:black;color:white;font:18px sans-serif;padding:28px"><h1>EASE <span style="color:#9cde19">/ Concept + progress</span></h1><video controls width="100%" style="max-width:1280px" src="ease_short_demo.mp4"></video><p>32 seconds · PiperX + OpenArm · Concept model visualization and measured simulation progress</p><a style="color:#9cde19" href="provenance.json">Sources and production notes</a></body></html>')
    for *_,r in rs:r.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','preview','render']);a=p.parse_args()
    if a.stage=='prepare':prepare()
    else:render(a.stage=='preview')
