"""50 fps physical replay and quiet, dissolving editorial treatment."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json, shutil
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw
from .known_arm_task_video import txt,center,GREEN,GRAY
from .benchmark import fingerprint
from .task_mesh_clearance import audit
SOURCE=Path('outputs/ease_known_arm_task_demo_v2')
ROOT=Path('outputs/ease_known_arm_task_demo_v3');ROOT.mkdir(exist_ok=True)

def render(name):
    p=ROOT/f'{name}_attempt1.xml'; r=json.loads((ROOT/f'{name}_attempt1.json').read_text())
    assert r['success'] and fingerprint(p)==r['model_sha256']
    data=np.load(ROOT/f'{name}_attempt1_states.npz'); states=data['render_states']
    np.testing.assert_array_equal(states[1::2],data['states'])
    old=json.loads((ROOT/f'{name}_attempt0.json').read_text());assert old['trace']==r['trace']
    dense=ROOT/f'{name}_render_states.npz';np.savez_compressed(dense,states=states)
    check=audit(p,dense);assert check['minimum_table_clearance']['clearance']>0 and check['minimum_floor_clearance']>0
    check.update(success=True,repeat_trace_exact=True,frames=len(states),final_xy_error=float(np.linalg.norm(np.array(r['final_object'][:2])-r['target'])))
    (ROOT/f'{name}_clearance.json').write_text(json.dumps(check,indent=2))
    m=mujoco.MjModel.from_xml_path(str(p.resolve()));d=mujoco.MjData(m); renderer=mujoco.Renderer(m,height=480,width=640)
    opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.sitegroup[:]=0
    camera=mujoco.MjvCamera();camera.lookat[:]=[.20,0,.34];camera.distance=1.0 if name=='piperx' else 1.10;camera.azimuth=135;camera.elevation=-25
    writer=imageio.get_writer(ROOT/f'{name}_physical.mp4',fps=50,codec='libx264',quality=8,macro_block_size=16)
    for i,state in enumerate(states):
        d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_forward(m,d)
        renderer.update_scene(d,camera=camera,scene_option=opt);frame=renderer.render();writer.append_data(frame)
        if i in (205,360,610):Image.fromarray(frame).save(ROOT/f'{name}_check_{i}.png')
    writer.close();renderer.close();print(name,check,flush=True)

def smooth(x):
    x=np.clip(x,0,1);return float(x*x*(3-2*x))


def aperture(im,u):
    if u>=1:return im
    width=int(640*smooth(u));out=Image.new('RGB',im.size,'black')
    out.paste(im.crop((640-width,0,640+width,720)),(640-width,0))
    d=ImageDraw.Draw(out)
    for x in (640-width,640+width):
        d.line((x,0,x,720),fill=GREEN,width=2)
        d.ellipse((x-5,355,x+5,365),fill=GREEN)
    return out

def scan_panels(old,new,u):
    # Three staggered shutters echo the three benchmark columns.
    if u>=1:return new.copy()
    out=old.copy();d=ImageDraw.Draw(out)
    for k,(left,right) in enumerate(((0,432),(432,848),(848,1280))):
        v=smooth((u-k*.11)/.78);y=int(720*(1-v))
        if y<720:out.paste(new.crop((left,y,right,720)),(left,y))
        if 0<v<1:
            d.line((left,y,right,y),fill=GREEN,width=3)
            x=(left+right)//2
            d.ellipse((x-6,y-6,x+6,y+6),fill=GREEN)
            # Quiet trailing ticks give the sweep a measurement-instrument feel.
            for tick in range(left+24,right,40):d.line((tick,y,tick,y+8),fill=GREEN,width=1)
    return out

def joint_iris(old,new,u):
    # A two-link graphic unfolds around the centre, opening a circular aperture.
    # This is editorial geometry, never overlaid during robot execution.
    if u>=1:return new.copy()
    import math
    v=smooth(u);r=int(760*v)
    mask=Image.new('L',old.size,0);md=ImageDraw.Draw(mask)
    md.ellipse((640-r,360-r,640+r,360+r),fill=255)
    out=Image.composite(new,old,mask);d=ImageDraw.Draw(out)
    if r>0:d.ellipse((640-r,360-r,640+r,360+r),outline=GREEN,width=3)
    # Linkage stays outside the reveal, then exits with its expanding boundary.
    a=-math.pi/2+v*math.pi*1.3
    pts=[]
    for offset,scale in ((0,1),(.48,1.10),(.94,1.02)):
        pts.append((int(640+(r+22)*scale*math.cos(a+offset)),int(360+(r+22)*scale*math.sin(a+offset))))
    d.line(pts,fill=GREEN,width=3)
    for x,y in pts:d.ellipse((x-6,y-6,x+6,y+6),fill=(0,0,0),outline=GREEN,width=3)
    return out

def compose():
    names=('piperx','openarm'); clips={};traces={}
    for name in names:
        reader=imageio.get_reader(SOURCE/f'{name}_physical.mp4');clips[name]=[Image.fromarray(f) for f in reader];reader.close()
        traces[name]=json.loads((SOURCE/f'{name}_attempt1.json').read_text())['trace']
    card=Image.new('RGB',(1280,720),'black');d=ImageDraw.Draw(card)
    txt(d,(48,30),'EASE / PROGRESS',22,GREEN,True);center(d,130,'More tasks completed.',48)
    for k,(label,score,den) in enumerate([('TRAINING BODIES','22 → 31','of 36 episodes'),('UNSEEN BODIES','28 → 31','of 48 episodes'),('6-JOINT RELEASE','14 → 16','of 24 episodes')]):
        x=56+k*416;txt(d,(x,300),label,19,GRAY,True);txt(d,(x,355),score,54,GREEN,True);txt(d,(x,435),den,22)
    center(d,580,'Prior successes retained.',27);center(d,658,'Separate physics-assisted benchmarks · E111–E113',18,GRAY)
    end=Image.new('RGB',(1280,720),'black');d=ImageDraw.Draw(end)
    center(d,240,'Different kinematics.',50);center(d,310,'Shared control.',50,GREEN);center(d,418,'Learning how each body moves.',25)
    writer=imageio.get_writer(ROOT/'ease_pick_place_demo.mp4',fps=50,codec='libx264',quality=8,macro_block_size=16)
    for f in range(1500):
        t=f/50;sim=max(0,t-1.5);im=Image.new('RGB',(1280,720),'black');d=ImageDraw.Draw(im)
        txt(d,(48,28),'EASE',24,GREEN,True);txt(d,(1000,32),'SIMULATION / 1×',17,GRAY)
        center(d,78,'One controller. Two robot arms.',36)
        for k,name in enumerate(names):
            idx=min(len(clips[name])-1,int(sim*50));im.paste(clips[name][idx],(640*k,160))
            txt(d,(48+640*k,140),'PiperX / 6 joints' if k==0 else 'OpenArm / 7 joints',21,GREEN,True)
        # Short stable phrases; each dissolves while the physical footage continues.
        captions=[(0,'Observe motion. Build history.'),(4,'Grasp and lift.'),(9,'Carry and place.'),(14,'Release. Task complete.')]
        at=max(i for i,(start,_) in enumerate(captions) if t>=start)
        layer=Image.new('RGB',(1280,720),'black');ld=ImageDraw.Draw(layer)
        center(ld,630,captions[at][1],25)
        if at and t<captions[at][0]+.6:
            u=smooth((t-captions[at][0])/.6);layer=Image.new('RGB',(1280,720),'black')
            ld=ImageDraw.Draw(layer)
            center(ld,630-int(48*u),captions[at-1][1],25)
            center(ld,630+int(48*(1-u)),captions[at][1],25)
        im.paste(layer.crop((0,625,1280,670)),(0,625));d=ImageDraw.Draw(im)
        count=min(len(traces['piperx']),int(sim*25));txt(d,(48,685),'FROZEN NETWORK · ONLINE HISTORY',14,GRAY)
        txt(d,(928,685),f'{count:03d} measured transitions',15,GREEN)
        if t>=20:im=scan_panels(im,card,(t-20)/1.15)
        if t>=26:im=joint_iris(im,end,(t-26)/1.1)
        if t<.8:im=aperture(im,t/.8)
        writer.append_data(np.asarray(im))
        if f in (15,175,215,350,600,800,1015,1025,1040,1150,1310,1325,1340,1400):im.save(ROOT/f'film_preview_{f}.png')
    writer.close()
    reader=imageio.get_reader(ROOT/'ease_pick_place_demo.mp4');assert reader.count_frames()==1500;reader.close()
    manifest=dict(seconds=30,fps=50,frames=1500,sha256=fingerprint(ROOT/'ease_pick_place_demo.mp4'),motion='Native physics snapshots every 20 ms, 1x playback; no interpolated or animated object motion. Staggered scanning panels, a two-link iris, split-aperture opening, and masked rolling captions. No fades; complete task footage unchanged. Both tasks successful and deterministic repeat verified.',scope='Development demos, same frozen E99 estimator; configured servos/pads and home geometry prior. Online measured history, no network weight updates. Metrics card is separate E111–E113 research.',clearance={n:json.loads((SOURCE/f'{n}_clearance.json').read_text()) for n in names})
    (ROOT/'video_manifest.json').write_text(json.dumps(manifest,indent=2))
    (ROOT/'index.html').write_text('<html><body style="background:#000;color:#eee;font:18px sans-serif;padding:28px"><h1>EASE / Pick, adapt, place</h1><video controls autoplay muted width="100%" style="max-width:1280px" src="ease_pick_place_demo.mp4"></video><p>PiperX + OpenArm · 50 fps · Physics simulation</p><a style="color:#9cde19" href="video_manifest.json">Verification notes</a></body></html>')
    for file in ('build_motion_edit.py','motion_task_video.py'):shutil.copy2(Path(__file__).with_name(file),ROOT/file)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='compose':compose()
    else:render(sys.argv[1])
