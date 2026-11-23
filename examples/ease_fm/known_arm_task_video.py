"""Replay recorded physical states and measured history overlays; no object animation."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,shutil
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
from .known_arm_task_demo import ROOT
from .benchmark import fingerprint

GREEN=(156,222,25);WHITE=(246,247,245);GRAY=(155,163,151)
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
SELECT={'piperx':(2,3),'openarm':(1,2)}
def txt(d,xy,s,size=24,color=WHITE,bold=False):d.text(xy,s,font=ImageFont.truetype(BOLD if bold else FONT,size),fill=color)
def center(d,y,s,size,color=WHITE):
    f=ImageFont.truetype(BOLD,size);d.text(((1280-d.textlength(s,font=f))/2,y),s,font=f,fill=color)

def verify():
    checks=[]
    for name,(old,new) in SELECT.items():
        a=json.loads((ROOT/f'{name}_attempt{old}.json').read_text());b=json.loads((ROOT/f'{name}_attempt{new}.json').read_text())
        assert a['trace']==b['trace'] and a['phases']==b['phases'] and a['success']==b['success']==True
        x=np.load(ROOT/f'{name}_attempt{old}_states.npz')['states'];y=np.load(ROOT/f'{name}_attempt{new}_states.npz')['states'];np.testing.assert_array_equal(x,y)
        assert fingerprint(b['checkpoint'])==b['checkpoint_sha256']
        checks.append(dict(robot=name,success=b['success'],full_trace_and_states_exact=True,transitions=len(b['trace']),peak_object_z=max(t['object'][2] for t in b['trace']),final_xy_error=float(np.linalg.norm(np.array(b['final_object'][:2])-b['target'])),mean_prediction_error_mm=float(np.mean([t['position_prediction_error'] for t in b['trace']])*1000),initial_estimate_counterfactual_error_mm=float(np.mean([t['initial_position_prediction_error'] for t in b['trace']])*1000),checkpoint=b['checkpoint'],checkpoint_sha256=b['checkpoint_sha256']))
    (ROOT/'demo_verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))

def render(name,preview=False,failure=False):
    number=0 if failure else SELECT[name][1];r=json.loads((ROOT/f'{name}_attempt{number}.json').read_text());assert r['success']!=failure
    p=ROOT/f'{name}_attempt{number}.xml';assert fingerprint(p)==r['model_sha256'];m=mujoco.MjModel.from_xml_path(str(p.resolve()));d=mujoco.MjData(m)
    states=np.load(ROOT/f'{name}_attempt{number}_states.npz')['states'];renderer=mujoco.Renderer(m,height=480,width=640)
    opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.sitegroup[:]=0;opt.sitegroup[0]=1
    camera=mujoco.MjvCamera();camera.lookat[:]=[.20,0,.34];camera.distance=1.0 if name=='piperx' else 1.10;camera.azimuth=135;camera.elevation=-25
    indices=[next(i for i,t in enumerate(r['trace']) if t['phase']==phase)+15 for phase in ('calibrate','close','transfer','release')] if preview else list(range(0,len(states),2))
    writer=None if preview else imageio.get_writer(ROOT/(f'{name}_attempt0_failure.mp4' if failure else f'{name}_physical.mp4'),fps=12.5,codec='libx264',quality=8,macro_block_size=16)
    for count,i in enumerate(indices):
        d.qpos[:]=states[i,:m.nq];d.qvel[:]=states[i,m.nq:];mujoco.mj_forward(m,d)
        site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'task_tcp');obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'cube')
        np.testing.assert_allclose(d.site_xpos[site],r['trace'][i]['tool'],atol=1e-12);np.testing.assert_allclose(d.xpos[obj],r['trace'][i]['object'],atol=1e-12)
        renderer.update_scene(d,camera=camera,scene_option=opt);im=Image.fromarray(renderer.render())
        if preview:im.save(ROOT/f'{name}_{r["trace"][i]["phase"]}_preview.png')
        else:
            writer.append_data(np.asarray(im))
            if count%50==0:print(name,'rendered',count,'/',len(indices),flush=True)
    if writer:writer.close()
    renderer.close()

def compose():
    clips={};traces={};sources={}
    for name,(_,number) in SELECT.items():
        p=ROOT/f'{name}_physical.mp4';reader=imageio.get_reader(p);clips[name]=[Image.fromarray(f) for f in reader];reader.close();traces[name]=json.loads((ROOT/f'{name}_attempt{number}.json').read_text())['trace'];sources[name]=dict(video_sha256=fingerprint(p),trace_sha256=fingerprint(ROOT/f'{name}_attempt{number}.json'),state_sha256=fingerprint(ROOT/f'{name}_attempt{number}_states.npz'))
    writer=imageio.get_writer(ROOT/'ease_pick_place_demo.mp4',fps=25,codec='libx264',quality=8,macro_block_size=16)
    for f in range(30*25):
        t=f/25;sim=max(0,t-2);im=Image.new('RGB',(1280,720),'black');draw=ImageDraw.Draw(im)
        if t<21 or t>=27:
            for k,name in enumerate(('piperx','openarm')):
                idx=min(len(clips[name])-1,int(sim*12.5));im.paste(clips[name][idx],(k*640,154));row=traces[name][min(len(traces[name])-1,int(sim*25))]
                txt(draw,(48+k*640,114),'PiperX  /  6 joints' if k==0 else 'OpenArm  /  7 joints',22,GREEN,True)
                txt(draw,(48+k*640,605),row['phase'].upper(),20,GREEN,True)
                txt(draw,(48+k*640,640),f"History: {row['history']} measured transitions",18)
                end=min(len(traces[name]),int(sim*25)+1);window=traces[name][max(0,end-12):end]
                residual=np.mean([v['position_prediction_error'] for v in window])*1000
                txt(draw,(48+k*640,669),f'Motion prediction residual: {residual:.2f} mm',16,GRAY)
            txt(draw,(48,24),'EASE',26,WHITE,True);txt(draw,(933,30),'PHYSICS SIMULATION / 1x',17,GREEN)
            if t<2:center(draw,65,'One controller. Two robot arms.',37)
            elif t<4:center(draw,65,'Measure motion. Update the response.',34)
            elif t<10:center(draw,65,'Adapt from history. Grasp and lift.',36)
            elif t<16:center(draw,65,'Carry the object. Keep updating.',36)
            elif t<19.1:center(draw,65,'Place. Release. Return.',37)
            elif t<21:center(draw,65,'Task complete.',37)
            else:
                draw.rectangle((0,0,1280,154),fill='black');center(draw,38,'Different kinematics. Shared control.',42,GREEN);center(draw,103,'Learning how each body moves.',25)
        else:
            txt(draw,(48,24),'EASE',26,WHITE,True);center(draw,112,'Progress, measured.',56)
            for k,(label,score,den) in enumerate([('TRAINING BODIES','22 → 31','36 episodes'),('UNSEEN BODIES','28 → 31','48 episodes'),('6-JOINT RELEASE','14 → 16','24 episodes')]):
                x=52+k*416;txt(draw,(x,280),label,20,GRAY,True);txt(draw,(x,345),score,58,GREEN,True);txt(draw,(x,435),' / '+den,23)
            center(draw,541,'More completed tasks. Prior successes retained.',28)
            center(draw,630,'Physics-assisted procedural benchmarks · E111–E113',20,GRAY)
        txt(draw,(48,701),'Demo: frozen neural estimator + online history · configured servos/contact pads',12,GRAY)
        draw.rectangle((0,717,int(1280*t/30),720),fill=GREEN)
        writer.append_data(np.asarray(im))
        if f in (75,175,275,425,600,725):im.save(ROOT/f'film_preview_{f}.png')
    writer.close();reader=imageio.get_reader(ROOT/'ease_pick_place_demo.mp4');assert reader.count_frames()==750;reader.close()
    manifest=dict(seconds=30,frames=750,fps=25,sha256=fingerprint(ROOT/'ease_pick_place_demo.mp4'),sources=sources,motion='Complete physical trajectories at1x simulation playback, native render sampled every2 control steps; final states held. Object positions replayed from executed free-body physics. Network weights fixed; response estimate changes from measured history. Residual is12-step rolling XYZ motion-prediction error.',scope='Two development demos, same frozen E99 checkpoint. Demonstration-specific position servos, gravity compensation, simplified pads and home-geometry prior. Not hardware or an independent transfer evaluation. Prior failed attempts retained. Metrics card refers separately to E111-E113 procedural experiments.')
    (ROOT/'video_manifest.json').write_text(json.dumps(manifest,indent=2))
    (ROOT/'index.html').write_text('<html><body style="background:black;color:white;font:18px sans-serif;padding:28px"><h1>EASE / Pick, adapt, place</h1><video controls width="100%" style="max-width:1280px" src="ease_pick_place_demo.mp4"></video><p>PiperX + OpenArm · Actual simulated pick-and-place · Measured online history · 30 seconds</p><a href="video_manifest.json" style="color:#9cde19">Production and verification notes</a></body></html>')
    for name in ('known_arm_task_demo.py','known_arm_task_video.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/name)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['verify','preview','render','compose','failure']);p.add_argument('--robot',default='piperx');a=p.parse_args()
    if a.stage=='verify':verify()
    elif a.stage=='compose':compose()
    else:render(a.robot,a.stage=='preview',a.stage=='failure')
