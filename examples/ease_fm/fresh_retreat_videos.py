import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
import shutil
from pathlib import Path
import numpy as np
import torch
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from .fresh_retreat import ROOT,retained,evaluate as fresh_evaluate
MOTION=('retreat',)
def evaluate(case,t,seed,mode,parents,students,record_step=None,affected_phase=None):
    return fresh_evaluate(case,t,seed,mode,parents,students,record_step)
from .mixed_sim import AXES
from .benchmark import fingerprint


def run():
    torch.set_num_threads(4);out=ROOT/'videos';out.mkdir(exist_ok=True)
    tasks=json.loads((ROOT/'cases.json').read_text());AXES.update(json.loads((ROOT/'axis_catalog.json').read_text()));parents,hashes=retained.previous.models();students,hh=retained.load_students();hashes.update(hh)
    rows=json.loads((ROOT/'records.json').read_text());selection=[(0,0,None,'baseline'),(0,0,'retreat','budget'),(0,0,'retreat','gentle')]
    lookup=lambda c,s,p,m:next(r for r in rows if r['case']==c and r['seed']==s and r['affected_phase']==p and r['mode']==m)
    for r in rows:
        if r['mode']!='gentle':continue
        if any(r['success']!=lookup(r['case'],r['seed'],p,m)['success'] for p,m in ((None,'baseline'),('retreat','budget'))):
            for p,m in ((None,'baseline'),('retreat','budget'),('retreat','gentle')):
                key=(r['case'],r['seed'],p,m)
                if key not in selection:selection.append(key)
    for outcome in (True,False):
        if any(r['success']==outcome for r in rows) and not any(lookup(*key)['success']==outcome for key in selection):
            r=next(r for r in sorted(rows,key=lambda r:(r['case'],r['seed'],r['affected_phase'] or '',r['mode'])) if r['success']==outcome);selection.append((r['case'],r['seed'],r['affected_phase'],r['mode']))
    manifest=dict(selection='First accepted case seed0 all modes; every gain or loss versus either control with all modes; append missing outcome category. Outcome-selected illustrations.',playback='Full executed task except32 calibration steps. Every fourth40ms step at6.25fps plus final frame. Frozen neural weights; privileged route planner and geometry filter.',videos=[])
    for case,seed,phase,mode in selection:
        ref=ROOT/'trials'/f'{case}_{seed}_{phase}_{mode}.json';old=json.loads(ref.read_text());label='success' if old['success'] else 'failure';file=f'case{case}_seed{seed}_{phase}_{mode}_{label}.mp4';renderer=None;last=None;frames=0
        camera=mujoco.MjvCamera();camera.lookat[:]=[.25,0,.30];camera.distance=2.6;camera.azimuth=130;camera.elevation=-25
        writer=imageio.get_writer(out/file,fps=6.25,codec='libx264',quality=8,macro_block_size=16)
        def capture(env,force=False):
            nonlocal renderer,last,frames
            last=env
            if not force and (env.steps-32)%4:return
            if renderer is None:renderer=mujoco.Renderer(env.model,height=480,width=640)
            renderer.update_scene(env.data,camera=camera);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,58),fill='black');draw.text((8,4),f'E105 fresh-body case {case} | seed {seed} | {phase}/{mode} | {label.upper()}',fill='white');draw.text((8,21),'EXECUTED | privileged geometry | fixed neural weights',fill='white');draw.text((8,38),f'{env.phase} | {(env.steps-32)*.04:.2f}s | forbidden collision: {env.bad}',fill='white');writer.append_data(np.asarray(im));frames+=1
            if frames==40:im.save(out/f'{Path(file).stem}_preview.png')
        try:
            result=evaluate(case,tasks[case],seed,mode,parents,students,capture,affected_phase=phase)
            if (last.steps-32)%4:capture(last,True)
        finally:
            writer.close()
            if renderer is not None:renderer.close()
        assert json.loads(json.dumps(result))==old
        reader=imageio.get_reader(out/file);assert reader.count_frames()==frames;reader.close()
        manifest['videos'].append(dict(file=file,case=case,seed=seed,affected_phase=phase,mode=mode,success=old['success'],frames=frames,exact_replay=True,sha256=fingerprint(out/file),reference_sha256=fingerprint(ref)));(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(file,'verified',flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h
    html='<html><body style="background:#111;color:#eee;font:16px sans-serif;margin:30px"><h1>Prospective body and placement replication</h1><p>'+manifest['selection']+'</p><p>'+manifest['playback']+'</p>'
    for v in manifest['videos']:html+=f'<h2>{v["file"]}</h2><video controls width="640" src="{v["file"]}"></video>'
    (out/'index.html').write_text(html+'</body></html>');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)


if __name__=='__main__':run()


