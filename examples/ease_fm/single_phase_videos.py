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
from .single_phase import ROOT,MOTION,evaluate,prior
from .mixed_sim import AXES
from .benchmark import fingerprint


def run():
    torch.set_num_threads(4);out=ROOT/'videos';out.mkdir(exist_ok=True)
    tasks=json.loads((prior.TASKS/'cases.json').read_text());AXES.update(json.loads((prior.TASKS/'axis_catalog.json').read_text()));parents,hashes=prior.previous.models();students,hh=prior.e99.load_students();hashes.update(hh)
    rows=json.loads((ROOT/'records.json').read_text());selection=[(2,0,None,'baseline')]+[(2,0,p,m) for p in MOTION for m in ('budget','gentle')]
    lookup=lambda c,s,p,m:next(r for r in rows if r['case']==c and r['seed']==s and r['affected_phase']==p and r['mode']==m)
    for phase in MOTION:
        for control in ('baseline','budget'):
            refphase=None if control=='baseline' else phase
            losses=[r for r in rows if r['mode']=='gentle' and r['affected_phase']==phase and not r['success'] and lookup(r['case'],r['seed'],refphase,control)['success']]
            if losses:
                r=min(losses,key=lambda r:(r['case'],r['seed']))
                for p,m in ((refphase,control),(phase,'gentle')):
                    key=(r['case'],r['seed'],p,m)
                    if key not in selection:selection.append(key)
    for outcome in (True,False):
        if not any(lookup(*key)['success']==outcome for key in selection):
            r=next(r for r in sorted(rows,key=lambda r:(r['case'],r['seed'],r['affected_phase'] or '',r['mode'])) if r['success']==outcome);selection.append((r['case'],r['seed'],r['affected_phase'],r['mode']))
    manifest=dict(selection='Prespecified training case2 seed0 baseline and all eight phase controls. Append first gentle loss versus each reference in each phase, plus missing outcome category. Outcome-selected illustrations.',playback='Complete executed task,32 calibration steps omitted. Every fourth40ms step at6.25fps plus final frame. Fixed E99 weights; privileged geometry filter, no route planner.',videos=[])
    for case,seed,phase,mode in selection:
        ref=ROOT/'trials'/f'{case}_{seed}_{phase}_{mode}.json';old=json.loads(ref.read_text());label='success' if old['success'] else 'failure';file=f'case{case}_seed{seed}_{phase}_{mode}_{label}.mp4';renderer=None;last=None;frames=0
        camera=mujoco.MjvCamera();camera.lookat[:]=[.25,0,.30];camera.distance=2.6;camera.azimuth=130;camera.elevation=-25
        writer=imageio.get_writer(out/file,fps=6.25,codec='libx264',quality=8,macro_block_size=16)
        def capture(env,force=False):
            nonlocal renderer,last,frames
            last=env
            if not force and (env.steps-32)%4:return
            if renderer is None:renderer=mujoco.Renderer(env.model,height=480,width=640)
            renderer.update_scene(env.data,camera=camera);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,58),fill='black');draw.text((8,4),f'E104 training case {case} | seed {seed} | {phase}/{mode} | {label.upper()}',fill='white');draw.text((8,21),'EXECUTED | privileged geometry | fixed neural weights',fill='white');draw.text((8,38),f'{env.phase} | {(env.steps-32)*.04:.2f}s | forbidden collision: {env.bad}',fill='white');writer.append_data(np.asarray(im));frames+=1
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
    html='<html><body style="background:#111;color:#eee;font:16px sans-serif;margin:30px"><h1>Isolated phase-transition training screen</h1><p>'+manifest['selection']+'</p><p>'+manifest['playback']+'</p>'
    for v in manifest['videos']:html+=f'<h2>{v["file"]}</h2><video controls width="640" src="{v["file"]}"></video>'
    (out/'index.html').write_text(html+'</body></html>');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)


if __name__=='__main__':run()
