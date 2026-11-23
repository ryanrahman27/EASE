"""E102 parent versus distilled neural post-probe continuation videos."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
from pathlib import Path
import shutil
import numpy as np
import torch
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from .onpolicy_history import evaluate,previous,MotionTransformer,load_students,e99
from .grasp_contact_audit import ROOT
PRIOR=previous.PRIOR
models=previous.models
from .mixed_sim import AXES
from .benchmark import fingerprint


def run():
    torch.set_num_threads(4);out=ROOT/'videos';out.mkdir(exist_ok=True)
    tasks=json.loads((PRIOR/'cases.json').read_text());AXES.update(json.loads((PRIOR/'axis_catalog.json').read_text()));loaded,hashes=models()
    students,new_hashes=load_students();hashes.update(new_hashes)
    starts,start_hashes=e99.load_students();hashes.update(start_hashes)
    students.update({('start',seed):starts[('retained',seed)] for seed in range(3)})
    selection=json.loads((ROOT/'protocol.json').read_text())['selections']
    manifest=dict(selection='All five E101 non-collision regressions, each paired with a successful same-case same-seed reference. Outcome-selected diagnostic examples.',playback='Complete executed task; calibration omitted. Every fourth40ms step at6.25fps plus final frame. Original trajectories reproduced exactly. Privileged planning.',videos=[])
    for case,method,mode in selection:
        ref=ROOT/'trials'/f'{case}_{method}_{mode}.json';old=json.loads(ref.read_text());label='SUCCESS' if old['success'] else 'FAILURE'
        filename=f'case{case}_{method}_{mode}_{label.lower()}.mp4';renderer=None;frames=0;last_env=None
        camera=mujoco.MjvCamera();camera.lookat[:]=[.25,0,.30];camera.distance=2.6;camera.azimuth=130;camera.elevation=-25
        writer=imageio.get_writer(out/filename,fps=6.25,codec='libx264',quality=8,macro_block_size=16)
        def capture(env,force=False):
            nonlocal renderer,frames,last_env
            last_env=env
            if not force and (env.steps-32)%4:return
            if renderer is None:renderer=mujoco.Renderer(env.model,height=480,width=640)
            renderer.update_scene(env.data,camera=camera);frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame);draw.rectangle((0,0,640,58),fill='black')
            draw.text((8,4),f'E102 case {case} | {method} | {mode} | {label}',fill='white')
            draw.text((8,21),f'EXECUTED | post-probe: {mode} | privileged route planner',fill='white')
            draw.text((8,38),f'{env.phase} | {(env.steps-32)*.04:.2f}s | collision: {env.bad} | route selected: {env.route is not None}',fill='white')
            writer.append_data(np.asarray(frame));frames+=1
            if frames==40:frame.save(out/f'{Path(filename).stem}_preview.png')
        try:
            result=evaluate(case,tasks[case],method,loaded[method],students[(mode if mode!='parent' else 'replay',int(method[-1]))],mode,capture)
            if (last_env.steps-32)%4:capture(last_env,True)
        finally:
            writer.close()
            if renderer is not None:renderer.close()
        # Compare the canonical JSON representation: saved histories encode tuples as lists.
        assert json.loads(json.dumps(result))==old
        reader=imageio.get_reader(out/filename);assert reader.count_frames()==frames;reader.close()
        manifest['videos'].append(dict(file=filename,case=case,method=method,mode=mode,success=old['success'],collision=old['collision'],route_used=old['route_used'],exact_replay=True,frames=frames,sha256=fingerprint(out/filename),reference_sha256=fingerprint(ref)))
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(filename,'verified',flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h
    html='<html><head><meta charset="utf-8"><title>E102 post-probe continuation</title></head><body style="background:#111;color:#eee;font:16px sans-serif;margin:30px"><h1>Grasp contacts during lowering and release</h1><p>'+manifest['selection']+'</p><p>'+manifest['playback']+'</p>'
    for v in manifest['videos']:html+=f'<h2>{v["file"]}</h2><video controls width="640" src="{v["file"]}"></video><p>Route used: {v["route_used"]}. Exact replay verified.</p>'
    (out/'index.html').write_text(html+'</body></html>',encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)


if __name__=='__main__':run()

