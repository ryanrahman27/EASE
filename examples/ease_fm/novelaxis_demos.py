"""Recorded exact replays: disclose illustrative success/failure selection."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import patch
import imageio.v2 as imageio
from PIL import Image, ImageDraw
import mujoco
import torch
from .novelaxis_oracle import ExactPose, ROOT, PRIOR
from .geometry_filter import GeometryArm
from .direction_newbody import AblationArm
from .direction_fulltask import sequence
from .arm_pick_place import rollout
from .mixed_sim import AXES,Arm
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .real_arm_reach import load_models
from .benchmark import fingerprint


def run():
    torch.set_num_threads(4)
    out=ROOT/'videos';out.mkdir(exist_ok=False)
    AXES.update(json.loads((PRIOR/'axis_catalog.json').read_text()))
    cases=json.loads((PRIOR/'cases.json').read_text())
    models,hashes=load_models('outputs/ease_estimator_distillation_v1/analytical')
    selection=[(6,'baseline'),(6,'response'),(2,'exact_jacobian'),(4,'exact_jacobian')]
    manifest=dict(selection='Illustrative outcome-selected examples, not a random sample: matched E87 candidate_0 case6 failure/recovery; E88 exact-Jacobian case2 success and case4 failure. Full quantitative results include every case.',
        playback='Real-time simulation speed: every second 40ms control step at12.5fps. Calibration omitted; complete task shown without cuts. Labels disclose final outcome, phase, time and cumulative contact. No audio.',videos=[])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for case,mode in selection:
        t=cases[case];is_exact=mode=='exact_jacobian'
        reference=(ROOT/'trials'/f'{case}_{mode}.json') if is_exact else (PRIOR/'trials'/f'{case}_candidate_0_{mode}.json')
        old=json.loads(reference.read_text());label='SUCCESS' if old['success'] else 'FAILURE'
        filename=f'case{case}_{mode}_{label.lower()}.mp4';path=out/filename
        env=(GeometryArm if is_exact else AblationArm)(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry()
        if not is_exact:env.initialize_response()
        c=ExactPose(env) if is_exact else GuardedPose(env.n,models['neural_0'])
        assert calibrate(env,t['home'],c,872000+t['candidate'])==json.loads((PRIOR/f'probe_{case}.json').read_text())
        env.controller=c;env.filter_enabled=True;env.filter_log=[]
        if not is_exact:
            env.correction_log=[];env.mode='baseline' if mode=='baseline' else 'sustained';env.command_ablation=False;env.first_warning=None;env.pulse_used=0
        renderer=mujoco.Renderer(env.model,height=480,width=640)
        camera=mujoco.MjvCamera();camera.lookat[:]=[.25,0,.30];camera.distance=1.25;camera.azimuth=130;camera.elevation=-25
        writer=imageio.get_writer(path,fps=12.5,codec='libx264',quality=8,macro_block_size=16)
        original=env.step;frame_count=0
        def captured_step(action,aperture):
            nonlocal frame_count
            result=original(action,aperture)
            if (env.steps-32)%2==0:
                renderer.update_scene(env.data,camera=camera)
                frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame)
                draw.rectangle((0,0,640,54),fill='black')
                draw.text((8,5),f'E88 exact kinematics | case {case} | {label}' if is_exact else f'E87 candidate_0 | {mode} | case {case} | {label}',fill='white')
                draw.text((8,22),f'{t["arm"]["family"]} | {env.phase} | {(env.steps-32)*.04:.2f}s',fill='white')
                draw.text((8,38),f'Collision so far: {env.bad} | full task, calibration omitted',fill='white')
                writer.append_data(__import__('numpy').asarray(frame));frame_count+=1
                if frame_count==1:frame.save(out/f'{path.stem}_preview.png')
            return result
        env.step=captured_step
        try:
            with (nullcontext() if is_exact else patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Task Jacobian forbidden'))):
                result=rollout(env,t['home'],c,reset=False,sequence=sequence(env))
        finally:
            writer.close();renderer.close()
        assert all(result[k]==old[k] for k in result),f'Replay differs: {filename}'
        reader=imageio.get_reader(path);decoded=reader.count_frames();reader.close();assert decoded==frame_count and decoded>0
        manifest['videos'].append(dict(file=filename,case=case,mode=mode,success=result['success'],collision=result['collision'],frames=frame_count,fps=12.5,exact_replay=True,reference=str(reference),reference_sha256=fingerprint(reference),video_sha256=fingerprint(path)))
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
        print(filename,'verified',frame_count,'frames',flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h
    html='<html><head><meta charset="utf-8"><title>E87/E88 robot demos</title></head><body style="background:#111;color:#eee;font:16px sans-serif;max-width:1000px;margin:30px auto"><h1>Unseen-arm successes and failures</h1><p>'+manifest['selection']+'</p><p>'+manifest['playback']+'</p>'
    for v in manifest['videos']:html+=f'<h2>{v["file"]}</h2><video controls preload="metadata" width="640" src="{v["file"]}"></video><p>Exact replay verified. Collision: {v["collision"]}.</p>'
    (out/'index.html').write_text(html+'</body></html>',encoding='utf-8')


if __name__=='__main__':run()
