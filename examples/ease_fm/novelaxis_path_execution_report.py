"""E90 quantitative report and exact-replay executed task videos."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
import shutil
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from .novelaxis_path_execution import ROOT,PRIOR,evaluate
from .mixed_sim import AXES
from .benchmark import fingerprint


def run():
    rows=json.loads((ROOT/'records.json').read_text());audit=json.loads((ROOT/'audit.json').read_text())
    assert len(rows)==48 and audit['exact_baselines']==16
    spec=json.loads((ROOT/'protocol.json').read_text())
    for p,h in spec['inputs'].items():assert fingerprint(p)==h
    table=[];changes=[]
    for group in ('all','6','7'):
        for mode in ('baseline','route100','route300'):
            rr=[r for r in rows if r['mode']==mode and (group=='all' or str(r['dof'])==group)]
            table.append(dict(group=group,mode=mode,trials=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),
                descent_reached=sum(next(p for p in r['phases'] if p['phase']=='descend')['reached'] for r in rr),steps=sum(r['steps'] for r in rr)))
    for b in rows:
        if b['mode']!='baseline':continue
        for mode in ('route100','route300'):
            r=next(r for r in rows if r['case']==b['case'] and r['mode']==mode)
            if r['success']!=b['success']:changes.append(dict(case=r['case'],mode=mode,change='gain' if r['success'] else 'loss'))
    (ROOT/'summary.json').write_text(json.dumps(dict(table=table,changes=changes),indent=2))
    out=ROOT/'videos';out.mkdir(exist_ok=True);manifest=dict(selection='Cases0 and4 in all three modes, fixed in protocol before evaluation. Executed physics; complete task follows original approach/calibration. Calibration omitted from video. Not a representative success-rate sample.',playback='Every fourth40ms control step at6.25fps, with final state included. Near-real-time simulated playback, no cuts or audio.',videos=[])
    tasks=json.loads((PRIOR/'cases.json').read_text());AXES.update(json.loads((PRIOR/'axis_catalog.json').read_text()))
    for case in (0,4):
        for mode in ('baseline','route100','route300'):
            ref=ROOT/'trials'/f'{case}_{mode}.json';old=json.loads(ref.read_text());label='SUCCESS' if old['success'] else 'FAILURE';filename=f'case{case}_{mode}_{label.lower()}.mp4'
            renderer=None;camera=mujoco.MjvCamera();camera.lookat[:]=[.25,0,.30];camera.distance=2.6;camera.azimuth=130;camera.elevation=-25
            writer=imageio.get_writer(out/filename,fps=6.25,codec='libx264',quality=8,macro_block_size=16);frames=0;last_env=None
            def capture(env,force=False):
                nonlocal renderer,frames,last_env
                last_env=env
                if not force and (env.steps-32)%4:return
                if renderer is None:renderer=mujoco.Renderer(env.model,height=480,width=640)
                renderer.update_scene(env.data,camera=camera);frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame);draw.rectangle((0,0,640,58),fill='black')
                draw.text((8,4),f'E90 case {case} | {mode} | {label} | EXECUTED PHYSICS',fill='white')
                draw.text((8,21),'Privileged control diagnostic (not a learned-policy result)',fill='white')
                draw.text((8,38),f'{env.phase} | {(env.steps-32)*.04:.2f}s | collision: {env.bad} | waypoint: {env.waypoint}',fill='white')
                writer.append_data(np.asarray(frame));frames+=1
                if frames==40:frame.save(out/f'{Path(filename).stem}_preview.png')
            try:
                result=evaluate(case,tasks[case],mode,capture)
                if (last_env.steps-32)%4:capture(last_env,True)
            finally:
                writer.close()
                if renderer is not None:renderer.close()
            assert result==old
            reader=imageio.get_reader(out/filename);assert reader.count_frames()==frames;reader.close()
            manifest['videos'].append(dict(file=filename,case=case,mode=mode,success=old['success'],collision=old['collision'],exact_replay=True,frames=frames,sha256=fingerprint(out/filename),reference_sha256=fingerprint(ref)))
            (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(filename,'verified',flush=True)
    html='<html><head><meta charset="utf-8"><title>E90 executed path trials</title></head><body style="background:#111;color:#eee;font:16px sans-serif;margin:30px"><h1>Executed route tracking: successes and failures</h1><p>'+manifest['selection']+'</p><p>'+manifest['playback']+'</p>'
    for v in manifest['videos']:html+=f'<h2>{v["file"]}</h2><video controls width="640" src="{v["file"]}"></video>'
    (out/'index.html').write_text(html+'</body></html>',encoding='utf-8')
    lines=['# E90: physical execution of frozen descent routes','',
        'All 16 fixed cases run in three conditions: E88 exact-Jacobian baseline; frozen E89 joint-waypoint tracking during descent with the original100-step budget; and the same tracker with300 descent steps. All other phases, budgets, measured bias updates, physics, geometry filtering and full-task success criteria are unchanged. The13 cases with a sampled-clear selected route receive the intervention; the other three fall back to the original controller with original budgets. No controller tuning occurred after outcomes.',
        '', '| Group | Mode | Full-task success | Collisions | Descent target reached | Steps including calibration |','|---|---|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['trials']} | {r['collisions']} | {r['descent_reached']} | {r['steps']} |")
    lines+=['','Paired success changes:','']
    for r in changes:lines.append('- '+json.dumps(r))
    lines+=['','## Interpretation and next decision','',
        'Full-task success increases from8/16 to11/16 under the original descent budget and13/16 with the longer budget. Route100 recovers cases1,12,13; route300 additionally recovers cases4,8. No baseline successes are lost. The difficult-family case4, previously only a static-clear route, now completes physical pick-and-place with300 descent steps. Under100 steps it reaches waypoint50/100, then proceeds through the unchanged phase sequence and fails with contact. Cases5-7 have no selected clear route and remain unchanged failures.',
        '', 'This establishes dynamic executability of the selected case4 route and shows that the specific waypoint tracker needs more than the original budget. It does not show that every possible controller requires that larger budget. Improved downstream outcomes after only a descent intervention support posture/path dependence, but do not isolate a single cause for every recovered case.',
        '', 'These are privileged development results, not improved learned-policy scores: exact Jacobians and offline geometry/IK route search are used, and E89 route selection is informed by these development cases. The longer-budget condition is explicitly not interaction-matched. Neither condition constitutes new held-out generalization evidence. Learned E87 scores remain46/96 with response correction.',
        '', 'Next separate the two requirements: test whether the frozen learned motion estimators can complete the remaining phases after the same privileged route intervention, with the planner contribution clearly disclosed; independently investigate multi-segment routes for cases5-7. A matched learned comparison should preserve original and longer budgets separately and retain all cases. Do not train on these outcomes and then describe these cases as unseen evaluation.',
        '', 'Verification:48 saved full rollouts;16 exact E88 baseline replays; six additional unchanged no-route runs; exact descent-start qpos/qvel across interventions; unchanged input hashes;112 passing tests, including waypoint waiting/clipping and non-descent action preservation. Six executed full-task videos reproduce every saved result and trace exactly; decoded frame counts and video hashes are recorded.',
        '', '## Saved videos','', 'Gallery: `outputs/ease_novelaxis_path_execution_v1/videos/index.html`. Cases0 and4 in all three modes were selected before evaluation. Labels disclose executed physics, privileged control, final outcome and cumulative collision. These are full-task executions, unlike the E89 static path animations.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print(json.dumps(table[:3],indent=2))


if __name__=='__main__':run()
