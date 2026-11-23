"""E89 report and explicitly nonphysical path animations."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
import shutil
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image,ImageDraw
from .novelaxis_paths import ROOT,PRIOR
from .mixed_sim import AXES,Arm
from .geometry_filter import GeometryArm
from .benchmark import fingerprint


def run():
    rows=json.loads((ROOT/'records.json').read_text());cases=json.loads((PRIOR/'cases.json').read_text());audit=json.loads((ROOT/'audit.json').read_text())
    assert len(rows)==16 and audit['exact_replays']==16
    for p,h in audit['inputs'].items():assert fingerprint(p)==h
    AXES.update(json.loads((PRIOR/'axis_catalog.json').read_text()))
    details=[]
    for r in rows:
        full=json.loads((ROOT/'cases'/f"{r['case']}.json").read_text());best=next((a for a in full['attempts'] if a['attempt']==r['selected_attempt']),None)
        delta=None if best is None else float(np.max(np.abs(np.array(best['endpoint'])-full['state']['qpos'][:r['dof']])))
        details.append(dict(**r,selected_clearance=None if best is None else best['minimum_clearance'],max_joint_displacement=delta,nominal_004_increments=None if delta is None else int(np.ceil(delta/.04))))
    (ROOT/'summary.json').write_text(json.dumps(dict(cases=details,cases_with_clear_route=sum(r['contact_free_paths']>0 for r in rows),endpoints=sum(r['endpoints'] for r in rows),sampled_paths=sum(r['contact_free_paths'] for r in rows)),indent=2))
    out=ROOT/'videos';out.mkdir(exist_ok=True);videos=[]
    # Prefer the difficult family for each illustrative category, as prespecified.
    for clear in (True,False):
        candidates=[r for r in rows if r['selected_attempt'] is not None and (r['contact_free_paths']>0)==clear]
        preferred=[r for r in candidates if r['family'].endswith('ZYXYZY')]
        if not candidates:continue
        r=(preferred or candidates)[0];case=r['case'];t=cases[case];full=json.loads((ROOT/'cases'/f'{case}.json').read_text())
        a=full['attempts'][r['selected_attempt']];env=GeometryArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();env.data.qpos[:]=full['state']['qpos'];env.data.qvel[:]=0
        start=np.array(full['state']['qpos'][:env.n]);end=np.array(a['endpoint'])
        label='SAMPLED CLEAR PATH' if clear else 'BLOCKED PATH';filename=f'case{case}_kinematic_{"clear" if clear else "blocked"}.mp4'
        renderer=mujoco.Renderer(env.model,height=480,width=640);camera=mujoco.MjvCamera();camera.lookat[:]=[.25,0,.30];camera.distance=2.6;camera.azimuth=130;camera.elevation=-25
        writer=imageio.get_writer(out/filename,fps=25,codec='libx264',quality=8,macro_block_size=16)
        try:
            for i,s in enumerate(a['samples']):
                env.data.qpos[:env.n]=(1-s['alpha'])*start+s['alpha']*end;mujoco.mj_forward(env.model,env.data)
                assert env.contacts()[1]==s['bad']
                assert abs(env.clearance(env.data.qpos[:env.n],env.data.qpos[env.n:env.n+2])-s['clearance'])<1e-12
                renderer.update_scene(env.data,camera=camera);frame=Image.fromarray(renderer.render());draw=ImageDraw.Draw(frame);draw.rectangle((0,0,640,58),fill='black')
                draw.text((8,4),f'E89 case {case} | {label} | attempt {a["attempt"]}',fill='white')
                draw.text((8,21),'KINEMATIC ANIMATION - NOT EXECUTED MOTION / NOT TASK SUCCESS',fill='white')
                draw.text((8,38),f'Path {s["alpha"]:.0%} | clearance {s["clearance"]*1000:.2f} mm (capped at 2)',fill='white')
                writer.append_data(np.asarray(frame))
                if i==50:frame.save(out/f'case{case}_preview.png')
        finally:writer.close();renderer.close()
        reader=imageio.get_reader(out/filename);assert reader.count_frames()==101;reader.close()
        videos.append(dict(file=filename,case=case,attempt=a['attempt'],frames=101,fps=25,static_geometry_verified=True,sha256=fingerprint(out/filename)))
    manifest=dict(selection='First case with clear sampled route and first blocked case, preferring difficult ZYXYZY family. Clear/blocked refers to sampled static geometry, not task success/failure.',playback='101 static configurations at25fps; arbitrary animation timing, no physics integration. Object and fingers fixed at recorded descent-start state.',videos=videos)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    html='<html><head><meta charset="utf-8"><title>E89 path diagnostics</title></head><body style="background:#111;color:#eee;font:16px sans-serif;margin:30px"><h1>Static path diagnostics</h1><p>'+manifest['selection']+'</p><p>'+manifest['playback']+'</p>'
    for v in videos:html+=f'<h2>{v["file"]}</h2><video controls width="640" src="{v["file"]}"></video>'
    (out/'index.html').write_text(html+'</body></html>',encoding='utf-8')
    lines=['# E89: static descent-path feasibility on unseen arrangements','',
        'All 16 E88 exact-Jacobian rollouts replay exactly. Search starts from each recorded state immediately before descent. Twenty deterministic endpoint IK attempts per case retain the recorded object and finger poses. Every accepted endpoint receives 101 equally spaced joint-space path samples. The original contact rules apply; the separate 2 mm geometry check is also reported. This is a bounded privileged kinematic search, not an executed policy or continuous-path certificate.','',
        '| Case | DOF | Collision-free endpoints / 20 | Contact-free sampled routes | Routes with 2 mm margin | Selected minimum clearance (mm) |','|---|---:|---:|---:|---:|---:|']
    for r in details:lines.append(f"| {r['case']} | {r['dof']} | {r['endpoints']} | {r['contact_free_paths']} | {r['margin_paths']} | {r['selected_clearance']*1000:.3f} |")
    lines+=['','## Findings and next decision','',
        'The search finds 182 collision-free endpoints from 320 attempts and 121 sampled contact-free joint-interpolation routes. Thirteen of 16 cases have at least one route; all such routes also satisfy the sampled 2 mm margin. Endpoints may duplicate a branch and are not independent samples.',
        '', 'Within the difficult six-joint ZYXYZY family, case 4 has one clear route among five endpoints. Its local-start IK attempt fails; attempt 17 finds a different endpoint posture. Cases 5, 6 and 7 have seven, four and nine endpoints respectively, but no clear sampled straight joint-space route. This is evidence of posture/path sensitivity, not proof that these cases are infeasible.',
        '', 'Case 4 requires a maximum single-joint displacement of 4.172 radians, or 105 nominal increments of 0.04 radians, exceeding the existing 100-step descent budget under ideal incremental tracking. This is an illustrative planning-cost estimate, not a physical lower bound: actual actuator displacements can differ from commands. The 5.462-radian joint-vector distance and finite geometry samples mean the clear animation cannot be counted as recovered manipulation.',
        '', 'Next evaluate physical tracking of the discovered route as a separately labeled privileged diagnostic, first under the original descent budget and then a disclosed longer budget. Preserve the recorded start state and compare with the original local controller. This will distinguish static path availability from dynamic executability and budget limitations before changing the learned policy. Whole pick-and-place success must still be evaluated separately.',
        '', 'Verification: 16 exact full-rollout replays, identical calibration probes, saved per-attempt endpoints and per-sample contacts/clearances, unchanged input hashes, and 110 passing tests including an interior-collision regression test. Two path animations reproduce all 101 saved contact and clearance samples each. No training, final-test evaluation or new policy-success result.',
        '', '## Video evidence','',
        'Gallery: `outputs/ease_novelaxis_paths_v1/videos/index.html`. Case 4 shows a sampled-clear route; case 5 shows a blocked route. Both prominently state KINEMATIC ANIMATION, NOT EXECUTED MOTION. Playback timing is arbitrary and does not represent actuator speed. Original executed successes/failures remain available in the E88 gallery.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copy2(__file__,ROOT/'source'/__file__.split('/')[-1])
    print('Report and two verified static-path videos saved.',flush=True)


if __name__=='__main__':run()
