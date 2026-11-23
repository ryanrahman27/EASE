"""E106: unchanged E105 replays with contact and filter diagnostics."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from . import fresh_retreat as prior
from .arm_pick_place import PickPlace
from .geometry_filter import GeometryArm
from .benchmark import fingerprint

ROOT=Path('outputs/ease_retreat_contact_audit_v1')

def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False)
    for folder in ('trials','source','videos'):(ROOT/folder).mkdir()
    changes=json.loads((prior.ROOT/'summary.json').read_text())['changes'];pairs=sorted({(x['case'],x['seed']) for x in changes})
    tasks=json.loads((prior.ROOT/'cases.json').read_text());prior.AXES.update(json.loads((prior.ROOT/'axis_catalog.json').read_text()))
    parents,hashes=prior.retained.previous.models();students,hh=prior.retained.load_students();hashes.update(hh)
    inputs={str(p):fingerprint(p) for p in (prior.ROOT/'summary.json',prior.ROOT/'cases.json',prior.ROOT/'axis_catalog.json')}
    spec=dict(experiment='E106',pairs=pairs,checkpoints=hashes,inputs=inputs,design='All E105 changed pairs, all three modes. Observe existing contacts calls at2ms during release/retreat/settle; record forbidden pair distances/forces, tool pose, q/qvel, finger positions. Wrap existing geometry-filter calls to record proposed/executed actions, measured bias and filter score. No extra physics steps or forward calls, no control change. Require exact complete serialized E105 replay.',limits='Outcome-selected diagnostic. Static geometric filter scores are not dynamic clearance guarantees. MuJoCo contact arrays may describe solver-stage geometry. No causal intervention, tuning or independent success-rate estimate. Reuse exact verified E105 videos, clearly labelled.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    contact_fn=PickPlace.contacts;select_fn=GeometryArm.select_action;summary=[]
    for case,seed in pairs:
        for mode in prior.MODES:
            samples=[];actions=[];last=[None]
            def contacts(env):
                result=contact_fn(env)
                if getattr(env,'phase',None) in ('release','retreat','settle') and env.data.time!=last[0]:
                    last[0]=env.data.time;obs=env.observe();bad=[]
                    for i,c in enumerate(env.data.contact[:env.data.ncon]):
                        names=sorted(mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or str(g) for g in (c.geom1,c.geom2))
                        if tuple(names) not in {tuple(sorted(p)) for p in result[1]}:continue
                        force=np.zeros(6);mujoco.mj_contactForce(env.model,env.data,i,force)
                        bad.append(dict(pair=names,distance=float(c.dist),normal_force=float(force[0])))
                    samples.append(dict(time=float(env.data.time),step=env.steps,phase=env.phase,tool=obs['position'].tolist(),tool_z=obs['rotation'][:,2].tolist(),q=obs['q'].tolist(),qvel=env.data.qvel[:env.n].tolist(),fingers=env.data.qpos[env.n:env.n+2].tolist(),bad=bad))
                return result
            def select(env,action,aperture,bias):
                selected,entry=select_fn(env,action,aperture,bias)
                if getattr(env,'phase',None) in ('release','retreat','settle'):
                    obs=env.observe();actions.append(dict(time=float(env.data.time),step=env.steps,phase=env.phase,proposed=np.asarray(action).tolist(),selected=selected.tolist(),bias=np.asarray(bias).tolist(),q=obs['q'].tolist(),qvel=env.data.qvel[:env.n].tolist(),tool=obs['position'].tolist(),**entry))
                return selected,entry
            with patch.object(PickPlace,'contacts',contacts),patch.object(GeometryArm,'select_action',select):
                r=prior.evaluate(case,tasks[case],seed,mode,parents,students)
            phase=None if mode=='baseline' else 'retreat';ref=prior.ROOT/'trials'/f'{case}_{seed}_{phase}_{mode}.json';inputs[str(ref)]=fingerprint(ref)
            assert json.loads(json.dumps(r))==json.loads(ref.read_text())
            stem=f'{case}_{seed}_{mode}';(ROOT/'trials'/f'{stem}.json').write_text(json.dumps(dict(result=r,contacts=samples,actions=actions)))
            retreat=[x for x in samples if x['phase']=='retreat'];aa=[x for x in actions if x['phase']=='retreat'];first=next((x for x in retreat if x['bad']),None)
            before=None if first is None else next((x for x in reversed(aa) if x['time']<first['time']),None)
            row=dict(case=case,seed=seed,mode=mode,success=r['success'],samples=len(samples),retreat_samples=len(retreat),retreat_start=aa[0],first_contact=first,preceding_filter=before,retreat_min_tool_z=min(x['tool'][2] for x in retreat),retreat_start_tool_z=aa[0]['tool'][2],retreat_filter_fallbacks=sum(not x['feasible'] for x in aa),retreat_filter_changes=sum(x['changed'] for x in aa))
            summary.append(row);print(stem,'exact replay',len(samples),'samples',flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));(ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    original=json.loads((prior.ROOT/'videos/manifest.json').read_text());videos=[]
    for v in original['videos']:
        if (v['case'],v['seed']) not in pairs:continue
        assert fingerprint(prior.ROOT/'videos'/v['file'])==v['sha256'] and v['exact_replay']
        shutil.copy2(prior.ROOT/'videos'/v['file'],ROOT/'videos'/v['file']);videos.append(v)
    (ROOT/'videos/manifest.json').write_text(json.dumps(dict(provenance='Unmodified E105 full-task replay-verified videos; E106 instrumentation reproduces the exact same full results.',videos=videos),indent=2))
    html='<html><body style="background:#111;color:#eee;font:16px sans-serif"><h1>E106 contact diagnosis</h1><p>Unmodified E105 executed videos. E106 diagnostic replays exactly match these trajectories.</p>'
    for v in videos:html+=f'<h2>{v["file"]}</h2><video controls width="640" src="{v["file"]}"></video>'
    (ROOT/'videos/index.html').write_text(html+'</body></html>')
    (ROOT/'verification.json').write_text(json.dumps(dict(exact_replays=len(summary),contact_samples=sum(x['samples'] for x in summary),hashes_unchanged=True,reused_verified_videos=len(videos),no_control_change=True,test_evaluated=False),indent=2))

if __name__=='__main__':run()
