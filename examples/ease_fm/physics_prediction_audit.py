"""E107: privileged short-horizon physics prediction on training bodies only."""
import json
import shutil
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from . import single_phase as prior
from .geometry_filter import GeometryArm,allowed_pair
from .arm_pick_place import PickPlace
from .benchmark import fingerprint

ROOT=Path('outputs/ease_physics_prediction_v1')

def forbidden(model,data,gids):
    return sorted({tuple(sorted(mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or str(g) for g in (c.geom1,c.geom2))) for c in data.contact[:data.ncon] if not allowed_pair({int(c.geom1),int(c.geom2)},gids)})

def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False)
    for f in ('trials','source','videos'):(ROOT/f).mkdir()
    tasks=json.loads((prior.prior.TASKS/'cases.json').read_text());prior.AXES.update(json.loads((prior.prior.TASKS/'axis_catalog.json').read_text()))
    assert len(tasks)==12 and all(t['arm']['split']=='train' for t in tasks)
    parents,hashes=prior.prior.previous.models();students,hh=prior.prior.e99.load_students();hashes.update(hh)
    refs=[prior.ROOT/'trials'/f'{c}_{s}_None_baseline.json' for c in range(12) for s in range(3)]
    inputs={str(p):fingerprint(p) for p in [prior.prior.TASKS/'cases.json',prior.prior.TASKS/'axis_catalog.json',*refs]}
    spec=dict(experiment='E107',checkpoints=hashes,inputs=inputs,design='All12 original training-body placements x3 fixed seeds, baseline only:36 unchanged full tasks. Audit every retreat command, both clean and already-contact states. Third placement/body remains held placement validation (24 train/12 held pairs). No fresh E105 states, training, or controller intervention.',prediction='Copy full live MjData into separate data using the same unchanged physical model. Apply the already selected geometry-filter joint target and finger aperture exactly as execution. Simulate20/40/80 substeps (40/80/160ms). Hold that absolute control target throughout; mj_forward at each20-step boundary matches actual control boundary processing. At40ms require exact predicted qpos/qvel equality to actual executed state and contact-union equality. Longer horizons hold targets while real policy replans: compare them to actual future contacts only as warning usefulness, not exact-model fidelity. Never access task Jacobians.',classification='Primary clean-state binary detection at40ms. Static warning: selected clearance below2mm minus1e-6 OR filter infeasible. Physics warning: any forbidden contact at horizon. Report TP/FP/FN/TN, latency, all-steps and clean-steps separately, train/held placement separately.80/160ms outcomes require complete future horizons within retreat; exclude censored tails, report counts. These are privileged exact-model predictions and cannot establish learned dynamics generalization or effective avoidance.',videos='Reuse original E104 baseline videos with verified hashes and full-result equality. No counterfactual animation presented as executed policy.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    select_fn=GeometryArm.select_action;step_fn=PickPlace.step;all_rows=[];checks=[]
    for case,t in enumerate(tasks):
        for seed in range(3):
            rows=[];pending={};scratch={}
            def select(env,action,aperture,bias):
                selected,entry=select_fn(env,action,aperture,bias)
                if getattr(env,'phase',None)=='retreat':
                    start=time.perf_counter();d=scratch.setdefault('data',mujoco.MjData(env.model));mujoco.mj_copyData(d,env.model,env.data)
                    current=forbidden(env.model,d,env.gids);q=env.observe()['q'];d.ctrl[:env.n]=np.clip(q+np.clip(selected,-.04,.04),-2.5,2.5);d.ctrl[env.n:]=aperture
                    hits=set();pred={};first=None
                    for k in range(1,81):
                        mujoco.mj_step(env.model,d);bad=forbidden(env.model,d,env.gids);hits.update(bad)
                        if bad and first is None:first=k
                        if k%20==0:mujoco.mj_forward(env.model,d)
                        if k in (20,40,80):pred[str(k*2)]=dict(collision=bool(hits),pairs=sorted(hits),elapsed_ms=(time.perf_counter()-start)*1000)
                        if k==20:pending['qpos']=d.qpos.copy();pending['qvel']=d.qvel.copy()
                    row=dict(case=case,seed=seed,body=t['arm']['seed'],dof=env.n,validation=case%3==2,step=env.steps,clean=not current,current_contacts=current,static_warning=not entry['feasible'] or entry['selected_clearance']<.002-1e-6,filter=entry,predictions=pred,first_predicted_contact_ms=None if first is None else first*2)
                    rows.append(row);pending['row']=row
                return selected,entry
            def step(env,action,aperture):
                r=step_fn(env,action,aperture)
                if getattr(env,'phase',None)=='retreat':
                    row=pending['row'];actual=sorted({tuple(sorted(p)) for p in r[2]['bad']})
                    assert actual==row['predictions']['40']['pairs']
                    np.testing.assert_array_equal(env.data.qpos,pending['qpos']);np.testing.assert_array_equal(env.data.qvel,pending['qvel'])
                    row['actual_collision']=bool(actual);row['actual_pairs']=actual;row['exact_40ms_state']=True
                return r
            with patch.object(GeometryArm,'select_action',select),patch.object(PickPlace,'step',step):r=prior.evaluate(case,t,seed,'baseline',parents,students)
            ref=prior.ROOT/'trials'/f'{case}_{seed}_None_baseline.json';assert json.loads(json.dumps(r))==json.loads(ref.read_text())
            for i,row in enumerate(rows):
                row['future_actual']={str(n*40):any(x['actual_collision'] for x in rows[i:i+n]) if i+n<=len(rows) else None for n in (1,2,4)}
            (ROOT/'trials'/f'{case}_{seed}.json').write_text(json.dumps(dict(result=r,predictions=rows)));all_rows.extend(rows);checks.append(dict(case=case,seed=seed,success=r['success'],steps=len(rows),exact_replay=True))
        print('case',case,'complete;',len(all_rows),'predictions',flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(all_rows));(ROOT/'verification.json').write_text(json.dumps(dict(exact_replays=36,exact_40ms_states=len(all_rows),trials=checks,hashes_unchanged=True,test_evaluated=False),indent=2))
    old=json.loads((prior.ROOT/'videos/manifest.json').read_text());videos=[]
    for v in old['videos']:
        if v['mode']!='baseline':continue
        assert fingerprint(prior.ROOT/'videos'/v['file'])==v['sha256']
        shutil.copy2(prior.ROOT/'videos'/v['file'],ROOT/'videos'/v['file']);videos.append(v)
    (ROOT/'videos/manifest.json').write_text(json.dumps(dict(provenance='Unmodified E104 baseline executions; E107 full results match exactly. Prediction audit does not alter commands.',videos=videos),indent=2))
    html='<html><body><h1>E107 unchanged training-body executions</h1><p>Reused verified E104 baseline videos. Physics predictions are audit-only and do not control these motions.</p>'
    for v in videos:html+=f'<h2>{v["file"]}</h2><video controls width="640" src="{v["file"]}"></video>'
    (ROOT/'videos/index.html').write_text(html+'</body></html>')

if __name__=='__main__':run()
