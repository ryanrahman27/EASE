"""Posthoc case37 fixed-lift continuations; no controller deployment or tuning."""
from collections import deque
from pathlib import Path
import shutil
import numpy as np
import mujoco
from .matched_unseen_analyze import ROOT,old,prior,DiagnosticArm,restore,dls,bounded_action,LocalPose,Arm,AXES,geometry_ids,distances,ROT_SCALE,rotation_error
from .unseen_estimator import distance_gradient
from .abstaining_task_value_gate import read,save,fingerprint

METHODS=('e120','analytical','e120_no_correction','e120_analytical_metric','analytical_e120_metric')

def run(full_phase=False):
    horizon=100 if full_phase else 20;methods=METHODS[:3] if full_phase else METHODS;out=ROOT/('lift_phase_continuations' if full_phase else 'lift_continuations');out.mkdir(exist_ok=False)
    save(out/'protocol.json',dict(case=37,reason='Retrospective follow-up: short swaps do not explain the accumulated lift failure.',starts='Lift entry on both source trajectories' if full_phase else 'Lift entry and40 steps later, on each source trajectory.',horizon=horizon,methods=methods,control='Unchanged fixed lift goal/aperture, frozen bias, measured-history updates and safety projection. Each clone starts with identical full integration state and exact32 measured transitions. Stop a candidate if the original5-step reach condition is met; do not simulate later phases or claim complete-task rescue.',source_validation='Recorded source-policy continuation must exactly match every saved state until the original reach condition or horizon.',posthoc=True))
    _,models,hashes=old.broad.setup();AXES.update(read(ROOT/'axis_catalog.json'));task=read(ROOT/'cases.json')[37];rows=[];source_steps=0;source_continuations=0
    for source in ('e120_ensemble','analytical'):
        folder=ROOT/f'replay_{source}';stem=f'37_{source}_frozen_bias';logs=read(folder/'clearance'/f'{stem}.json');saved=np.load(folder/'states'/f'{stem}.npz')['states'];first=next(i for i,l in enumerate(logs) if l['phase']=='lift');env=DiagnosticArm(Arm(**task['arm']),task['xy'],task['target']);env.ids,env.table=geometry_ids(env.model,6,True);scratch=mujoco.MjData(env.model)
        for start in ((first,) if full_phase else (first,first+40)):
            remaining=sum(x['phase']=='lift' for x in logs[start:]);assert all(l['phase']=='lift' and not l['route_active'] and not l['handoff_control'] for l in logs[start:start+min(horizon,remaining)]);l=logs[start];h=[]
            for i in range(start-32,start):
                restore(env,np.array(logs[i]['integration_state']));before=env.observe();env.data.qpos[:]=saved[i,:env.model.nq];env.data.qvel[:]=saved[i,env.model.nq:];mujoco.mj_forward(env.model,env.data);after=env.observe();h.append((after['q']-before['q'],np.r_[after['position']-before['position'],ROT_SCALE*(after['rotation'][:,2]-before['rotation'][:,2])]))
            for method in methods:
                restore(env,np.array(l['integration_state']));c=LocalPose(6,None);c.bias=np.array(l['bias']);c.local=True;c.history=deque(((q.copy(),x.copy()) for q,x in h),maxlen=32);c.rebuild();np.testing.assert_allclose(c.gram,l['estimator_gram'],atol=1e-12,rtol=0);np.testing.assert_allclose(c.cross,l['estimator_cross'],atol=1e-12,rtol=0)
                # Use the saved statistics at the branch point, not rounded reconstructed ones.
                c.gram=np.array(l['estimator_gram']);c.cross=np.array(l['estimator_cross']);goal=np.array(l['goal']);trace=[];branch_states=[];held=0;initial=env.observe();initial_error=float(np.linalg.norm(goal-initial['position']));object_before=env.data.xpos[env.obj].copy()
                for step in range(horizon):
                    before=env.observe();error=np.r_[goal-before['position'],ROT_SCALE*rotation_error(np.eye(3),before['rotation'])];js={}
                    for group in ('reference','correction'):
                        values=[]
                        for seed in range(3):c.model=models[f'{group}_{seed}'];values.append(c.predict()[0])
                        js[group]=np.mean(values,axis=0)
                    js['analytical']=np.linalg.solve(c.gram,c.cross).T;ar=dls(js['reference'],error,c.bias);ac=dls(js['correction'],error,c.bias);aa=dls(js['analytical'],error,c.bias);ae=bounded_action(ar,ac)
                    nominal=aa if method in ('analytical','analytical_e120_metric') else ar if method=='e120_no_correction' else ae;metric_j=js['analytical'] if method in ('analytical','e120_analytical_metric') else js['reference'];d=distances(env.model,env.data,env.ids,env.table)
                    if np.min(d)<=prior.ACTIVATION:
                        g=distance_gradient(env.model,env.data,scratch,env.ids,env.table,6);j=metric_j.astype(np.float64);action,solve,_=prior.project(nominal,c.bias,before['q'],d,g,j.T@j+.015**2*np.eye(6))
                    else:action=nominal;solve=dict(fallback=False)
                    env.minimum=d.copy();env.record_minimum=True;after,command,info=env.step(action,l['aperture']);env.record_minimum=False;bias=c.bias.copy();c.update(before,after,command);c.bias=bias;state=np.r_[env.data.qpos.copy(),env.data.qvel.copy()];branch_states.append(state);pe=float(np.linalg.norm(goal-after['position']));re=float(np.linalg.norm(rotation_error(np.eye(3),after['rotation'])));held=held+1 if pe<.008 and re<.08 else 0
                    trace.append(dict(step=step,position_error_mm=1000*pe,axis_error=re,q=after['q'].tolist(),action=action.tolist(),bad=info['bad'],pads=info['pads'],object=env.data.xpos[env.obj].tolist(),minimum_clearance_mm=1000*float(env.minimum.min()),fallback=solve['fallback']))
                    if method==('e120' if source=='e120_ensemble' else 'analytical'):
                        assert logs[start+step]['phase']=='lift';np.testing.assert_allclose(state,saved[start+step],atol=1e-12,rtol=0);source_steps+=1
                    if held>=5:break
                if method==('e120' if source=='e120_ensemble' else 'analytical'):
                    source_continuations+=1;assert len(trace)==min(horizon,remaining)
                name=f'{source}_{start}_{method}';save(out/f'{name}.json',trace);np.savez_compressed(out/f'{name}.npz',states=np.array(branch_states));rows.append(dict(source=source,start=start,method=method,steps=len(trace),initial_error_mm=1000*initial_error,final_error_mm=trace[-1]['position_error_mm'],progress_mm=1000*initial_error-trace[-1]['position_error_mm'],final_axis_error=trace[-1]['axis_error'],collision=any(t['bad'] for t in trace),lift_phase_reached=held>=5,final_object_z_mm=1000*trace[-1]['object'][2],object_xy_motion_mm=1000*float(np.linalg.norm(np.array(trace[-1]['object'][:2])-object_before[:2]))))
                print(rows[-1],flush=True)
    save(out/'summary.json',rows)
    for p,h in hashes.items():assert fingerprint(p)==h
    save(out/'verification.json',dict(exact_source_continuations=source_continuations,source_steps_verified=source_steps,paired_branches=len(rows),horizon=horizon,no_full_task_success_claim=True,weights_unchanged=True));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    title='## Case37: full original lift-budget follow-up' if full_phase else '## Case37: posthoc matched lift continuations'
    lines=['',title,'',f'Posthoc phase-local comparison with a{horizon}-step maximum and the original5-step reach rule. All candidates share the full starting state and exact32-transition history. The lift goal, aperture, bias, update rule and safety constraints are unchanged. These are not full task evaluations or controller improvements. {source_continuations} source continuations reproduce all{source_steps} saved states to1e-12.','','| Source / start | Candidate | Initial error (mm) | Final error (mm) | Final axis error | Collision | Lift phase reached |','|---|---|---:|---:|---:|---|---|']
    for r in rows:lines.append(f"| {r['source']} / {r['start']} | {r['method']} | {r['initial_error_mm']:.2f} | {r['final_error_mm']:.2f} | {r['final_axis_error']:.3f} | {r['collision']} | {r['lift_phase_reached']} |")
    p=ROOT/'report.md';p.write_text(p.read_text(encoding='utf-8')+'\n'.join(lines)+'\n',encoding='utf-8')

if __name__=='__main__':
    import sys
    run(len(sys.argv)>1 and sys.argv[1]=='full_phase')
