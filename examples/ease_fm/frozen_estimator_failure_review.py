"""Describe paired failures without changing the frozen controller."""
from pathlib import Path
import shutil
import numpy as np
import mujoco
from .frozen_estimator_comparison import ROOT
from .abstaining_task_value_gate import read,save

def run():
    paired=read(ROOT/'paired.json');rows=[]
    for pair in paired['cases']:
        if pair['e120_success']==pair['analytical_success'] and pair['e120_collision']==pair['analytical_collision']:continue
        case=pair['case'];details={}
        for method in ('e120_ensemble','analytical'):
            root=ROOT/method;stem=f'{case}_{method}_frozen_bias';t=read(root/'trials'/f'{stem}.json');logs=read(root/'clearance'/f'{stem}.json');m=mujoco.MjModel.from_xml_path(str(root/'states'/f'{stem}.xml'));d=mujoco.MjData(m);site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'object');close=next(l for l in logs if l['phase']=='close');d.qpos[:]=close['full_qpos_before'];d.qvel[:]=close['full_qvel_before'];mujoco.mj_forward(m,d)
            handoff=dict(ready=close['close_ready'],object_xy=d.xpos[obj,:2].tolist(),tool_position=d.site_xpos[site].tolist(),axis_error=float(np.linalg.norm(d.site_xmat[site].reshape(3,3)[:,2]-[0,0,1])))
            if close['pick_goal'] is not None:
                pick=np.array(close['pick_goal']);gap=d.site_xpos[site]-pick;handoff.update(pick_goal=pick.tolist(),xy_gap_mm=1000*float(np.linalg.norm(gap[:2])),z_gap_mm=1000*float(gap[2]),object_drift_mm=1000*float(np.linalg.norm(d.xpos[obj,:2]-pick[:2])))
            i=next((i for i,x in enumerate(t['trace']) if x['bad']),None);contact=None
            if i is not None:
                l=logs[i];contact=dict(step=i,phase=l['phase'],pairs=t['trace'][i]['bad'],distance_before_mm=(1000*np.array(l['distance_before'])).tolist(),predicted_distance_mm=None if l['predicted_distance'] is None else (1000*np.array(l['predicted_distance'])).tolist(),actual_distance_after_mm=(1000*np.array(l['distance_after'])).tolist(),minimum_substep_distance_mm=(1000*np.array(l['substep_min'])).tolist(),solver=l['solve'],eligible=l['eligible'],allowance_mm=(1000*np.array(l['allowance'])).tolist())
            trigger=next((l for l in logs if l['planning_attempted']),None);details[method]=dict(success=t['success'],collision=t['collision'],close_entry=handoff,aborted=logs[-1]['aborted'],trigger_reason=None if trigger is None else trigger['trigger_reason'],trigger_descent_observation=None if trigger is None else trigger['descent_observations'],first_forbidden_contact=contact,first_action=logs[0]['final_action'])
        rows.append(dict(case=case,methods=details,initial_action_difference_rad=float(np.max(np.abs(np.array(details['e120_ensemble']['first_action'])-details['analytical']['first_action'])))))
    save(ROOT/'failure_review.json',rows)
    lines=['','## Frozen-comparison failure review','','No controller or estimator settings were changed after observing these outcomes. Distances below use the same conservative geometry measure as the safety controller; the strict physical collision score is recorded separately.']
    for r in rows:
        lines+=['',f"### Case{r['case']}",'',f"First executed-action difference from the same calibrated state: {r['initial_action_difference_rad']:.6g} rad (maximum joint difference). Later histories differ, so downstream observations do not isolate a single estimator error."]
        for method,x in r['methods'].items():
            h=x['close_entry'];lines+=['',f"{method}: success={x['success']}, collision={x['collision']}; recovery={x['trigger_reason']} at descent observation{x['trigger_descent_observation']}; aborted={x['aborted']}."]
            if 'xy_gap_mm' in h:lines.append(f"Close-entry measurements: XY gap={h['xy_gap_mm']:.3f}mm (limit8), Z gap={h['z_gap_mm']:.3f}mm (absolute limit12), axis error={h['axis_error']:.4f} (limit.08), object drift={h['object_drift_mm']:.3f}mm (limit8); ready={h['ready']}.")
            c=x['first_forbidden_contact']
            if c:lines.append(f"First forbidden contact: step{c['step']}, phase{c['phase']}, pairs={c['pairs']}. Safety projection eligible={c['eligible']}, solver success={c['solver']['solver_success']}, fallback={c['solver']['fallback']}. Full pre/predicted/post/substep clearance vectors are in failure_review.json (six links, palm, left pad, right pad). Passing a local linear constraint is not a guarantee of collision-free physical tracking.")
    p=ROOT/'report.md';s=p.read_text(encoding='utf-8')
    if '## Frozen-comparison failure review' not in s:p.write_text(s+'\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(rows,flush=True)

if __name__=='__main__':run()
