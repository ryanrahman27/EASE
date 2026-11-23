"""E175 causal near-object/tilt guard; thresholds selected on fitting references."""
import numpy as np
import mujoco
from .early_object_diagnosis import ROOT,REFERENCE,pad_separation
from .object_preserving_plan import plan
from .abstaining_task_value_gate import read,save,fingerprint

def early_gate(history):
    if len(history)<5:return False
    x=history[-1];previous=history[-5]
    return bool(x['pad_gap']<.012 and x['axis']>.15 and x['error']>.015 and x['axis']-previous['axis']>.05 and previous['pad_gap']-x['pad_gap']>.015)

def observation(model,data,goal):
    site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_center')
    return dict(pad_gap=pad_separation(model,data),axis=float(np.linalg.norm(data.site_xmat[site].reshape(3,3)[:,2]-[0,0,1])),error=float(np.linalg.norm(np.asarray(goal)-data.site_xpos[site])))

def run():
    rows=[]
    for r in read(ROOT/'reference_descent_states.json'):
        history=[];hit=None
        for x in r['series']:
            history.append(x)
            if x['route']:break
            if early_gate(history):hit=x;break
        if hit is None:continue
        case=r['case'];stem=f'{case}_e120_ensemble_frozen_bias';l=read(REFERENCE/'clearance'/f'{stem}.json')[hit['step']];m=mujoco.MjModel.from_xml_path(str(REFERENCE/'states'/f'{stem}.xml'));result=plan(m,l['full_qpos_before'],l['full_qvel_before'],l['goal']);(ROOT/'plans').mkdir(exist_ok=True);save(ROOT/'plans'/f'{case}.json',result);rows.append(dict(case=case,trigger=hit,available=result['available'],nodes=len(result['dense_path']),failure=result.get('failure')));print(rows[-1],flush=True)
    save(ROOT/'planning_summary.json',rows)
    save(ROOT/'planning_protocol.json',dict(experiment='E175',scope='Fitting reference-state diagnostic before interventions. Thresholds designed from these fitting histories, not a held evaluation.',guard='During descent before any plan, last5 measured states: SAT finger/cube gap<12mm, tool Z-axis error>.15, position error>15mm; gap closes>15mm and axis error increases>.05 across4 steps (160ms). No case or embodiment IDs in controller. Existing late stall recovery retained.',planning='Unmodified E174 strict grasp-first IK/reverse approach/geometric connection. No contact exceptions or new dynamics privileges. Current object pose/geometry explicitly privileged.',reference_sha256=fingerprint(REFERENCE/'records.json')))

if __name__=='__main__':run()
