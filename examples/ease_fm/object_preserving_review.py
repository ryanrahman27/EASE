"""E174 actual object preservation, grasp decisions and comparison to E173."""
import shutil
from pathlib import Path
import numpy as np
from .object_preserving import ROOT,CONDITIONS
from .abstaining_task_value_gate import read,save

def run():
    prior=read('outputs/ease_posture_route_v1/object_handoff_diagnosis.json')['route_extended'];prior={r['case']:r for r in prior};details={}
    for c in CONDITIONS:
        rows=[]
        for case in (26,62,65,67):
            stem=f'{case}_e120_ensemble_frozen_bias';t=read(ROOT/c/'trials'/f'{stem}.json');logs=read(ROOT/c/'clearance'/f'{stem}.json');desc=[i for i,l in enumerate(logs) if l['phase']=='descend'];close=[i for i,l in enumerate(logs) if l['phase']=='close'];active=[i for i,l in enumerate(logs) if l['route_active']];a=np.array(t['trace'][desc[0]]['object']);b=np.array(t['trace'][desc[-1]]['object']);lastclose=logs[close[-1]];traceclose=t['trace'][close[-1]]
            rows.append(dict(case=case,route_executed=bool(active),success=t['success'],collision=t['collision'],grasped=t['grasped'],lifted=t['lifted'],descend_steps=len(desc),descent_reached=next(p['reached'] for p in t['phases'] if p['phase']=='descend'),terminal_descent_error_mm=1000*t['trace'][desc[-1]]['position_error'],object_displacement_descent_mm=1000*float(np.linalg.norm(b[:2]-a[:2])),previous_route_object_displacement_mm=prior[case]['object_displacement_descent_mm'],close_xy_gap_mm=1000*float(np.linalg.norm(np.array(traceclose['position'])[:2]-np.array(traceclose['object'])[:2])),close_ready=lastclose['close_ready'],final_sustained_two_pad_steps=lastclose['close_good'],closed_aperture_steps=sum(logs[i]['aperture']>0 for i in close),aborted=logs[-1]['aborted'],failed_phases=[p['phase'] for p in t['phases'] if not p['reached']]))
        details[c]=rows
    save(ROOT/'object_handoff_diagnosis.json',details)
    lines=['','## Actual object motion and grasp handoff','','| Variant / case | Route | Task success | Object moved (mm) | Previous route moved (mm) | Close XY gap (mm) | Final two-pad steps | Abort |','|---|---|---|---:|---:|---:|---:|---|']
    for c,rows in details.items():
        for x in rows:lines.append(f"| {c} / {x['case']} | {x['route_executed']} | {x['success']} | {x['object_displacement_descent_mm']:.2f} | {x['previous_route_object_displacement_mm']:.2f} | {x['close_xy_gap_mm']:.2f} | {x['final_sustained_two_pad_steps']} | {x['aborted']} |")
    lines+=['','Object movement is measured from the first to last executed descent state; it includes pre-trigger motion. Case26 has no accepted strict route and is an exact fallback, so it is not evidence of the new route preserving the object. All endpoints and grasp guards are evaluated from measured states. Lower displacement, a reached hand pose or an aborted lift is not scored as a successful pick-and-place.','', 'The guarded and unguarded variants have identical route/probe prefixes; their divergence starts at closure. This isolates the handoff on these fitting trajectories, while the broader route changes relative to E173 also include grasp-first approach planning, current object XY and gain1 tracking. Do not attribute a between-study improvement to a single component.','', 'The related methods review is in literature_review.md. We implemented staged planning and a guarded task-state transition, not the cited cuRobo/STORM/PDDLStream/contact-MPC algorithms. All findings remain privileged fitting diagnostics with frozen E120; no unseen-embodiment or learned-policy superiority claim is made.']
    p=ROOT/'report.md';text=p.read_text(encoding='utf-8')
    if '## Actual object motion and grasp handoff' not in text:p.write_text(text+'\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(details,flush=True)

if __name__=='__main__':run()
