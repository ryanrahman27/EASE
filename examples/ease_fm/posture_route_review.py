"""Separate route arrival from object retention and full-task completion."""
import shutil
import numpy as np
from .posture_route import ROOT,CONDITIONS
from .abstaining_task_value_gate import read,save

def run():
    summary=read(ROOT/'summary.json');details={}
    for c in CONDITIONS:
        items=[]
        for case in (26,62,65,67):
            stem=f'{case}_e120_ensemble_frozen_bias';t=read(ROOT/c/'trials'/f'{stem}.json');logs=read(ROOT/c/'clearance'/f'{stem}.json');indices=[i for i,x in enumerate(t['trace']) if x['phase']=='descend'];trigger=next(i for i,l in enumerate(logs) if l['route_active']);end=indices[-1];close=[x for x in t['trace'] if x['phase']=='close'];original=np.array(t['trace'][indices[0]]['object']);at_trigger=np.array(t['trace'][trigger-1]['object']);after=np.array(t['trace'][end]['object']);route=read(ROOT/c/'planning'/f'{case}.json');last=logs[end]
            items.append(dict(case=case,success=t['success'],collision=t['collision'],grasped=t['grasped'],lifted=t['lifted'],descent_reached=next(p['reached'] for p in t['phases'] if p['phase']=='descend'),descent_steps=len(indices),terminal_descent_error_mm=1000*t['trace'][end]['position_error'],object_displacement_descent_mm=1000*float(np.linalg.norm(after[:2]-original[:2])),object_displacement_after_route_trigger_mm=1000*float(np.linalg.norm(after[:2]-at_trigger[:2])),object_distance_from_goal_at_trigger_mm=1000*float(np.linalg.norm(at_trigger[:2]-np.array(route['goal'])[:2])),object_distance_from_goal_after_descent_mm=1000*float(np.linalg.norm(after[:2]-np.array(route['goal'])[:2])),close_tool_object_xy_distance_mm=1000*float(np.linalg.norm(np.array(close[-1]['position'])[:2]-np.array(close[-1]['object'])[:2])),route_nodes=len(route['dense_path']),last_route_node=last['route_index'],max_descent_pad_contacts=max(t['trace'][i]['pads'] for i in indices)))
        details[c]=items
    save(ROOT/'object_handoff_diagnosis.json',details)
    lines=['','## Dynamic route and grasp handoff','','| Variant / case | Descent reached | Descent steps | Descent error (mm) | Object moved during descent (mm) | Closing tool/object XY gap (mm) | Grasped / lifted | Task success |','|---|---|---:|---:|---:|---:|---|---|']
    for c,items in details.items():
        for x in items:lines.append(f"| {c} / {x['case']} | {x['descent_reached']} | {x['descent_steps']} | {x['terminal_descent_error_mm']:.2f} | {x['object_displacement_descent_mm']:.2f} | {x['close_tool_object_xy_distance_mm']:.2f} | {x['grasped']} / {x['lifted']} | {x['success']} |")
    lines+=['','The planner preserves the original forbidden-contact definition: finger-pad/cube contact is allowed. Thus an accepted static path may touch the cube, while static checks do not integrate its subsequent displacement. Joint tracking and contact dynamics can also deviate from the sampled path. Actual object motion and full-task scoring above remain decisive; endpoint/path feasibility is not counted as a successful pick-and-place.','', 'The long-budget result should be interpreted together with its extra interactions and grasp outcome. A route can solve the local arm-pose blockage while moving the object away from the fixed grasp target. An object-preserving approach and explicit grasp handoff would be a separate privileged control intervention requiring a new frozen fitting comparison, not a reason to relabel these failures.','', 'Publication implication: these results diagnose where controller/task assumptions fail. They do not establish a new learned-adaptation advantage. The next publication-critical experiment remains a frozen comparison of learned versus analytical history under identical control/privilege on untouched kinematic embodiments. See publication_readiness.md.']
    p=ROOT/'report.md';text=p.read_text(encoding='utf-8')
    if '## Dynamic route and grasp handoff' not in text:p.write_text(text+'\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copy2(__file__,ROOT/'source'/__file__.split('/')[-1]);print(details,flush=True)

if __name__=='__main__':run()
