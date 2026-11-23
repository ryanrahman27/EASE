"""E80 one-step avoidance and successful-control retention summary."""
import json
import shutil
from pathlib import Path
import numpy as np
from .action_retention_audit import ROOT
from .benchmark import fingerprint


def run():
    audit=json.loads((ROOT/'audit.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text())
    for p,h in {**audit['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    rows=[]
    for p in (ROOT/'trials').glob('*.json'):
        t=json.loads(p.read_text())
        rows.extend(dict(case=t['case'],method=t['method'],dof=t['dof'],task_success=t['success'],**r) for r in t['attribution'])
    assert len(rows)==136 and sum(r['branches']['selected']['collision'] for r in rows)==26
    table=[]
    for group in ('collision','collision_6','collision_7','successful_controls'):
        rr=[r for r in rows if (r['task_success'] if group=='successful_controls' else r['branches']['selected']['collision'] and (group=='collision' or r['dof']==int(group[-1])))]
        for name in ('selected','half','quarter','bias_only','analytical_same_history'):
            delta=np.array([r['branches'][name]['task_error']-r['branches']['selected']['task_error'] for r in rr])
            table.append(dict(group=group,action=name,states=len(rr),contacts=sum(r['branches'][name]['collision'] for r in rr),
                mean_task_error_delta=float(delta.mean()),median_task_error_delta=float(np.median(delta)),max_task_error_delta=float(delta.max()),
                worse_than_selected=int(sum(delta>1e-9)),loses_all_pads_vs_selected=sum(r['branches']['selected']['pads']>0 and r['branches'][name]['pads']==0 for r in rr)))
    (ROOT/'summary.json').write_text(json.dumps(dict(table=table,states=len(rows),branches=5*len(rows),exact_replays=audit['exact_replays']),indent=2))
    lines=['# E80: bounded action counterfactuals with successful controls','',
        'Compare five actions from identical states at every E78 learned first contact (26 states) and every response warning on successful learned runs (110 states from two six-joint trajectories). Each branch advances one 40 ms control step with original incoming velocity, finger aperture and physics. The selected action reproduces the original observation and contact label exactly. Full reference trajectories and filter logs also replay exactly.','',
        'Actions: selected geometry-filter action; half or quarter of its task-motion component while preserving clipped measured bias; bias-only; analytical Jacobian from the same measured Gram/cross history and bias. All commands remain clipped to +/-0.04. Alternatives bypass the geometry filter, so this evaluates raw bounded alternatives, not a completed new policy. Cloned branches do not update the controller or alter the reference rollout.','',
        '| State group | Action | Contacts / states | Mean task-error change vs selected (mm-equivalent) | Worse-error states | Lost all pad contacts vs selected |','|---|---|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['action']} | {r['contacts']}/{r['states']} | {1000*r['mean_task_error_delta']:.4f} | {r['worse_than_selected']} | {r['loses_all_pads_vs_selected']} |")
    lines+=['',
        'Task error is sqrt(position_error^2 + (0.15 * tool_direction_error)^2), following the controller task scaling. Positive change means worse progress than the selected action. Worse-error counts use only a 1e-9 numerical tolerance; magnitudes, not these counts alone, determine practical relevance. Pad counts are the existing within-step contact union, not proof of a stable retained grasp.','',
        'This is outcome-selected, retrospective diagnosis with privileged simulator cloning. The 110 controls are dependent samples from only two successful six-joint trajectories; they do not establish broad retention. A branch avoiding one contact can fail on the next step. No branch is selected by an executable prediction rule here, and no full task is recovered. E78 remains a failed prediction gate.','',
        'Next decision must use both collision avoidance and successful-control interference. A promising action family would still need an explicit causal selection rule and complete-task validation against unchanged controls, including all casewise regressions and analytical comparisons. Do not substitute one-step avoided-contact counts for task success.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(table,indent=2))


if __name__=='__main__':run()
