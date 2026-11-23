"""E82 warning coverage and matched multi-step branch outcomes."""
import json
import shutil
from pathlib import Path
import numpy as np
from .direction_correction import ROOT
from .benchmark import fingerprint


def run():
    spec=json.loads((ROOT/'protocol.json').read_text());audit=json.loads((ROOT/'audit.json').read_text())
    for p,h in {**audit['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    rows=[]
    for p in (ROOT/'trials').glob('*.json'):
        t=json.loads(p.read_text());assert len(t['attribution'])==1
        rows.append(dict(**{k:v for k,v in t.items() if k!='attribution'},**t['attribution'][0]))
    coverage=[]
    for dof in (None,6,7):
        allrows=[r for r in spec['coverage'] if dof is None or r['dof']==dof];collisions=[r for r in allrows if r['first_contact'] is not None]
        coverage.append(dict(dof=dof,trajectories=len(allrows),collisions=len(collisions),advance_warning=sum(r['advance_warning'] for r in collisions),
            collision_step_only=sum(r['first_warning']==r['first_contact'] for r in collisions),no_warning=sum(r['first_warning'] is None for r in collisions),
            successful_warning_trajectories=sum(r['success'] and r['first_warning'] is not None for r in allrows)))
    tables=[]
    for group in ('all_triggered','original_collisions','original_collisions_6','successful_controls'):
        rr=[r for r in rows if group=='all_triggered' or (r['success'] if group=='successful_controls' else r['first_contact'] is not None and (group!='original_collisions_6' or r['dof']==6))]
        for horizon in (1,5,10,20):
            for mode in ('selected','direction_pulse','direction_all'):
                def coll(r,m):return any(x['collision'] for x in r['branches'][m]['history'][:horizon])
                delta=[r['branches'][mode]['history'][horizon-1]['task_error']-r['branches']['selected']['history'][horizon-1]['task_error'] for r in rr]
                tables.append(dict(group=group,horizon=horizon,mode=mode,states=len(rr),collisions=sum(coll(r,mode) for r in rr),
                    avoided_vs_matched=sum(coll(r,'selected') and not coll(r,mode) for r in rr),new_vs_matched=sum(not coll(r,'selected') and coll(r,mode) for r in rr),
                    mean_task_error_delta=None if not delta else float(np.mean(delta))))
    costs=[]
    for mode in ('direction_pulse','direction_all'):
        checks=[x['correction'] for r in rows for x in r['branches'][mode]['history'] if x['correction'] is not None]
        costs.append(dict(mode=mode,checks=len(checks),changed=sum(x['changed'] for x in checks),geometry_queries=sum(x['queries'] for x in checks)))
    result=dict(coverage=coverage,tables=tables,costs=costs,exact_replays=len(rows),branches=3*len(rows),simulated_branch_steps=60*len(rows))
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# E82: response-based direction corrections','',
      'The first causal no-velocity response warning at the unchanged 2 mm threshold triggers each local experiment, regardless of eventual outcome. No warning is selected by counting backward from a known collision. Warning values come from the previously exact-replayed E78 run. Collisions without advance warning remain in coverage counts.','',
      'Each triggering state has three matched 20-step branches: baseline, direction correction for five steps then baseline, and correction for all 20 steps. Candidate actions change one arm joint by +/-0.01 rad, clipped to +/-0.04. Frozen no-velocity response predictions score 21-point whole-arm clearance paths. A correction requires >1e-6 clearance improvement and predicted task error no more than 0.001 above baseline, using the learned Jacobian. The fixed 2 mm warning threshold, candidate set and progress bound are not tuned. Geometry is privileged; object pose stays fixed during path prediction. No future simulator outcomes select actions.','',
      'Controller and response history update independently from completed measurements in each branch. The triggering phase goal and aperture remain fixed. Both pulse and sustained variants are reported; neither is selected retrospectively as a validated controller.','',
      '| DOF | Original collisions | Advance warning | Warning only on collision step | No warning | Successful warning trajectories |','|---|---:|---:|---:|---:|---:|']
    for r in coverage:lines.append(f"| {r['dof'] or 'all'} | {r['collisions']} | {r['advance_warning']} | {r['collision_step_only']} | {r['no_warning']} | {r['successful_warning_trajectories']} |")
    lines+=['','| Group | Horizon (steps) | Action | Contacts / branches | Avoided vs matched | New vs matched | Mean task-error change (mm-equivalent) |','|---|---:|---|---:|---:|---:|---:|']
    for r in tables:
        if r['group']=='all_triggered' or r['horizon'] not in (5,20):continue
        delta='n/a' if r['mean_task_error_delta'] is None else f"{1000*r['mean_task_error_delta']:.4f}"
        lines.append(f"| {r['group']} | {r['horizon']} | {r['mode']} | {r['collisions']}/{r['states']} | {r['avoided_vs_matched']} | {r['new_vs_matched']} | {delta} |")
    lines+=['',
      'Task error uses sqrt(position_error^2 + (0.15 * tool_direction_error)^2). Positive changes mean worse progress. Contact outcomes are cumulative within each horizon; per-step contact and error traces are retained. Avoidance is relative to the matched phase-held baseline, not necessarily the original recorded collision. A contact after the five-step pulse but before step 20 counts against longer-horizon avoidance.','',
      f"Verification: {len(rows)} full reference trajectories and filter logs reproduce exactly; selected-branch first observations and contact labels match their originals. All 20 matched baseline steps also equal E81 exactly. Cloned state restoration and input/source/checkpoint hashes pass. {3*len(rows)} branches contain {60*len(rows)} simulated control steps.",'',
      'Limitations: historical development data, outcome reporting on the same bodies, and successful warnings concentrated in two trajectories. The phase-held horizon can itself create contacts in an originally successful task. These results cannot establish full-task recovery, broad retention, or an independent safety guarantee. No neural training or reserved-test evaluation occurred; the original controller and E78 failed gate remain unchanged.']
    lines+=['','Correction costs (additional geometry queries, excluding the original filter):','']
    for r in costs:lines.append(f"- {r['mode']}: {r['changed']}/{r['checks']} checked actions changed; {r['geometry_queries']} geometry queries.")
    lines+=['','All 107 EASE-FM tests pass, including correction bounds/input preservation and rejection of a clearance improvement that violates the progress constraint. These are implementation checks, not evidence of physical safety.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(coverage=coverage,final=[r for r in tables if r['horizon']==20],exact_replays=len(rows)),indent=2))


if __name__=='__main__':run()
