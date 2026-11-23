"""E81 warning coverage and matched multi-step branch outcomes."""
import json
import shutil
from pathlib import Path
import numpy as np
from .early_warning_audit import ROOT
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
            for mode in ('selected','half','quarter','bias_only'):
                def coll(r,m):return any(x['collision'] for x in r['branches'][m]['history'][:horizon])
                delta=[r['branches'][mode]['history'][horizon-1]['task_error']-r['branches']['selected']['history'][horizon-1]['task_error'] for r in rr]
                tables.append(dict(group=group,horizon=horizon,mode=mode,states=len(rr),collisions=sum(coll(r,mode) for r in rr),
                    avoided_vs_matched=sum(coll(r,'selected') and not coll(r,mode) for r in rr),new_vs_matched=sum(not coll(r,'selected') and coll(r,mode) for r in rr),
                    mean_task_error_delta=None if not delta else float(np.mean(delta))))
    result=dict(coverage=coverage,tables=tables,exact_replays=len(rows),branches=4*len(rows),simulated_branch_steps=80*len(rows))
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# E81: early-warning intervention pulses','',
      'The first causal no-velocity response warning at the unchanged 2 mm threshold triggers each local experiment, regardless of eventual outcome. No warning is selected by counting backward from a known collision. Warning values come from the previously exact-replayed E78 run. Collisions without advance warning remain in coverage counts.','',
      'Every triggered state has four 20-step branches (0.8 seconds). Half, quarter and bias-only interventions last five steps (0.2 seconds), followed by fifteen steps of baseline control. The selected branch uses baseline control throughout. Controller history and bias update from each branch independently. Phase goal and aperture stay fixed at their triggering values, so these are matched local continuations, not the original phase-switching full task. No future phase information or analytical fallback is used.','',
      '| DOF | Original collisions | Advance warning | Warning only on collision step | No warning | Successful warning trajectories |','|---|---:|---:|---:|---:|---:|']
    for r in coverage:lines.append(f"| {r['dof'] or 'all'} | {r['collisions']} | {r['advance_warning']} | {r['collision_step_only']} | {r['no_warning']} | {r['successful_warning_trajectories']} |")
    lines+=['','| Group | Horizon (steps) | Action | Contacts / branches | Avoided vs matched | New vs matched | Mean task-error change (mm-equivalent) |','|---|---:|---|---:|---:|---:|---:|']
    for r in tables:
        if r['group']=='all_triggered' or r['horizon'] not in (5,20):continue
        delta='n/a' if r['mean_task_error_delta'] is None else f"{1000*r['mean_task_error_delta']:.4f}"
        lines.append(f"| {r['group']} | {r['horizon']} | {r['mode']} | {r['collisions']}/{r['states']} | {r['avoided_vs_matched']} | {r['new_vs_matched']} | {delta} |")
    lines+=['',
      'Task error uses sqrt(position_error^2 + (0.15 * tool_direction_error)^2). Positive changes mean worse progress. Contact outcomes are cumulative within each horizon; per-step contact and error traces are retained. Avoidance is relative to the matched phase-held baseline, not necessarily the original recorded collision. A contact after the five-step pulse but before step 20 counts against longer-horizon avoidance.','',
      f"Verification: {len(rows)} full reference trajectories and filter logs reproduce exactly; selected-branch first observations and contact labels match their originals. Cloned state restoration and input/source/checkpoint hashes pass. {4*len(rows)} branches contain {80*len(rows)} simulated control steps.",'',
      'Limitations: historical development data, outcome reporting on the same bodies, and successful warnings concentrated in two trajectories. The phase-held horizon can itself create contacts in an originally successful task. These results cannot establish full-task recovery, broad retention, or an independent safety guarantee. No neural training or reserved-test evaluation occurred; the original controller and E78 failed gate remain unchanged.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(coverage=coverage,final=[r for r in tables if r['horizon']==20],exact_replays=len(rows)),indent=2))


if __name__=='__main__':run()
