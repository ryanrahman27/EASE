"""E83 complete-task paired retention and resource accounting."""
import json
from pathlib import Path
from .direction_fulltask import ROOT
from .benchmark import fingerprint


def run():
    rows=json.loads((ROOT/'records.json').read_text());audit=json.loads((ROOT/'audit.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text())
    assert len(rows)==336 and audit['exact_baselines']==112
    for p,h in {**audit['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    table=[]
    for group in ('learned','learned_6','learned_7','analytical',*sorted({r['method'] for r in rows if r['method']!='analytical'})):
        for mode in ('baseline','sustained','pulse'):
            rr=[r for r in rows if r['mode']==mode and ((r['method']!='analytical' and (group=='learned' or r['dof']==int(group[-1]))) if group.startswith('learned') else r['method']==group)]
            table.append(dict(group=group,mode=mode,trials=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),
                steps=sum(r['steps'] for r in rr),changed_actions=sum(r['changed_actions'] for r in rr),correction_queries=sum(r['correction_queries'] for r in rr),filter_queries=sum(r['filter_queries'] for r in rr)))
    pairs=[];gates=[]
    for mode in ('sustained','pulse'):
        changes=[]
        for base in (r for r in rows if r['mode']=='baseline'):
            new=next(r for r in rows if r['mode']==mode and r['case']==base['case'] and r['method']==base['method'])
            if base['success']!=new['success']:
                changes.append(dict(mode=mode,case=base['case'],method=base['method'],dof=base['dof'],change='gain' if new['success'] else 'loss',baseline_collision=base['collision'],new_collision=new['collision']))
        pairs.extend(changes)
        base=next(r for r in table if r['group']=='learned' and r['mode']=='baseline');new=next(r for r in table if r['group']=='learned' and r['mode']==mode)
        losses=[r for r in changes if r['method']!='analytical' and r['dof']==6 and r['change']=='loss']
        gates.append(dict(mode=mode,improves_success=new['success']>base['success'],nonincreasing_collisions=new['collisions']<=base['collisions'],six_joint_losses=losses,
            passes=new['success']>base['success'] and new['collisions']<=base['collisions'] and not losses))
    result=dict(table=table,changes=pairs,gates=gates,exact_baselines=112)
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# E83: complete-task retention with frozen direction correction','',
      'All 16 original placements and seven estimators are evaluated in three modes (336 full pick-and-place tasks). Sustained correction is the exploratory primary candidate; one five-step pulse at the first online warning is secondary. All phase transitions, budgets, collision rules and final success criteria are unchanged. Calibration is identical across modes; all 112 baseline traces, probes and filter logs reproduce exactly.','',
      'The correction implementation and numerical settings are frozen from E82. Selection uses causal measured-response predictions, whole-arm geometry and the learned Jacobian task-progress constraint; analytical comparison rows use their corresponding analytical estimator. Geometry and exact current simulator state remain privileged information, and object motion is not predicted. No simulation lookahead selects actions.','',
      '| Group | Mode | Success | Collisions | Control steps incl. calibration | Changed actions | Additional correction queries |','|---|---|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['trials']} | {r['collisions']} | {r['steps']} | {r['changed_actions']} | {r['correction_queries']} |")
    lines+=['','The advancement gate requires improved pooled learned success, no increased pooled collisions, and preservation of every baseline-successful six-joint case.','']
    for r in gates:lines.append(f"- {r['mode']}: {'PASS' if r['passes'] else 'FAIL'}; improved success={r['improves_success']}, nonincreasing collisions={r['nonincreasing_collisions']}, six-joint casewise losses={len(r['six_joint_losses'])}.")
    lines+=['','| Mode | Case | Estimator | DOF | Outcome change | Baseline collision | New collision |','|---|---:|---|---:|---|---|---|']
    for r in pairs:lines.append(f"| {r['mode']} | {r['case']} | {r['method']} | {r['dof']} | {r['change']} | {r['baseline_collision']} | {r['new_collision']} |")
    lines+=['',
      'These are pooled results across checkpoints and repeated historical development placements, not independent samples or one policy success rate. All casewise losses remain visible, even if a checkpoint retains its aggregate score. A passed gate would justify separate-placement validation, not a new-embodiment or conference-ready generalization claim.','',
      'The pulse triggers once per trajectory and continues for five total steps even if a phase transition occurs; it never retriggers. Sustained correction remains available throughout. Query costs include all 21-point candidate paths and are separate from original geometry-filter queries; no runtime advantage is claimed.','',
      'Verification: full baseline trace and filter-log equality, identical calibration probes, one response update per measured control step, unchanged input/checkpoint/source hashes. Original artifacts and model weights are preserved. No reserved final-test evaluation or retraining occurred.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(main=table[:12],changes=pairs,gates=gates),indent=2))


def document():
    import shutil
    result=json.loads((ROOT/'summary.json').read_text())
    def cell(group,mode):return next(r for r in result['table'] if r['group']==group and r['mode']==mode)
    b=cell('learned','baseline');s=cell('learned','sustained');p=cell('learned','pulse');six=cell('learned_6','sustained')
    gate=next(r for r in result['gates'] if r['mode']=='sustained')
    gains=[r for r in result['changes'] if r['mode']=='sustained' and r['method']!='analytical' and r['change']=='gain']
    losses=[r for r in result['changes'] if r['mode']=='sustained' and r['method']!='analytical' and r['change']=='loss']
    lead=(f"**E83 completed:** Full-task sustained correction yields {s['success']}/{s['trials']} learned successes versus baseline {b['success']}/{b['trials']}, with {s['collisions']} versus {b['collisions']} collisions. Pulse yields {p['success']}/{p['trials']}. Sustained has {len(gains)} casewise gains and {len(losses)} losses; six-joint success is {six['success']}/{six['trials']}. The primary retention gate {'passes' if gate['passes'] else 'fails'}. All 112 baselines reproduce exactly; 107 tests pass. These are historical development tasks, not independent generalization. See Section 4.73.")
    decision=('Freeze the sustained controller and replicate it on the separate E73 placement suite, retaining every previously successful six-joint case and reporting all regressions and query costs. Do not retune on this development gain.' if gate['passes'] else 'Keep the original controller as the reference. Diagnose complete-task regressions and unrecovered failures before changing this method; do not retune thresholds to turn this gate into a pass.')
    rp=ROOT/'report.md';rp.write_text(rp.read_text()+'\n### Decision\n\n'+lead+'\n\n'+decision+'\n\nAll 107 EASE-FM tests pass. Local avoidance and development task gains do not establish superiority on new embodiments.\n')
    folder=Path(__file__).parent;path=folder/'TECHNICAL_REPORT.md';text=path.read_text();assert '## 4.73 E83:' not in text
    text=text.replace('Revision 68','Revision 69',1)
    text=text.replace('## 1. Research question and claim criteria\n','## 1. Research question and claim criteria\n\n'+lead+'\n',1)
    line=next(x for x in text.splitlines(True) if x.startswith('| E82 |'))
    text=text.replace(line,line+'| E83 | Frozen direction correction, full-task retention | Complete: 336 tasks, 112 exact baseline replays | `ease_direction_fulltask_v1` |\n',1)
    section=rp.read_text().replace('# E83:', '## 4.73 E83:',1)
    text=text.replace('## 5. Limitations and next decisions',section+'\n## 5. Limitations and next decisions',1)
    text=text.replace('## 5. Limitations and next decisions\n\n','## 5. Limitations and next decisions\n\n'+lead+' '+decision+'\n\n',1)
    text=text.replace('## Revision history\n','## Revision history\n\n- Revision 69, 1 November 2026: E83 evaluates 336 complete tasks with frozen correction settings, original sequencing and success rules. Reports paired gains/losses, six-joint casewise retention, analytical comparisons and query costs. All 112 baseline traces and filter logs reproduce exactly; 107 tests pass. No independent generalization or reserved-test claim.\n',1);path.write_text(text)
    for name in ('README.md','MIXED_DESIGN.md'):
        path=folder/name;first,rest=path.read_text().split('\n',1);path.write_text(first+'\n\n'+lead+'\n'+rest)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)


if __name__=='__main__':
    import sys
    document() if '--document' in sys.argv else run()
