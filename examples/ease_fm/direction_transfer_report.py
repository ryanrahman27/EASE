"""E84 complete-task paired retention and resource accounting."""
import json
from pathlib import Path
from .direction_transfer import ROOT
from .benchmark import fingerprint


def run():
    rows=json.loads((ROOT/'records.json').read_text());audit=json.loads((ROOT/'audit.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text())
    assert len(rows)==238 and audit['exact_baselines']==119
    for p,h in {**audit['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    table=[]
    for group in ('learned','learned_6','learned_7','analytical',*sorted({r['method'] for r in rows if r['method']!='analytical'})):
        for mode in ('baseline','sustained'):
            rr=[r for r in rows if r['mode']==mode and ((r['method']!='analytical' and (group=='learned' or r['dof']==int(group[-1]))) if group.startswith('learned') else r['method']==group)]
            table.append(dict(group=group,mode=mode,trials=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),
                steps=sum(r['steps'] for r in rr),changed_actions=sum(r['changed_actions'] for r in rr),correction_queries=sum(r['correction_queries'] for r in rr),filter_queries=sum(r['filter_queries'] for r in rr)))
    pairs=[];gates=[]
    for mode in ('sustained',):
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
    result=dict(table=table,changes=pairs,gates=gates,exact_baselines=119)
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# E84: complete-task retention with frozen direction correction','',
      'All 17 separate E73 placements and seven estimators are evaluated in two modes (238 full pick-and-place tasks). Sustained correction is the exploratory primary candidate; no pulse comparison or settings retuning is performed. All phase transitions, budgets, collision rules and final success criteria are unchanged. Calibration is identical across modes; all 119 baseline traces, probes and filter logs reproduce exactly.','',
      'The correction implementation and numerical settings are frozen from E82. Selection uses causal measured-response predictions, whole-arm geometry and the learned Jacobian task-progress constraint; analytical comparison rows use their corresponding analytical estimator. Geometry and exact current simulator state remain privileged information, and object motion is not predicted. No simulation lookahead selects actions.','',
      '| Group | Mode | Success | Collisions | Control steps incl. calibration | Changed actions | Additional correction queries |','|---|---|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['trials']} | {r['collisions']} | {r['steps']} | {r['changed_actions']} | {r['correction_queries']} |")
    lines+=['','The advancement gate requires improved pooled learned success, no increased pooled collisions, and preservation of every baseline-successful six-joint case.','']
    for r in gates:lines.append(f"- {r['mode']}: {'PASS' if r['passes'] else 'FAIL'}; improved success={r['improves_success']}, nonincreasing collisions={r['nonincreasing_collisions']}, six-joint casewise losses={len(r['six_joint_losses'])}.")
    lines+=['','| Mode | Case | Estimator | DOF | Outcome change | Baseline collision | New collision |','|---|---:|---|---:|---|---|---|']
    for r in pairs:lines.append(f"| {r['mode']} | {r['case']} | {r['method']} | {r['dof']} | {r['change']} | {r['baseline_collision']} | {r['new_collision']} |")
    lines+=['',
      'These are pooled results across checkpoints and repeated historical development placements, not independent samples or one policy success rate. All casewise losses remain visible, even if a checkpoint retains its aggregate score. A passed gate would justify separate-placement validation, not a new-embodiment or conference-ready generalization claim.','',
      'Sustained correction remains available throughout. These placements differ from E83 but reuse the same robot bodies and were previously used in development; this is not independent final-test or new-embodiment validation. Query costs include all 21-point candidate paths and are separate from original geometry-filter queries; no runtime advantage is claimed.','',
      'Verification: full baseline trace and filter-log equality, identical calibration probes, one response update per measured control step, unchanged input/checkpoint/source hashes. Original artifacts and model weights are preserved. No reserved final-test evaluation or retraining occurred.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(main=table[:12],changes=pairs,gates=gates),indent=2))



if __name__=="__main__":run()
