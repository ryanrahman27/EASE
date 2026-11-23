"""E86 complete-task paired retention and resource accounting."""
import json
from pathlib import Path
from .direction_expanded import ROOT
from .benchmark import fingerprint


def run():
    rows=json.loads((ROOT/'records.json').read_text());audit=json.loads((ROOT/'audit.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text())
    cases=json.loads((ROOT/'cases.json').read_text());screen=json.loads((ROOT/'screening.json').read_text())
    assert len(rows)==21*len(cases) and audit['exact_baselines']==7*len(cases)
    for p,h in {**audit['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    table=[]
    for group in ('learned','learned_6','learned_7','analytical',*sorted({r['method'] for r in rows if r['method']!='analytical'}),*[f'body_{b}' for b in sorted({r['body'] for r in rows})]):
        for mode in ('baseline','response','command'):
            rr=[r for r in rows if r['mode']==mode and ((r['method']!='analytical' and (group=='learned' or r['dof']==int(group[-1]))) if group.startswith('learned') else (r['method']!='analytical' and r['body']==int(group[5:]) if group.startswith('body_') else r['method']==group))]
            table.append(dict(group=group,mode=mode,trials=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),
                steps=sum(r['steps'] for r in rr),changed_actions=sum(r['changed_actions'] for r in rr),correction_queries=sum(r['correction_queries'] for r in rr),filter_queries=sum(r['filter_queries'] for r in rr)))
    pairs=[];gates=[]
    for mode in ('response','command'):
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
    response=next(r for r in table if r['group']=='learned' and r['mode']=='response')
    command=next(r for r in table if r['group']=='learned' and r['mode']=='command')
    attribution=response['success']>command['success'] and response['collisions']<=command['collisions']
    paired=[]
    for r in rows:
        if r['mode']!='response' or r['method']=='analytical':continue
        command_row=next(x for x in rows if x['case']==r['case'] and x['method']==r['method'] and x['mode']=='command')
        if r['success']!=command_row['success']:
            paired.append(dict(case=r['case'],body=r['body'],method=r['method'],dof=r['dof'],change='gain' if r['success'] else 'loss'))
    body_effects=[]
    for body in sorted({r['body'] for r in rows}):
        rr=[r for r in paired if r['body']==body]
        t=next(t for t in cases if t['body']==body)
        body_effects.append(dict(body=body,family=t['arm']['family'],replicate=t['replicate'],seed=t['arm']['seed'],dof=len(t['arm']['lengths']),
            gains=sum(r['change']=='gain' for r in rr),losses=sum(r['change']=='loss' for r in rr)))
    result=dict(table=table,changes=pairs,gates=gates,attribution_passes=attribution,response_vs_command=paired,body_effects=body_effects,screening=screen,exact_baselines=audit['exact_baselines'])
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# E86: complete-task retention with frozen direction correction','',
      f"Twelve new parameterizations (same axis families/topology) and two fixed placements per body were saved before screening and policy execution. {len(cases)}/24 placements passed nine-endpoint IK screening; no rejected candidate was replaced. Evaluate all accepted cases with seven frozen estimators in three modes. Original phase budgets and success criteria remain unchanged.",'',
      'Baseline uses the original geometry filter. Response mode uses frozen E82 measured-response direction correction. Command mode uses the identical correction implementation, candidate set, geometry, sampling, output clipping and learned-Jacobian progress bound, replacing predicted displacement with commanded displacement. Response histories still update identically, but are unused in command scoring. Any changed trigger frequency or query cost follows from that sole predictor difference. No dynamic simulator lookahead chooses actions.','',
      '| Group | Mode | Success | Collisions | Control steps incl. calibration | Changed actions | Additional correction queries |','|---|---|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['trials']} | {r['collisions']} | {r['steps']} | {r['changed_actions']} | {r['correction_queries']} |")
    lines+=['','The advancement gate requires improved pooled learned success, no increased pooled collisions, and preservation of every baseline-successful six-joint case.','']
    for r in gates:lines.append(f"- {r['mode']}: {'PASS' if r['passes'] else 'FAIL'}; improved success={r['improves_success']}, nonincreasing collisions={r['nonincreasing_collisions']}, six-joint casewise losses={len(r['six_joint_losses'])}.")
    lines+=['','| Mode | Case | Estimator | DOF | Outcome change | Baseline collision | New collision |','|---|---:|---|---:|---|---|---|']
    for r in pairs:lines.append(f"| {r['mode']} | {r['case']} | {r['method']} | {r['dof']} | {r['change']} | {r['baseline_collision']} | {r['new_collision']} |")
    lines+=['',
      'These are prospective development results on new parameterizations derived from six existing body templates. They share existing axis families and are pooled across dependent checkpoint/placement repetitions, not independent samples or one policy success rate. All casewise losses remain visible, even if a checkpoint retains its aggregate score. A passed gate would justify separate-placement validation, not a new-embodiment or conference-ready generalization claim.','',
      'Both correction modes remain available throughout the task. Query costs include all 21-point candidate paths and are separate from original geometry-filter queries; no runtime advantage is claimed.','',
      'Verification: full baseline result and filter-log equality against a separately instantiated original GeometryArm, identical calibration probes, one response update per measured control step, unchanged input/checkpoint/source hashes. Original artifacts and model weights are preserved. No reserved final-test evaluation or retraining occurred.']
    lines+=['',f"Measured-response attribution gate: **{'PASS' if attribution else 'FAIL'}**. Requires higher learned success than the command ablation without more collisions. Passing only the baseline-improvement gate does not establish response-model benefit.",'','Screening records:','']
    for r in screen:lines.append(f"- Candidate {r['candidate']}, body {r['body']}, {r['dof']} joints: {'accepted' if r['accepted'] else 'rejected'}, {r['endpoints']}/9 endpoints; failed phase {r['failed_phase']}.")
    lines+=['','Direct response-versus-command paired effects, by accepted body (checkpoint/placement outcomes are dependent within each body):','',
        '| Body | Family | Parameter seed | DOF | Response-only successes | Command-only successes |','|---|---|---:|---:|---:|---:|']
    for r in body_effects:lines.append(f"| {r['body']} | {r['family']} | {r['seed']} | {r['dof']} | {r['gains']} | {r['losses']} |")
    lines+=['','Bodies with no accepted placement are absent from rollout comparisons but retained in screening. A pooled success difference is not evidence of consistent effects across independently sampled bodies; inspect concentration and checkpoint identity before strengthening the claim.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(main=table[:12],changes=pairs,gates=gates),indent=2))



if __name__=="__main__":run()
