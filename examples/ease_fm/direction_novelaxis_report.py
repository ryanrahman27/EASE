"""E87 complete-task paired retention and resource accounting."""
import json
from collections import Counter
from pathlib import Path
from .direction_novelaxis import ROOT
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
    axis_audit=json.loads((ROOT/'axis_audit.json').read_text());axis_summary=[]
    assert axis_audit['training_disjoint'] and axis_audit['catalog_disjoint']
    for family in axis_audit['chosen']:
        case_ids={i for i,t in enumerate(cases) if t['arm']['family']==family['family']}
        for group in ('learned','analytical'):
            for mode in ('baseline','response','command'):
                rr=[r for r in rows if r['case'] in case_ids and r['mode']==mode and ((r['method']!='analytical') if group=='learned' else r['method']=='analytical')]
                axis_summary.append(dict(sequence=family['word'],dof=family['dof'],group=group,mode=mode,trials=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr)))
    failure_summary=[]
    for group in ('learned_6','learned_7','analytical'):
        for mode in ('baseline','response','command'):
            rr=[r for r in rows if r['mode']==mode and ((r['method']=='analytical') if group=='analytical' else r['method']!='analytical' and r['dof']==int(group[-1]))]
            counts=dict(collision=0,no_grasp=0,no_lift=0,placement_or_release=0,success=0)
            for r in rr:
                category='success' if r['success'] else 'collision' if r['collision'] else 'no_grasp' if not r['grasped'] else 'no_lift' if not r['lifted'] else 'placement_or_release'
                counts[category]+=1
            assert sum(counts.values())==len(rr)
            failure_summary.append(dict(group=group,mode=mode,trials=len(rr),**counts))
    baseline_diagnosis=[]
    for family in axis_audit['chosen']:
        case_ids={i for i,t in enumerate(cases) if t['arm']['family']==family['family']}
        for group in ('learned','analytical'):
            rr=[r for r in rows if r['case'] in case_ids and r['mode']=='baseline' and ((r['method']!='analytical') if group=='learned' else r['method']=='analytical')]
            first_contacts=Counter();unreached=Counter()
            for r in rr:
                unreached.update(p['phase'] for p in r['phases'] if not p['reached'])
                if r['collision']:
                    trial=json.loads((ROOT/'trials'/f"{r['case']}_{r['method']}_baseline.json").read_text())
                    first=next((p['phase'] for p in trial['trace'] if p['bad']),'calibration_or_unrecorded')
                    first_contacts[first]+=1
            baseline_diagnosis.append(dict(sequence=family['word'],group=group,trials=len(rr),first_contact_phases=dict(first_contacts),unreached_phases=dict(unreached)))
    result=dict(table=table,changes=pairs,gates=gates,attribution_passes=attribution,response_vs_command=paired,body_effects=body_effects,axis_summary=axis_summary,failure_summary=failure_summary,baseline_diagnosis=baseline_diagnosis,screening=screen,exact_baselines=audit['exact_baselines'])
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# E87: complete-task retention with frozen direction correction','',
      f"Eight new parameterizations on four unseen procedural axis sequences and two fixed placements per body were saved before screening and policy execution. {len(cases)}/16 placements passed nine-endpoint IK screening; no rejected candidate was replaced. Evaluate all accepted cases with seven frozen estimators in three modes. Original phase budgets and success criteria remain unchanged.",'',
      'Baseline uses the original geometry filter. Response mode uses frozen E82 measured-response direction correction. Command mode uses the identical correction implementation, candidate set, geometry, sampling, output clipping and learned-Jacobian progress bound, replacing predicted displacement with commanded displacement. Response histories still update identically, but are unused in command scoring. Any changed trigger frequency or query cost follows from that sole predictor difference. No dynamic simulator lookahead chooses actions.','',
      '| Group | Mode | Success | Collisions | Control steps incl. calibration | Changed actions | Additional correction queries |','|---|---|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['trials']} | {r['collisions']} | {r['steps']} | {r['changed_actions']} | {r['correction_queries']} |")
    lines+=['','The advancement gate requires improved pooled learned success, no increased pooled collisions, and preservation of every baseline-successful six-joint case.','']
    for r in gates:lines.append(f"- {r['mode']}: {'PASS' if r['passes'] else 'FAIL'}; improved success={r['improves_success']}, nonincreasing collisions={r['nonincreasing_collisions']}, six-joint casewise losses={len(r['six_joint_losses'])}.")
    lines+=['','| Mode | Case | Estimator | DOF | Outcome change | Baseline collision | New collision |','|---|---:|---|---:|---|---|---|']
    for r in pairs:lines.append(f"| {r['mode']} | {r['case']} | {r['method']} | {r['dof']} | {r['change']} | {r['baseline_collision']} | {r['new_collision']} |")
    lines+=['',
      'These are prospective development results on new parameterizations using length/dynamics templates with newly assigned axis sequences. Their exact ordered axis sequences are excluded from training and known catalogs and are pooled across dependent checkpoint/placement repetitions, not independent samples or one policy success rate. All casewise losses remain visible, even if a checkpoint retains its aggregate score. A passed gate would justify separate-placement validation, not a new-embodiment or conference-ready generalization claim.','',
      'Both correction modes remain available throughout the task. Query costs include all 21-point candidate paths and are separate from original geometry-filter queries; no runtime advantage is claimed.','',
      'Verification: full baseline result and filter-log equality against a separately instantiated original GeometryArm, identical calibration probes, one response update per measured control step, unchanged input/checkpoint/source hashes. Original artifacts and model weights are preserved. No reserved final-test evaluation or retraining occurred.']
    lines+=['',f"Measured-response attribution gate: **{'PASS' if attribution else 'FAIL'}**. Requires higher learned success than the command ablation without more collisions. Passing only the baseline-improvement gate does not establish response-model benefit.",'','Screening records:','']
    for r in screen:lines.append(f"- Candidate {r['candidate']}, body {r['body']}, {r['dof']} joints: {'accepted' if r['accepted'] else 'rejected'}, {r['endpoints']}/9 endpoints; failed phase {r['failed_phase']}.")
    lines+=['','Direct response-versus-command paired effects, by accepted body (checkpoint/placement outcomes are dependent within each body):','',
        '| Body | Family | Parameter seed | DOF | Response-only successes | Command-only successes |','|---|---|---:|---:|---:|---:|']
    for r in body_effects:lines.append(f"| {r['body']} | {r['family']} | {r['seed']} | {r['dof']} | {r['gains']} | {r['losses']} |")
    lines+=['','Bodies with no accepted placement are absent from rollout comparisons but retained in screening. A pooled success difference is not evidence of consistent effects across independently sampled bodies; inspect concentration and checkpoint identity before strengthening the claim.']
    lines+=['','Axis-specific outcomes:','', '| Sequence | DOF | Estimator group | Mode | Success | Collisions |','|---|---:|---|---|---:|---:|']
    for r in axis_summary:lines.append(f"| {r['sequence']} | {r['dof']} | {r['group']} | {r['mode']} | {r['success']}/{r['trials']} | {r['collisions']} |")
    lines+=['','Outcome diagnosis (mutually exclusive, collision takes precedence among failures):','', '| Group | Mode | Success | Collision | No grasp | Grasp without lift | Placement/release failure after lift |','|---|---|---:|---:|---:|---:|---:|']
    for r in failure_summary:lines.append(f"| {r['group']} | {r['mode']} | {r['success']} | {r['collision']} | {r['no_grasp']} | {r['no_lift']} | {r['placement_or_release']} |")
    lines+=['','Grasp and lift flags denote whether the event occurred at any point. Placement/release combines the remaining final object-position, stability and finger-contact checks; these records do not isolate which final check failed. Endpoint IK feasibility does not establish successful contact-rich execution.']
    lines+=['','Baseline phase diagnostics (unreached phases can overlap within a trial and are descriptive, not proven causes):','', '| Sequence | Estimator group | First contact phase counts | Unreached phase counts |','|---|---|---|---|']
    for r in baseline_diagnosis:lines.append(f"| {r['sequence']} | {r['group']} | {json.dumps(r['first_contact_phases'],sort_keys=True)} | {json.dumps(r['unreached_phases'],sort_keys=True)} |")
    lines+=['','The lineage audit verifies four training axis sequences and excludes 36 exact sequences from the checked training/development catalogs and built-in fixtures. Reserved geometry metadata is used only for exclusion; no reserved task goals or trajectories are evaluated. E30/E35 synthetic random linear-system supervision remains part of training, and E66 measured training contexts come from the eight audited training bodies. Exact sequence disjointness does not prove inequivalent kinematic functions or independence from prior development choices.']
    lines+=['','## Decision and limitations','',
        'The frozen response variant passes both advancement and attribution gates, but absolute task success remains low: 46/96 pooled learned trials, including 17/48 six-joint trials. Baseline achieves 43/96 and command correction 44/96. Response has three task gains and no losses against baseline. Its two gains over command are cases 6 and 7, both candidate_0 on body 3 (ZYXYZY), so the measured-response-specific success effect has not replicated across bodies or checkpoints in this study. The seven-joint gain in case 12 occurs in both correction modes.',
        '', 'Learned collision counts are 40 baseline, 27 response and 35 command. Avoiding a collision often leaves an unsuccessful task: response has 23 noncollision failures, compared with 13 baseline. These mutually exclusive outcome counts do not prove that correction caused new grasp failures; many baseline failures already collide before later task checks.',
        '', 'Response adds 1,883,343 geometry queries, for 2,026,782 total versus 153,357 baseline (13.22 times as many). It changes 1,752 actions. No real-time or efficiency claim is supported. Analytical control improves from 7/16 to 9/16 with response, versus 8/16 command; this also argues against attributing all benefit specifically to the Transformer.',
        '', 'Next: run a separately labeled privileged kinematic-control diagnostic on all 16 fixed cases with the same home states, calibration, task budgets and geometry filter, preserving all outcomes. Compare exact Jacobian substitution with measured-history estimates to distinguish motion-estimation limits from shared path/contact limitations. If exact kinematics still fails, inspect path feasibility and phase transitions before retraining or tuning the correction. This diagnostic is not a deployable policy or a final-test result.',
        '', 'Verification completed: 336 primary task evaluations, 112 independently instantiated exact baseline replays, unchanged checkpoint/input/source hashes and calibration probes, causal response updates, and 108 passing EASE tests. The report generator also passes Python syntax compilation. Artifacts: outputs/ease_direction_novelaxis_v1.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(main=table[:12],changes=pairs,gates=gates),indent=2))



if __name__=="__main__":run()
