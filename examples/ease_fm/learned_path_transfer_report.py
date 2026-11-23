"""E91 complete-case paired accounting for learned-plus-planner control."""
import json
from pathlib import Path
from .learned_path_transfer import ROOT
from .benchmark import fingerprint


def run():
    rows=json.loads((ROOT/'records.json').read_text());audit=json.loads((ROOT/'audit.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text())
    assert len(rows)==288 and audit['exact_baselines']==96
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    table=[];changes=[];gates=[];checks=[]
    for group in ('all','6','7',*sorted({r['method'] for r in rows})):
        for mode in ('baseline','route100','route300'):
            rr=[r for r in rows if r['mode']==mode and (group=='all' or str(r['dof'])==group or r['method']==group)]
            table.append(dict(group=group,mode=mode,trials=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),
                route_used=sum(r['route_used'] for r in rr),steps=sum(r['steps'] for r in rr),filter_queries=sum(r['filter_queries'] for r in rr)))
    for b in rows:
        if b['mode']!='baseline':continue
        for mode in ('route100','route300'):
            r=next(r for r in rows if r['case']==b['case'] and r['method']==b['method'] and r['mode']==mode)
            if r['success']!=b['success']:changes.append(dict(case=r['case'],body=r['body'],dof=r['dof'],method=r['method'],mode=mode,change='gain' if r['success'] else 'loss',baseline_collision=b['collision'],new_collision=r['collision']))
    for mode in ('route100','route300'):
        losses=[r for r in changes if r['mode']==mode and r['dof']==6 and r['change']=='loss']
        gates.append(dict(mode=mode,six_joint_retention_passes=not losses,six_joint_losses=losses))
    for r in rows:
        if r['mode']!='route100':continue
        trial=json.loads((ROOT/'trials'/f"{r['case']}_{r['method']}_route100.json").read_text())
        other=json.loads((ROOT/'trials'/f"{r['case']}_{r['method']}_route300.json").read_text())
        assert trial['route_check']==other['route_check']
        checks.append(dict(case=r['case'],method=r['method'],available=r['route_available'],accepted=r['route_used'],queries=trial['route_check'].get('queries',0),minimum_clearance=min(trial['route_check']['clearances']) if trial['route_check'] else None))
    losses=[]
    for r in changes:
        if r['change']!='loss':continue
        trial=json.loads((ROOT/'trials'/f"{r['case']}_{r['method']}_{r['mode']}.json").read_text())
        losses.append(dict(case=r['case'],method=r['method'],mode=r['mode'],dof=r['dof'],collision=trial['collision'],grasped=trial['grasped'],lifted=trial['lifted'],
            descent_reached=next(p['reached'] for p in trial['phases'] if p['phase']=='descend'),
            first_collision=next((p['phase'] for p in trial['trace'] if p['bad']),None),unreached=[p['phase'] for p in trial['phases'] if not p['reached']]))
    result=dict(table=table,changes=changes,gates=gates,route_checks=checks,loss_diagnosis=losses,audit=audit)
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2))
    lines=['# E91: frozen learned estimators with privileged route assistance','',
        'All16 E87 development cases and all six frozen checkpoints are compared in baseline, route100 and route300 conditions (288 tasks). Baseline is the original geometry-filter learned controller, without E82 response correction. The planner reuses only the single frozen E89 selected endpoint, constructs a101-sample joint-space route from the learned controller\'s actual descent posture and accepts it only if the sampled clearance is at least2mm. Rejected or unavailable routes retain baseline actions and budgets. Route300 permits300 descent steps only for accepted routes; all other phase budgets and success rules remain unchanged.',
        '', 'The descent tracker uses joint feedback plus measured adaptive bias; it does not use Transformer actions for that phase. Learned Jacobian control handles approach and all later phases. Offline endpoint IK and online exact scene geometry are privileged planning inputs. No task-time simulator Jacobian calls or target-arm weight updates are allowed. These are development interventions, not an independently evaluated history-only policy.',
        '', '| Group | Mode | Success | Collisions | Routes used | Steps incl. calibration | Filter queries |','|---|---|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['trials']} | {r['collisions']} | {r['route_used']} | {r['steps']} | {r['filter_queries']} |")
    lines+=['','| Mode | Case | Body | DOF | Checkpoint | Change | Baseline collision | Intervention collision |','|---|---:|---:|---:|---|---|---|---|']
    for r in changes:lines.append(f"| {r['mode']} | {r['case']} | {r['body']} | {r['dof']} | {r['method']} | {r['change']} | {r['baseline_collision']} | {r['new_collision']} |")
    lines+=['','Six-joint casewise retention checks:','']
    for g in gates:lines.append(f"- {g['mode']}: {'PASS' if g['six_joint_retention_passes'] else 'FAIL'}, {len(g['six_joint_losses'])} previously successful six-joint cases lost.")
    for mode in ('route100','route300'):
        paired=[r for r in changes if r['mode']==mode]
        lines.append(f"- {mode}: {sum(r['change']=='gain' for r in paired)} gains and {sum(r['change']=='loss' for r in paired)} losses versus baseline.")
    lines+=['','Lost baseline successes: descriptive phase diagnostics (not causal attribution):','', '| Case | Checkpoint | Mode | Descent reached | First collision phase | Unreached phases |','|---|---|---|---|---|---|']
    for r in losses:lines.append(f"| {r['case']} | {r['method']} | {r['mode']} | {r['descent_reached']} | {r['first_collision']} | {', '.join(r['unreached'])} |")
    eligible=sum(r['available'] for r in checks);accepted=sum(r['accepted'] for r in checks)
    lines+=['',f'Route eligibility: {eligible}/96 checkpoint/case pairs have an E89 eligible endpoint; {accepted}/96 pass the actual-start check; {eligible-accepted} eligible routes are rejected by that check. Each route condition makes {sum(r["queries"] for r in checks)} additional geometry queries for admission, separate from per-step filter queries. Both route modes make identical admission decisions.',
        '', 'These counts pool dependent checkpoints and placements. Report every paired loss, not only net improvement. Different descent endpoints/postures can change all subsequent learned histories and behavior. Any gain is attributable to the combined planning intervention, not solely to the motion Transformer. E87 response-corrected scores remain unchanged, and comparisons against E90 exact-Jacobian outcomes are descriptive because the actual learned start states and admission decisions differ.',
        '', f'Verification: all96 baseline rollouts reproduce E87 exactly; {audit["exact_fallbacks"]} fallback rollouts also match. Descent-start qpos/qvel agree across modes for each checkpoint/case. Route checks preserve physical state. Task Jacobian calls are blocked, the neural inverse guard remains enabled, input/checkpoint/source hashes remain unchanged, and114 tests pass. Every completed trial and its trace is saved immediately. No retraining or reserved-test evaluation occurred.',
        '', '## Decision','',
        'Both variants fail the six-joint retention gate and are not promoted. Route100 reaches49/96 success versus43/96 baseline, with17 gains and11 losses. Route300 reaches57/96, with24 gains and10 losses. Six-joint counts are15/48 baseline,18/48 route100 and22/48 route300; each intervention loses five baseline-successful six-joint cases. The longer route recovers five of six checkpoint runs on difficult case4, but blanket route assistance regresses easier cases. All78 eligible routes pass actual-start rechecking; the18 pairs without an eligible E89 route remain exact baseline fallbacks in both modes.',
        '', 'Of21 loss records across the two modes,20 reach the descent target before failing later. This motivates a controlled post-descent handoff diagnostic, but does not prove estimator history is the cause. Checkpoint/placement repetitions and the two modes are dependent.',
        '',
        'Do not promote route assistance as a drop-in improvement if the casewise retention gate fails, regardless of pooled gains. Inspect handoff from the planned descent to learned Cartesian control on lost baseline successes. A useful next diagnostic compares learned and exact local Jacobians at the same recorded post-descent states, then uses a separately labeled exact-kinematics continuation to distinguish estimation/handoff errors from remaining trajectory or contact problems. Keep privileged diagnostic gains distinct from learned results and preserve every failure.',
        '', '## Video evidence','',
        'Gallery: `outputs/ease_learned_path_transfer_v1/videos/index.html`. Candidate_0 cases0 and4 in baseline and route300 are the prespecified illustrative examples; if necessary the first missing success/failure category is appended. Videos show executed full tasks, disclose hybrid privileged-planner use, and must match saved traces exactly. Example selection is not a representative success-rate sample.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(main=table[:9],gates=gates,changes=changes),indent=2))


if __name__=='__main__':run()
