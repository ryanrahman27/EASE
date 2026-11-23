"""E93 paired retention and original-baseline comparison."""
import json
import shutil
from pathlib import Path
from .local_history import ROOT,PRIOR,previous
from .benchmark import fingerprint


def run():
    rows=json.loads((ROOT/'records.json').read_text());audit=json.loads((ROOT/'audit.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text())
    assert len(rows)==192 and audit['exact_replays']==96
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    table=[];changes=[];original_changes=[]
    for group in ('all','6','7',*sorted({r['method'] for r in rows})):
        for mode in ('long_history','window32'):
            rr=[r for r in rows if r['mode']==mode and (group=='all' or str(r['dof'])==group or r['method']==group)]
            table.append(dict(group=group,mode=mode,trials=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),steps=sum(r['steps'] for r in rr),local_applied=sum(r['local_history_applied'] for r in rr)))
    for r in rows:
        if r['mode']!='window32':continue
        b=next(x for x in rows if x['case']==r['case'] and x['method']==r['method'] and x['mode']=='long_history')
        if r['success']!=b['success']:changes.append(dict(case=r['case'],method=r['method'],dof=r['dof'],change='gain' if r['success'] else 'loss',old_collision=b['collision'],new_collision=r['collision']))
        original=json.loads((PRIOR/'trials'/f"{r['case']}_{r['method']}_baseline.json").read_text())
        if r['success']!=original['success']:original_changes.append(dict(case=r['case'],method=r['method'],dof=r['dof'],change='gain' if r['success'] else 'loss'))
    before=table[0];after=table[1];losses6=[r for r in changes if r['dof']==6 and r['change']=='loss']
    gate=dict(improved_success=after['success']>before['success'],nonincreasing_collisions=after['collisions']<=before['collisions'],six_joint_losses=losses6)
    gate['passes']=gate['improved_success'] and gate['nonincreasing_collisions'] and not losses6
    e91=json.loads((previous.ROOT/'summary.json').read_text());oldlosses=[r for r in e91['changes'] if r['mode']=='route300' and r['change']=='loss']
    recovered=sum(next(r['success'] for r in rows if r['mode']=='window32' and r['case']==b['case'] and r['method']==b['method']) for b in oldlosses)
    summary=dict(table=table,changes=changes,original_baseline_changes=original_changes,gate=gate,e91_regressions_recovered=recovered,audit=audit)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# E93: fixed recent-history estimation after planned descent','',
        'All96 E91 checkpoint/task pairs receive two full rollouts: exact long-history route300 replay and window32. After an accepted planned descent, window32 reconstructs gram/cross from the32 most recent completed transitions using the original initial ridge, forgetting factor and diagonal floor; it repeats this reconstruction after each subsequent measured transition. Adaptive measured bias remains unchanged. No new probe, policy weight update, window sweep or simulator-Jacobian policy input is introduced. Cases without an admitted route retain the original history.',
        '', 'The privileged E89/E91 route planner and300-step descent allowance remain present in both conditions. This evaluates a history intervention in that hybrid controller on reused development cases, not a history-only robot policy or independent transfer test.',
        '', '| Group | Mode | Success | Collisions | Control steps including calibration | Window applied |','|---|---|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['trials']} | {r['collisions']} | {r['steps']} | {r['local_applied']} |")
    lines+=['','| Case | Checkpoint | DOF | Change vs long history | Old collision | New collision |','|---|---|---:|---|---|---|']
    for r in changes:lines.append(f"| {r['case']} | {r['method']} | {r['dof']} | {r['change']} | {r['old_collision']} | {r['new_collision']} |")
    lines+=['','## Decision and interpretation','',
        'Do not adopt window32. It produces58/96 successes versus57/96, but increases collisions from33 to34. Six-joint performance improves22/48 to28/48 while seven-joint performance falls35/48 to30/48. The11 gains and10 losses include one lost previously successful six-joint hybrid case. Only five of the ten E91 regressions recover, unlike the privileged exact-Jacobian continuation in E92. These opposing DOF aggregates do not justify selecting a history rule by joint count after observing the results.',
        '',
        f"The prespecified gate {'PASSES' if gate['passes'] else 'FAILS'}: success improvement={gate['improved_success']}, nonincreasing collisions={gate['nonincreasing_collisions']}, lost successful six-joint hybrid cases={len(losses6)}. There are{sum(r['change']=='gain' for r in changes)} gains and{sum(r['change']=='loss' for r in changes)} losses versus long history. Window32 recovers{recovered}/10 E91 regressions diagnosed in E92.",
        '', f"Relative to the original E87 learned geometry-filter baseline, window32 has{sum(r['change']=='gain' for r in original_changes)} gains and{sum(r['change']=='loss' for r in original_changes)} losses, including{sum(r['change']=='loss' and r['dof']==6 for r in original_changes)} six-joint losses. This separate comparison retains the known planner and longer-budget confounds.",
        '', 'A shorter history changes both locality and the amount/directional coverage of excitation. Near a stationary grasp,32 recent transitions may contain very little informative motion; frozen weights may also encounter statistics different from training. Outcomes cannot by themselves isolate stale history, insufficient excitation or distribution shift. Do not interpret aggregate success as estimator calibration or infer a failure threshold from this run.',
        '', 'If this fixed-window intervention fails retention, retain the previous controller and investigate measured excitation/conditioning at handoff before considering another history rule. Any replacement should preserve informative motion rather than blindly discarding older transitions. Predeclare subsequent checks and retain the same successful controls; do not search window sizes on these outcomes and claim independent validation.',
        '', f"Verification:96 exact E91 reference replays and{audit['exact_fallbacks']} exact unavailable-route fallbacks; identical pre-intervention handoff states and statistics; no task simulator Jacobian access; frozen checkpoint/input/source hashes;117 passing tests including exact last32-statistic replay, bias preservation and copied measurement history. Individual trials were saved immediately. No reserved-test evaluation occurred.",
        '', '## Saved videos','',
        'Gallery: `outputs/ease_local_history_v1/videos/index.html`. Prespecified candidate_0 cases0 and4 show both modes; append the first missing success/failure category if needed. A disclosed post-result addition shows the first newly introduced regression (case2 parent_1) in both modes, for six videos total. Executed video replays match the saved results exactly. These illustrative examples do not estimate success rate.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print(json.dumps(dict(main=table[:6],gate=gate,recovered=recovered,gains=sum(r['change']=='gain' for r in changes),losses=sum(r['change']=='loss' for r in changes)),indent=2))


if __name__=='__main__':run()
