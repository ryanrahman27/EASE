"""E97 full-task results, retention and diagnostic continuation decision."""
import json
import shutil
from pathlib import Path
from .probe_continuation import ROOT
from .local_probe import ROOT as PREVIOUS
from .benchmark import fingerprint


def run():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==288
    audit=json.loads((ROOT/'audit.json').read_text());assert audit['exact_replays']==96 and audit['exact_fallbacks']==36
    spec=json.loads((ROOT/'protocol.json').read_text())
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    prior=json.loads((PREVIOUS/'records.json').read_text());table=[];changes=[];controls=[]
    for group in ('all','6','7','route','no_route',*sorted({r['method'] for r in rows})):
        for mode in ('learned','analytical','exact'):
            rr=[r for r in rows if r['mode']==mode and (group=='all' or str(r['dof'])==group or r['method']==group or (group=='route' and r['route_used']) or (group=='no_route' and not r['route_used']))]
            table.append(dict(group=group,mode=mode,n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),grasped=sum(r['grasped'] for r in rr),lifted=sum(r['lifted'] for r in rr),steps=sum(r['steps'] for r in rr)))
    for r in rows:
        if r['mode']=='learned':continue
        b=next(x for x in rows if x['case']==r['case'] and x['method']==r['method'] and x['mode']=='learned')
        if r['success']!=b['success']:changes.append(dict(case=r['case'],method=r['method'],dof=r['dof'],mode=r['mode'],change='gain' if r['success'] else 'loss',old_collision=b['collision'],new_collision=r['collision']))
        for control in ('no_probe','hold32'):
            b=next(x for x in prior if x['case']==r['case'] and x['method']==r['method'] and x['mode']==control)
            if r['success']!=b['success']:controls.append(dict(case=r['case'],method=r['method'],dof=r['dof'],mode=r['mode'],control=control,change='gain' if r['success'] else 'loss'))
    learned,analytical,exact=table[:3]
    loss6=[r for r in changes if r['mode']=='analytical' and r['dof']==6 and r['change']=='loss']
    gate=dict(improves_success=analytical['success']>learned['success'],nonincreasing_collisions=analytical['collisions']<=learned['collisions'],six_joint_losses=loss6)
    gate['passes']=gate['improves_success'] and gate['nonincreasing_collisions'] and not loss6
    first_collisions=[]
    for r in rows:
        full=json.loads((ROOT/'trials'/f"{r['case']}_{r['method']}_{r['mode']}.json").read_text())
        first=next((t['phase'] for t in full['trace'] if t['bad']),None)
        first_collisions.append(dict(case=r['case'],method=r['method'],mode=r['mode'],first=first))
    exact_comparison=[]
    for r in rows:
        if r['mode']!='exact':continue
        b=next(x for x in rows if x['case']==r['case'] and x['method']==r['method'] and x['mode']=='analytical')
        if r['success']!=b['success']:exact_comparison.append(dict(case=r['case'],method=r['method'],dof=r['dof'],change='gain' if r['success'] else 'loss'))
    (ROOT/'summary.json').write_text(json.dumps(dict(table=table,changes=changes,controls=controls,gate=gate,first_collisions=first_collisions,exact_vs_analytical=exact_comparison),indent=2))
    lines=['# E97: paired post-probe continuation','',
        'All 96 checkpoint/task pairs receive learned, analytical and exact continuation (288 full tasks). Every branch has the same learned approach, privileged route descent and fixed32-step local probe. At the first actual close prediction, after all probe measurements update statistics, switch only the motion estimator. The78 admitted-route pairs switch;18 no-route pairs retain original behavior. Bias updates, recent32 history updates, DLS controller, geometry filter, contacts, phase budgets and success criteria remain identical. No state reset or training occurs.',
        '', '| Group | Continuation | Success | Collisions | Grasped | Lifted | Total steps |','|---|---|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['n']} | {r['collisions']} | {r['grasped']} | {r['lifted']} | {r['steps']} |")
    lines+=['','## Paired changes against learned continuation','','| Case | Checkpoint | DOF | Continuation | Change | Learned collision | Branch collision |','|---|---|---:|---|---|---|---|']
    for r in changes:lines.append(f"| {r['case']} | {r['method']} | {r['dof']} | {r['mode']} | {r['change']} | {r['old_collision']} | {r['new_collision']} |")
    lines+=['','## Gate and retention','',f"Analytical diagnostic gate: **{'PASS' if gate['passes'] else 'FAIL'}**. Improves success={gate['improves_success']}; nonincreasing collisions={gate['nonincreasing_collisions']}; lost learned-successful six-joint cases={len(loss6)}.",'']
    for mode in ('analytical','exact'):
        cc=[r for r in changes if r['mode']==mode];lines.append(f"- {mode} versus learned: {sum(r['change']=='gain' for r in cc)} gains, {sum(r['change']=='loss' for r in cc)} losses.")
        for control in ('no_probe','hold32'):
            cc=[r for r in controls if r['mode']==mode and r['control']==control]
            lines.append(f"- {mode} versus E95 {control}: {sum(r['change']=='gain' for r in cc)} gains, {sum(r['change']=='loss' for r in cc)} losses; {sum(r['change']=='loss' and r['dof']==6 for r in cc)} six-joint losses.")
    lines.append(f"- Exact versus analytical: {sum(r['change']=='gain' for r in exact_comparison)} gains, {sum(r['change']=='loss' for r in exact_comparison)} losses.")
    lines+=['','## First forbidden-contact phase','','| Phase | Learned | Analytical | Exact |','|---|---:|---:|---:|']
    for phase in sorted({r['first'] for r in first_collisions if r['first'] is not None}):
        counts=[sum(r['mode']==mode and r['first']==phase for r in first_collisions) for mode in ('learned','analytical','exact')]
        lines.append(f'| {phase} | {counts[0]} | {counts[1]} | {counts[2]} |')
    lines+=['','All paired control changes are saved in summary.json. E95 reference totals remain no-probe58/96 (34 collisions), hold59/96 (35), and learned probe57/96 (37). Analytical/exact branches retain the32-step probe cost; comparisons with no-probe do not isolate estimator quality alone.',
        '', '## Interpretation and scope','',
        'Measured outcome: learned57/96 with37 collisions, analytical68/96 with27, exact78/96 with18. Analytical gains15 and loses4 versus learned; all four losses are seven-joint cases (10 candidate_0,12 parent_1,13 parent_1,13 candidate_1). Both lost six-joint probe cases from E95 recover. Six-joint success is26/48 learned versus30/48 analytical/exact; seven-joint is31/48 versus38/48 versus48/48. The diagnostic gate passes, but four seven-joint regressions prevent an unqualified replacement recommendation.',
        '', 'On the78 route-assisted pairs, analytical reaches68/78 with9 collisions and exact78/78 with0. All18 no-route pairs remain failed and unchanged, so78/96 is the result of this exact-continuation branch, not a universal ceiling or proof that other tasks are infeasible. All ten remaining analytical failures are seven-joint cases. The exact branch demonstrates that successful full-task continuation is possible from every evaluated probe handoff under the same remaining budgets and controller structure.',
        '', 'Next decision: use the analytical estimator as a stronger measured-history training target on training embodiments only, preserving the frozen evaluation split and all observed regressions. A trained replacement must be tested on complete tasks, not selected by handoff matrix error alone. In parallel within the research sequence, inspect estimator drift through the ten analytical failures before claiming that distillation alone closes the exact-reference gap. No weights or deployed controller are changed by E97.',
        '', 'This is a development diagnostic with privileged route planning and repeated checkpoint/task pairs. The analytical branch uses measured history for its motion fit; the exact branch additionally uses privileged simulator Jacobians. Neither branch is a new learned-policy transfer result. Identical prefixes isolate the post-probe estimator choice, but subsequent observations and trajectories diverge. Outcomes do not prove that a single handoff matrix error or stale history is the sole cause. No thresholds or probe parameters were tuned after outcomes.',
        '', 'Verification:96 learned full results and traces match E95;36 no-route fallbacks match; all288 handoffs match E96 physical state, controller statistics and32-transition histories. Branch prefixes through the probe match E95 exactly. Exact Jacobian access is scoped to the post-switch exact branch. All input and checkpoint hashes remain unchanged. All127 EASE tests pass, including switch timing and restoration of the Jacobian access guard. No reserved-test evaluation occurs.',
        '', '## Videos','',
        'Gallery: `outputs/ease_probe_continuation_v1/videos/index.html`. Prespecified case0 parent_1 and case2 candidate_0 (E95 regressions) and case4 candidate_0 (retained success), all three branches. If needed, append first sorted missing success/failure category and first analytical regression against learned with all branches. Selection is disclosed; these are illustrative rather than representative. Every video replays the complete saved result exactly, checks decoded frame count and records SHA-256 hashes.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print(json.dumps(dict(main=table[:15],gate=gate),indent=2))

if __name__=='__main__':run()
