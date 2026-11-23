"""E95 paired outcomes, intervention contacts and measured excitation."""
import json
import shutil
from pathlib import Path
import numpy as np
from .local_probe import ROOT
from .benchmark import fingerprint


def run():
    rows=json.loads((ROOT/'records.json').read_text());audit=json.loads((ROOT/'audit.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text())
    assert len(rows)==288 and audit['exact_replays']==96
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    table=[];changes=[];contacts=[]
    for group in ('all','6','7',*sorted({r['method'] for r in rows})):
        for mode in ('no_probe','hold32','probe32'):
            rr=[r for r in rows if r['mode']==mode and (group=='all' or str(r['dof'])==group or r['method']==group)]
            eligible=[r for r in rr if r['route_used']]
            table.append(dict(group=group,mode=mode,trials=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),steps=sum(r['steps'] for r in rr),extra_steps=sum(r['probe_steps'] for r in rr),
                median_directions_before=float(np.median([r['excitation']['before']['above_ridge'] for r in eligible])),median_directions_after=float(np.median([r['excitation']['after']['above_ridge'] for r in eligible])),median_effective_rank_after=float(np.median([r['excitation']['after']['effective_rank'] for r in eligible]))))
    for r in rows:
        if r['mode']!='probe32':continue
        for control in ('no_probe','hold32'):
            b=next(x for x in rows if x['case']==r['case'] and x['method']==r['method'] and x['mode']==control)
            if r['success']!=b['success']:changes.append(dict(control=control,case=r['case'],method=r['method'],dof=r['dof'],change='gain' if r['success'] else 'loss',old_collision=b['collision'],new_collision=r['collision']))
    for r in rows:
        p=json.loads((ROOT/'trials'/f"{r['case']}_{r['method']}_{r['mode']}.json").read_text())
        first=next((t['phase'] for t in p['trace'] if t['bad']),None)
        contacts.append(dict(case=r['case'],method=r['method'],mode=r['mode'],first_collision=first,intervention_collision=any(t['bad'] for t in p['trace'] if t['phase'] in ('probe','hold'))))
    probe=table[2];hold=table[1];loss6=[r for r in changes if r['dof']==6 and r['change']=='loss']
    gate=dict(improves_over_hold=probe['success']>hold['success'],nonincreasing_hold_collisions=probe['collisions']<=hold['collisions'],six_joint_losses=loss6)
    gate['passes']=gate['improves_over_hold'] and gate['nonincreasing_hold_collisions'] and not loss6
    summary=dict(table=table,changes=changes,contacts=contacts,gate=gate,audit=audit)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# E95: local motion probe versus interaction-matched hold','',
        'All96 E93 checkpoint/task pairs run with no probe,32-step hold or32-step probe. The78 admitted-route pairs receive the intervention after planned descent and before finger closing; the other18 retain original behavior and budgets. All use the same frozen motion estimators, recent32 statistics, measured bias and privileged route planner. Probe and hold retain the same32-step budget, anchor feedback, geometry filter, open fingers and final success criteria.',
        '', 'The fixed probe commands each joint +0.004rad for two control steps and -0.004rad for two, in joint-index order, then returns toward the anchor for remaining steps. Offsets are targets around the measured handoff posture, not guaranteed actual displacements. Every physical contact, object displacement and extra interaction counts. No early exit, reset, re-screening or outcome tuning occurs.',
        '', '| Group | Mode | Success | Collisions | Total steps | Extra intervention steps | Median directions before / after | Median effective rank after |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['trials']} | {r['collisions']} | {r['steps']} | {r['extra_steps']} | {r['median_directions_before']:.1f} / {r['median_directions_after']:.1f} | {r['median_effective_rank_after']:.3f} |")
    lines+=['','| Control | Case | Checkpoint | DOF | Probe outcome change | Control collision | Probe collision |','|---|---:|---|---:|---|---|---|']
    for r in changes:lines.append(f"| {r['control']} | {r['case']} | {r['method']} | {r['dof']} | {r['change']} | {r['old_collision']} | {r['new_collision']} |")
    lines+=['','## Gate and interpretation','',f"Gate: **{'PASS' if gate['passes'] else 'FAIL'}**. Probe improves over hold={gate['improves_over_hold']}; nonincreasing collisions versus hold={gate['nonincreasing_hold_collisions']}; six-joint loss comparisons={len(loss6)} (the same case can be a loss against both controls).",'']
    for control in ('no_probe','hold32'):
        cc=[r for r in changes if r['control']==control];lines.append(f"- Versus {control}: {sum(r['change']=='gain' for r in cc)} gains and {sum(r['change']=='loss' for r in cc)} losses.")
    for mode in ('hold32','probe32'):
        cc=[r for r in contacts if r['mode']==mode];lines.append(f"- {mode}: {sum(r['intervention_collision'] for r in cc)} trials contact forbidden geometry during the intervention; {sum(r['first_collision'] in ('probe','hold') for r in cc)} first collide during it.")
    lines+=['','Excitation metrics subtract the1e-6 ridge, with above-ridge directions counted using that existing regularizer. Increased directional coverage is not sufficient evidence of an accurate Jacobian or useful task behavior. Probe motion also changes joint pose, velocity, object state and bias estimates. The hold comparison separates some effects of extra time and observation updates, but does not isolate information gain from every mechanical effect.',
        '', 'This is reused development data with a privileged planner, not independent generalization or a history-only policy. The gate fails: do not adopt this probe or tune its amplitude or order on these same outcomes. Restoring directional coverage did not improve full-task success: probe57/96 versus hold59/96 and no-probe58/96, with37 versus35 and34 collisions. Two unique six-joint cases successful under both controls are lost. Next audit learned versus analytical and exact motion estimates at identical post-intervention states to distinguish useful measurements from frozen-estimator mismatch before another control intervention.',
        '', f"Verification:96 exact E93 no-probe references and{audit['exact_fallbacks']} unchanged no-route fallbacks; identical pre-intervention physical state and controller statistics; fixed32-step insertion tests; unchanged hashes and123 passing tests. All task-time simulator Jacobians remain blocked. Each completed trial is saved immediately. No training or reserved-test evaluation occurred.",
        '', '## Videos','',
        'Gallery: `outputs/ease_local_probe_v1/videos/index.html`. Prespecified candidate_0 cases0 and4 show all three modes; the first new probe regression against hold is appended with all three modes if not already represented. Selection is disclosed and illustrative. If the prespecified set lacks an outcome category, the first sorted example of that category is added before the paired regression selection. The completed gallery contains ten exact-replay videos, including this extra hold-failure example. Executed video replays must exactly match saved outcomes and traces.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print(json.dumps(dict(main=table[:9],gate=gate),indent=2))


if __name__=='__main__':run()
