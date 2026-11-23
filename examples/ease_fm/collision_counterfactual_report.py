"""E74 outcome-conditioned collision chronology and one-step evidence."""
import json
from collections import Counter
from pathlib import Path
import shutil
from .collision_counterfactual import ROOT
from .benchmark import fingerprint


def run():
    rows=json.loads((ROOT/'records.json').read_text());audit=json.loads((ROOT/'audit.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text())
    assert len(rows)==10 and audit['exact_full_replays']==10 and audit['branches']==50
    for p,h in {**spec['checkpoints'],**audit['input_hashes']}.items():assert fingerprint(p)==h
    for name in ('collision_counterfactual.py','geometry_filter.py','arm_pick_place.py','pick_place_control.py','action_objective_audit.py'):
        assert fingerprint(Path(__file__).with_name(name))==fingerprint(ROOT/'source'/name)
    names=list(rows[0]['branches']);avoids={name:sum(not r['branches'][name]['collision'] for r in rows) for name in names}
    dense_clear=sum(r['min_dense_clearance']>=.002-1e-9 for r in rows)
    penetrating=sum(r['branches']['selected']['min_actual_path_clearance']<0 for r in rows)
    phases=dict(Counter(r['phase'] for r in rows))
    out=dict(events=len(rows),learned_events=sum(r['method']!='analytical' for r in rows),phases=phases,
        filter_accepted=sum(r['check']['feasible'] for r in rows),dense_2mm_clear=dense_clear,actual_path_penetrations=penetrating,
        immediate_collision_avoided=avoids,analytical_full_task_success_on_same_case=sum(r['analytical_full_task_success'] for r in rows))
    (ROOT/'summary.json').write_text(json.dumps(out,indent=2))
    lines=['# E74: first-collision counterfactual audit','',
        'All ten unsuccessful E73 geometry-filter trajectories are replayed exactly: nine learned-controller failures and one analytical failure. Their first forbidden contacts are finger/table contacts in seven descend and three lower phases. The paired E73 unfiltered trajectories fail on the same cases and first-contact control indices. This is outcome-conditioned diagnosis, not a new success-rate evaluation.','',
        '| Counterfactual at the pre-contact state | Immediate contact avoided /10 |',
        '|---|---:|']
    for name,n in avoids.items():lines.append(f'| {name} | {n} |')
    lines+=['','| Case | Estimator | Phase | 21-sample minimum (mm) | Actual substep minimum (mm) | Analytical action avoids | Zero-velocity avoids |',
        '|---|---|---|---:|---:|---|---|']
    for r in rows:lines.append(f"| {r['case']} | {r['method']} | {r['phase']} | {1000*r['min_dense_clearance']:.3f} | {1000*r['branches']['selected']['min_actual_path_clearance']:.3f} | {not r['branches']['analytical_same_history']['collision']} | {not r['branches']['selected_zero_incoming_velocity']['collision']} |")
    lines+=['','### Interpretation','',
        'All ten filtered actions were accepted by the original geometric test. Nine remain at the 2 mm clearance cap even when the commanded joint/finger interpolation is sampled at 21 points, including its start. Reconstructed actual dynamic substep configurations penetrate forbidden geometry in all ten. Thus simply adding interpolation samples does not resolve these nine misses: the commanded straight joint path is not the executed dynamic trajectory. One event already has about 0.042 mm penetration at the start of the diagnosed step; the original contact flag first records it during that step. This contact-timing distinction is retained.','',
        'Half motion and bias-only avoid zero immediate collisions. Changing only the estimator used to compute the action—from the same measured Gram/cross/bias and task error—avoids five. This is a controlled local action comparison and implicates the proposed motion direction, but it does not establish full-task recovery or uniquely identify neural-estimation error as the cause. The analytical controller\'s independent whole-task rollout succeeds on eight of the ten corresponding event cases, with repeated case identities; it is a different trajectory.','',
        'Resetting incoming arm/finger velocities avoids two immediate contacts, so incoming velocity matters in those states. This privileged reset also recomputes constraints and is not a realizable braking action. Its limited benefit does not exclude acceleration, unequal joint responses, tracking lag or contact dynamics in the other failures. The analytical-action and zero-velocity successes overlap; they are not additive recoveries.','',
        '### Decision','',
        'Retain the original controller and keep the static geometry filter unpromoted. Do not perform another geometry-margin or interpolation-density sweep. The next useful comparison should account for the predicted executed joint response and current velocity, while retaining the same-history analytical action as a comparator. Any simulator-dynamics lookahead must be labeled a privileged model-based diagnostic and separated from claims about history-only adaptation. Before claiming a useful fix, require complete-task evaluation and preservation of successful six-joint cases.','',
        '### Verification and limitations','',
        'Ten full trajectories, filter logs and calibration probes replay exactly; all ten selected-action branches reproduce the original contact-step tool position. Fifty branches run for one 40 ms control step each, with original state/velocity/time restored after each. Checkpoint, input and source hashes are unchanged. All 104 EASE-FM tests pass, including branch state/flag restoration. No weights, phase budgets, collision masks or task success criteria changed.','',
        'All chosen states precede an observed failure, so branch counts cannot estimate deployment success. There is no successful-state intervention control or multi-step recovery evaluation here. Actual-path clearance reconstruction uses the current object pose rather than updating it; the first forbidden pair in every event is finger/table, which does not involve the object. Discrete geometry samples do not guarantee continuous clearance. No training, native tuning or reserved final-test evaluation occurred.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axs=plt.subplots(1,2,figsize=(12,4));x=np.arange(len(rows))
    axs[0].plot(x,[1000*r['min_dense_clearance'] for r in rows],'o-',label='Commanded path: 21 samples')
    axs[0].plot(x,[1000*r['branches']['selected']['min_actual_path_clearance'] for r in rows],'s-',label='Executed dynamic substeps')
    axs[0].axhline(0,color='black',lw=.8);axs[0].set(ylabel='Minimum forbidden clearance (mm)',xlabel='Failure event index',xticks=x);axs[0].legend(fontsize=8)
    labels=['Selected','Half motion','Bias only','Analytical\naction','Zero incoming\nvelocity']
    bars=axs[1].bar(labels,[avoids[n] for n in names],color='#4c78a8');axs[1].bar_label(bars);axs[1].set(ylabel='Immediate collisions avoided /10',ylim=(0,11));axs[1].tick_params(axis='x',labelsize=8)
    fig.suptitle('E74: failure-conditioned one-step diagnostics, not task recoveries');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(out,indent=2))


def document():
    lead='**E74 completed:** All ten E73 geometry-filter failures first hit the table with a finger. Nine commanded joint paths remain 2 mm clear under 21-point sampling, yet all ten executed dynamic paths penetrate forbidden geometry. From identical pre-contact states, analytical same-history actions avoid 5/10 immediate contacts, velocity reset 2/10, and half-motion/bias-only 0/10. These are one-step failure-conditioned diagnostics, not recovered tasks. Ten full replays and 50 branches are audited; 104 tests pass. See Section 4.64.'
    folder=Path(__file__).parent;p=folder/'TECHNICAL_REPORT.md';text=p.read_text();assert '## 4.64 E74:' not in text
    text=text.replace('Revision 59 · 28 October 2026','Revision 60 · 28 October 2026',1)
    text=text.replace('## 1. Research question and claim criteria\n','## 1. Research question and claim criteria\n\n'+lead+'\n',1)
    line=next(x for x in text.splitlines(True) if x.startswith('| E73 |'))
    text=text.replace(line,line+'| E74 | Exact pre-contact counterfactual audit | Complete: 10 failed trajectories, 50 one-step branches; dynamic path mismatch | `ease_collision_counterfactual_v1` |\n',1)
    section=(ROOT/'report.md').read_text().replace('# E74: first-collision counterfactual audit','## 4.64 E74: first-collision counterfactual audit',1)
    text=text.replace('## 5. Limitations and next decisions',section+'\n\n## 5. Limitations and next decisions',1)
    text=text.replace('## 5. Limitations and next decisions\n\n','## 5. Limitations and next decisions\n\nE74 shows a gap between commanded kinematic paths and executed dynamic trajectories in the remaining E73 finger/table failures. Keep the original controller; do not retune static margins or waiting times. Next compare velocity/response-aware action prediction with a same-history analytical-action comparator, explicitly labeling any simulator-based lookahead as privileged. One-step contact avoidance must not be presented as full-task recovery.\n\n',1)
    text=text.replace('## Revision history\n','## Revision history\n\n- Revision 60, 28 October 2026: Completed E74 on every E73 geometry-filter failure: ten exact full replays and fifty matched one-step branches. Audited dense kinematic versus actual dynamic-path clearances, same-history analytical action and incoming-velocity interventions, unchanged hashes and branch-state restoration. Inspected figure; 104 tests pass. No training, task-recovery claim, native tuning or final-test evaluation.\n',1)
    p.write_text(text)
    for name in ('README.md','MIXED_DESIGN.md'):
        p=folder/name;first,rest=p.read_text().split('\n',1);p.write_text(first+'\n\n'+lead+'\n'+rest)
    shutil.copy2(Path(__file__),ROOT/'source'/Path(__file__).name)


if __name__=='__main__':
    import sys
    document() if '--document' in sys.argv else run()
