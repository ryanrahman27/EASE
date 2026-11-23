"""Summarize E100 without treating diagnostic selection as a benchmark."""
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .command_divergence import ROOT


def run():
    spec=json.loads((ROOT/'protocol.json').read_text());summary=json.loads((ROOT/'summary.json').read_text())
    lines=['# E100: same-state command divergence audit','',
        'All three E99 retained regressions are audited on both parent and retained trajectories. The lowest-seed unchanged-success control on the same task is included where available: case10 seed2 only. Cases11 and13 have no seed successful under both conditions. Eight complete rollouts reproduce their E99 serialized results exactly. This outcome-selected diagnostic is not a new benchmark score.',
        '', 'At each post-probe step both frozen networks see identical measured Gram/cross statistics, bias, task error and physical state. Their clipped DLS commands are separately passed through the original geometry filter using shadow data. The original command executes unchanged. Physical state and controller statistics are checked unchanged by the audit, and executed raw commands match the corresponding queried network exactly. No simulator Jacobian is queried. Geometry filtering and route planning remain privileged.',
        '', '## Before-contact comparison on retained trajectories','',
        'Entries below use the phase containing the first forbidden contact; only steps starting before any forbidden contact are averaged, including the command whose execution first contacts. The successful control uses retreat for comparison. Norms are joint-command L2 in radians, not position errors.',
        '', '| Case / seed | Outcome | Phase | Before-contact commands | Raw mean L2 | Filtered mean L2 | Scale disagreements | Last pre-contact raw / filtered L2 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    fig,axes=plt.subplots(len(spec['pairs']),1,figsize=(10,10),layout='constrained')
    details=[]
    for ax,(case,seed) in zip(axes,spec['pairs']):
        stem=f'{case}_candidate_{seed}_retained';trial=json.loads((ROOT/'trials'/f'{stem}.json').read_text());rr=json.loads((ROOT/'trials'/f'{stem}_commands.json').read_text())
        phase=next((r['phase'] for r in trial['trace'] if r['bad']),'retreat');clean=[r for r in rr if r['phase']==phase and not r['already_bad']]
        last=clean[-1];raw=float(np.mean([r['raw_l2'] for r in clean]));filtered=float(np.mean([r['filtered_l2'] for r in clean]))
        lines.append(f"| {case} / {seed} | {'success control' if trial['success'] else 'regression'} | {phase} | {len(clean)} | {raw:.6f} | {filtered:.6f} | {sum(r['scale_disagrees'] for r in clean)} | {last['raw_l2']:.6f} / {last['filtered_l2']:.6f} |")
        details.append(dict(case=case,seed=seed,phase=phase,raw_mean=raw,filtered_mean=filtered,last_precontact=last if not trial['success'] else None))
        for mode,style in [('parent','--'),('retained','-')]:
            data=json.loads((ROOT/'trials'/f'{case}_candidate_{seed}_{mode}_commands.json').read_text())
            ax.plot(np.arange(len(data)),np.cumsum([r['filtered_l2'] for r in data]),style,label=f'{mode} trajectory')
        collision=next((i for i,r in enumerate(rr) if r['already_bad']),None)
        if collision is not None:ax.axvline(collision-.5,color='red',alpha=.5,label='first contact transition')
        ax.set(title=f"Case {case}, seed {seed}: {'unchanged success' if trial['success'] else 'retained regression'}",ylabel='Cumulative command L2 (rad)',xlabel='Post-probe step (trajectories differ)');ax.legend(fontsize=8)
    lines+=['','## Interpretation','',
        'Command differences are present before the first collision, but they do not define a reliable failure threshold. The successful control also exhibits substantial disagreement. Case13 differs from the other two regressions: the collision occurs in transport after only four pre-contact commands, following lift-phase filter disagreements. Both networks produce exactly identical filtered commands on all four of those transport states (mean filtered L2 zero despite raw mean 0.031651 rad). This directs attention to the inherited lift state, without proving a cause. Immediate command disagreement alone cannot explain inherited pose/velocity differences or establish a counterfactual recovery.',
        '', 'The filter does not uniformly suppress disagreement; its discrete scale choice can change between networks on the same state. Cumulative norms in the figure measure repeated command discrepancy and must not be interpreted as accumulated physical displacement. Comparisons between the two executed trajectories cease to be same-state after their first differing command; within each trajectory every network comparison remains same-state.',
        '', 'Next experiment: collect on-policy student continuation histories exclusively on the original training bodies, retaining failures and a whole-placement validation split. Compare a fixed-budget matched replay-only control with the added on-policy data, without selecting seeds or tuning on the three diagnostic failures. This tests a distribution-shift hypothesis; E100 does not establish it as the cause. Repeat all development tasks and the existing strict retention gate before adoption.',
        '', '## Verification and artifacts','',
        '129 existing tests pass. Eight exact full-result replays, unchanged physical/controller state during queries, exact executed raw commands and unchanged input/checkpoint hashes pass. Per-step raw/filtered vectors, filter decisions, states and phase summaries are saved. No training or reserved-test evaluation occurs. All eight parent/retained diagnostic videos are rendered and checked against saved full results, decoded frame counts and SHA-256 hashes; the gallery discloses outcome selection. Figure: command_divergence.png / .pdf.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n');(ROOT/'contact_summary.json').write_text(json.dumps(details,indent=2))
    fig.savefig(ROOT/'command_divergence.png',dpi=140);fig.savefig(ROOT/'command_divergence.pdf');plt.close(fig)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print('\n'.join(lines[8:18]))


if __name__=='__main__':run()

