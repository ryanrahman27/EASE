"""E92 report: outcome-selected same-state continuation diagnostic."""
import json
import statistics
import shutil
from pathlib import Path
from .learned_handoff import ROOT
from .benchmark import fingerprint


def run():
    rows=json.loads((ROOT/'records.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text());audit=json.loads((ROOT/'audit.json').read_text())
    assert len(rows)==32 and audit['exact_replays']==16
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    table=[]
    for group in ('loss','retained_success'):
        for mode in ('learned','exact_after_descent'):
            rr=[r for r in rows if r['group']==group and r['mode']==mode]
            table.append(dict(group=group,mode=mode,trials=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),steps=sum(r['steps'] for r in rr),
                median_relative_error=statistics.median(r['relative_jacobian_error'] for r in rr),min_relative_error=min(r['relative_jacobian_error'] for r in rr),max_relative_error=max(r['relative_jacobian_error'] for r in rr)))
    pairs=[]
    for b in rows:
        if b['mode']!='learned':continue
        n=next(r for r in rows if r['case']==b['case'] and r['method']==b['method'] and r['mode']=='exact_after_descent')
        pairs.append(dict(group=b['group'],case=b['case'],method=b['method'],dof=b['dof'],learned_success=b['success'],exact_success=n['success'],learned_collision=b['collision'],exact_collision=n['collision'],relative_error=b['relative_jacobian_error']))
    (ROOT/'summary.json').write_text(json.dumps(dict(table=table,pairs=pairs),indent=2))
    lines=['# E92: same-state post-descent Jacobian intervention','',
        'Selection is explicitly outcome-informed: all ten E91 route300 regressions relative to the original learned baseline, plus the earliest retained success in each checkpoint/DOF stratum represented by those regressions (six controls). These16 contexts are not a representative benchmark. Each is run twice with identical learned approach and planned descent, then learned or exact-Jacobian continuation starting at close. No reset or extra probe occurs at the handoff. Adaptive measured bias, controller statistic updates, physics, geometry filter, phase budgets and success criteria remain unchanged.',
        '', '| Selected group | Continuation | Success | Collisions | Total steps | Median relative Jacobian error at handoff | Error range |','|---|---|---:|---:|---:|---:|---|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['trials']} | {r['collisions']} | {r['steps']} | {r['median_relative_error']:.4f} | {r['min_relative_error']:.4f}-{r['max_relative_error']:.4f} |")
    lines+=['','| Group | Case | Checkpoint | DOF | Learned success | Exact success | Learned collision | Exact collision | Relative error |','|---|---:|---|---:|---|---|---|---|---:|']
    for r in pairs:lines.append(f"| {r['group']} | {r['case']} | {r['method']} | {r['dof']} | {r['learned_success']} | {r['exact_success']} | {r['learned_collision']} | {r['exact_collision']} | {r['relative_error']:.4f} |")
    lines+=['','## Interpretation','',
        'All ten regressed cases recover, and all six retained-success controls remain successful under exact continuation. Seven collisions among the regressions become zero. Since the branches share the same prefix, physical state, measured bias and sufficient statistics at handoff, this is strong controlled evidence that changing the local kinematic estimate can resolve these selected failures. The intervention uses exact Jacobians throughout the remaining task, not only at one instant, and changes subsequent actions, observations and bias estimates.',
        '', 'It does not establish that accumulated history is the specific cause, or that a deployable estimator repair will reproduce these recoveries. Model approximation, insufficient local excitation, nonlinear motion accumulated across the route and subsequent estimation errors remain possible mechanisms. Relative Frobenius error of the6-by-n position/scaled-tool-axis Jacobian is descriptive: the loss group median is0.674 and control median0.590, with overlapping ranges (controls extend to0.969). There is no validated failure threshold or independent statistical inference.',
        '', 'No learned score is improved by this diagnostic. Do not report16/16 as generalization performance: the set was selected using E91 outcomes and the successful continuation is privileged. E91 hybrid scores and failed retention gates remain unchanged.',
        '', '## Next decision','',
        'Investigate a nonprivileged local-history repair with frozen weights. First reconstruct recent motion statistics at the recorded handoff states and compare fixed short-window estimates against the current exponentially accumulated history, without using exact Jacobians as policy inputs. Predeclare any candidate intervention and evaluate full rollouts on all E91 cases, including successful controls, under explicit interaction budgets. Do not tune a success/failure threshold on these16 selected contexts or claim fresh-test performance.',
        '', 'Verification:32 complete saved rollouts;16 exact E91 learned replays; exact approach/descent trace equality; identical handoff qpos/qvel/ctrl/time, measured gram/cross/bias, counters and task flags across branches; unchanged input/checkpoint/source hashes; guarded simulator Jacobian access before handoff except the disclosed read-only audit. All115 EASE tests pass. Four executed video replays match complete saved trajectories and decoded frame counts; manifests store hashes.',
        '', '## Videos','',
        'Gallery: `outputs/ease_learned_handoff_v1/videos/index.html`. Case0 candidate_0 and candidate_2 show matched learned failures and exact-continuation successes. They were prespecified illustrative regressions and are not representative. Labels disclose privileged continuation.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print(json.dumps(table,indent=2))


if __name__=='__main__':run()
