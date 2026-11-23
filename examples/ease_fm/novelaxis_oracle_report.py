"""Report the privileged E88 intervention without pooling it as learned success."""
import json
from .novelaxis_oracle import ROOT


def run():
    rows=json.loads((ROOT/'records.json').read_text());audit=json.loads((ROOT/'audit.json').read_text())
    assert len(rows)==32 and audit['exact_baselines']==16
    table=[]
    for group in ('all','6','7',*sorted({r['family'] for r in rows})):
        for mode in ('analytical','exact_jacobian'):
            a=[r for r in rows if r['mode']==mode and (group=='all' or str(r['dof'])==group or r['family']==group)]
            table.append(dict(group=group,mode=mode,trials=len(a),success=sum(r['success'] for r in a),collisions=sum(r['collision'] for r in a),steps=sum(r['steps'] for r in a)))
    changes=[]
    for b in rows:
        if b['mode']!='analytical':continue
        n=next(r for r in rows if r['case']==b['case'] and r['mode']=='exact_jacobian')
        if b['success']!=n['success'] or b['collision']!=n['collision']:
            changes.append(dict(case=b['case'],dof=b['dof'],baseline_success=b['success'],exact_success=n['success'],baseline_collision=b['collision'],exact_collision=n['collision']))
    (ROOT/'summary.json').write_text(json.dumps(dict(table=table,paired_changes=changes),indent=2))
    lines=['# E88: exact-Jacobian diagnostic on unseen arm arrangements','',
        'All 16 fixed E87 cases were evaluated with the analytical history estimator and an exact instantaneous simulator Jacobian. Only the Jacobian source changes: calibration, adaptive measured bias, geometry filter, phase sequence, task budgets, physics and success rules remain unchanged. Both variants continue updating measured statistics. Later histories differ because actions differ. No direction correction is applied. This is a privileged diagnostic, not a learned-policy result or a guaranteed performance upper bound.','',
        '| Group | Controller | Success | Collisions | Steps including calibration |','|---|---|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['trials']} | {r['collisions']} | {r['steps']} |")
    lines+=['','Paired outcome changes:','']
    for r in changes:lines.append('- '+json.dumps(r,sort_keys=True))
    lines+=['','Exact kinematics increases success from 7/16 to 8/16 and reduces collisions from 7 to 6. The sole recovery is six-joint case 2; no successes are lost. Six-joint success is 3/8 versus 2/8; seven-joint remains 5/8. All four ZYXYZY cases still collide and fail with exact Jacobians. Cases 1 and 8 fail without collisions in both variants.',
        '', 'Interpretation: local Jacobian estimation error explains at least one recoverable failure under this intervention, but correcting it does not resolve most failures. The shared controller, geometric path selection, actuator dynamics, grasp execution and phase logic remain candidate limitations. This experiment does not isolate which of them is causal, and exact local kinematics does not establish a collision-free global path.',
        '', 'Next: inspect the difficult six-joint family with a separately labeled path-feasibility diagnostic. Compare the actual descent path with feasible endpoint configurations and identify whether an alternate arm posture can reach the grasp without table contact. Keep fixed-case outcomes separate from any subsequently tuned controller. Do not claim broad transfer reliability or initiate more estimator training solely from this result.',
        '', 'Verification: 32 saved task traces; all 16 analytical baselines exactly reproduce E87; all calibration probes match; exact Jacobian call counts equal task steps and are zero during calibration; input, checkpoint and source hashes remain unchanged. All 109 EASE tests pass, including the new test that exact-Jacobian substitution preserves measured bias and statistics. No retraining or reserved-test evaluation occurred.',
        '', '## Saved video evidence','',
        'Videos are illustrative, explicitly outcome-selected examples, not a representative success-rate sample. The manifest records reference hashes, video hashes, frame counts and exact replay checks. Playback shows complete tasks at real-time simulation speed, omitting calibration; controller, final outcome, phase and cumulative collision status are labeled.',
        '', '- Matched E87 candidate_0 case 6: baseline failure and response-corrected success.',
        '- E88 exact-Jacobian case 2: success; case 4: failure.',
        '- Gallery: `outputs/ease_novelaxis_oracle_v1/videos/index.html`.',
        '', 'Ongoing recording convention: save protocol/settings before evaluation, each completed rollout immediately, aggregate results and audits at completion, and labeled video demos of both successes and failures when present. Verify video replays against saved results and disclose example selection. Preserve old artifacts.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(main=table[:6],changes=changes),indent=2))


if __name__=='__main__':run()
