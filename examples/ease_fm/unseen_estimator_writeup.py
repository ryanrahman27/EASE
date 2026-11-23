"""Prospective body-level comparison, without treating placements as bodies."""
from pathlib import Path
import shutil
import numpy as np
from .unseen_estimator import ROOT,CONDITIONS
from .abstaining_task_value_gate import read,save,fingerprint

def write_report(summary,metrics,case_rows,paired,pairs):
    tasks=read(ROOT/'cases.json');bodies=read(ROOT/'bodies.json');screen=read(ROOT/'screening.json');body_rows=[]
    for body in bodies:
        ids=[i for i,t in enumerate(tasks) if t['body_index']==body['body']];row=dict(body=body['body'],seed=body['arm']['seed'],family=body['arm']['family'],n=len(ids),methods={})
        for m in CONDITIONS:
            rr=[case_rows[m][i] for i in ids];row['methods'][m]=dict(success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),steps=sum(r['control_steps'] for r in rr),cases=ids)
        body_rows.append(row)
    accepted=[r for r in body_rows if r['n']];differences=np.array([(r['methods']['e120_ensemble']['success']-r['methods']['analytical']['success'])/r['n'] for r in accepted]);rng=np.random.default_rng(1779000);samples=differences[rng.integers(0,len(differences),(10000,len(differences)))].mean(1);interval=np.quantile(samples,[.025,.975]).tolist()
    # Body bootstrap is descriptive; axis families and initialization design are shared.
    groups={}
    for label,is_dev in (('familiar_axis_layouts',False),('prior_development_axis_layouts',True)):
        ids=[i for i,t in enumerate(tasks) if t['arm']['family'].startswith('axis_dev')==is_dev];groups[label]={}
        for m in CONDITIONS:
            rr=[case_rows[m][i] for i in ids];groups[label][m]=dict(n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr))
    save(ROOT/'body_results.json',dict(bodies=body_rows,layout_groups=groups,descriptive_body_bootstrap=dict(seed=1779000,resamples=10000,mean_paired_rate_difference=float(differences.mean()),percentile95=interval,independent_placement_claim=False)))
    lines=['# E177: frozen-controller transfer to prospective unseen bodies','','New body parameters were sampled after E176 was frozen. Twelve procedural six-joint bodies, three in each of four previously used axis families; four fixed placements per body. The cohort and all controller settings were frozen before initialization screening and before any prospective policy outcome. No replacement, training, tuning or best-checkpoint selection.','',f"Initialization accepted {len(tasks)}/{len(screen)} candidate tasks across {len(accepted)}/{len(bodies)} bodies. Only the starting posture was screened; no grasp, transport or policy-success screening. Rejections remain in screening.json and the all-candidate denominator.",'','| Estimator | Success / accepted | Success / all48 candidates | Collision episodes | Total control steps | Median steps |','|---|---:|---:|---:|---:|---:|']
    for m in CONDITIONS:
        x=summary['conditions'][m]['outcomes'];v=metrics[m];lines.append(f"| {m} | {x['success']}/{x['n']} | {x['success']}/48 | {x['collisions']} | {v['total_steps']} | {v['median_steps']:.1f} |")
    lines+=['',f'Paired outcomes: {paired}.','','## What is frozen and matched','','The AST-equivalence audit verifies unchanged geometry/projection helpers, ProtectedArm, phase targets, controller attachment and full rollout/scoring. The estimator evaluator is unchanged except removal of assertions against unrelated archived fitting cases. Both estimators exactly replayed five fitting cases (including successes and failures):10 compatibility episodes before prospective evaluation. E120 is the same reference/correction three-model ensemble with its bounded residual; analytical is the original regularized history solve. Neither is retrained.','','Both methods have identical body/placement/home/calibration, frozen calibrated bias,32 calibration steps and32 local probe steps. They share E175/E176 recovery gates, strict grasp-first planner, table/link/gripper and probe protections, causal clearance allowance, grasp guard and265mm release. Both have the same conditional step caps:944 total including calibration with an accepted route,744 otherwise. Descent is100 steps or300 after an accepted route; all other caps and early exits are unchanged. Actual steps differ and are reported above. Planning time/computation is not equalized. Exact geometry, object pose, contact information and IK initialization remain privileged.','','The independent audit reconstructs all histories from calibration and saved state transitions, verifies both estimator statistics and frozen bias, reconstructs analytical predictions/actions, checks executed commands and safety constraints, verifies every saved post-state geometry, and audits static paths, grasp guards and budgets. All raw trials, states, actions, histories and planning outcomes are saved.','','## Per-body results','','| Body seed | Axis family | Accepted tasks | E120 successes | Analytical successes | E120 / analytical collisions |','|---|---|---:|---:|---:|---|']
    for r in body_rows:
        a,b=r['methods']['e120_ensemble'],r['methods']['analytical'];lines.append(f"| {r['seed']} | {r['family']} | {r['n']} | {a['success']} | {b['success']} | {a['collisions']} / {b['collisions']} |")
    lines+=['',f"Descriptive paired body-mean success-rate difference (E120 minus analytical): {100*differences.mean():.2f} percentage points; body-bootstrap percentile95 interval [{100*interval[0]:.2f}, {100*interval[1]:.2f}] points (10,000 resamples, seed1779000). Only12 body clusters and four related axis families; this is not48 independent embodiment trials or a definitive significance claim.",'','## Layout groups','','| Previously used layouts | E120 successes | Analytical successes | E120 / analytical collisions |','|---|---|---|---|']
    for label,g in groups.items():
        a,b=g['e120_ensemble'],g['analytical'];lines.append(f"| {label} | {a['success']}/{a['n']} | {b['success']}/{b['n']} | {a['collisions']} / {b['collisions']} |")
    lines+=['','## Discordant task outcomes','','| Case | Body | E120 success | Analytical success | E120 collision | Analytical collision |','|---|---|---|---|---|---|']
    for p in pairs:
        if p['e120_success']!=p['analytical_success'] or p['e120_collision']!=p['analytical_collision']:lines.append(f"| {p['case']} | {tasks[p['case']]['arm']['seed']} | {p['e120_success']} | {p['analytical_success']} | {p['e120_collision']} | {p['analytical_collision']} |")
    lines+=['','## Scope and interpretation','','These body parameters and policy outcomes are prospective and absent from pre-existing case manifests. The four axis layouts were already used in training/development: two familiar fitting layouts and two prior development layouts. This tests transfer to newly sampled kinematics/dynamics within those layouts, not untouched axis-family generalization, commercial-arm transfer, hardware, or a learned-only policy. The inherited controller was developed around E120; equal frozen rules do not remove that development bias. No test-based adjustments are made. Legacy reserved cohorts remain untouched.','','A performance drop is retained as evidence about transfer of this entire privileged control stack. We do not repair failures within this cohort and then call it an untouched test. Task successes require the original grasp/lift/place/no-collision score; static feasibility or reaching a hand waypoint is not success. Videos show both estimators for the first case of each body and every outcome-discordant case, with additional categories if necessary. All cases retain full raw artifacts regardless of video selection.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)

def finish():
    videos=read(ROOT/'videos/manifest.json')['videos'];assert len(videos)==len(read(ROOT/'video_records.json'))
    for v in videos:assert v['all_saved_states_match_trace'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    s=read(ROOT/'summary.json')['conditions'];paired=read(ROOT/'paired.json')['summary'];n=len(read(ROOT/'cases.json'));paragraph=f"E120 {s['e120_ensemble']['outcomes']['success']}/{n}, analytical {s['analytical']['outcomes']['success']}/{n}; collisions {s['e120_ensemble']['outcomes']['collisions']} versus {s['analytical']['outcomes']['collisions']}. Paired: {paired}.12 prospectively sampled bodies across4 previously used axis families,4 placements each. Frozen E176 controller/budgets;10 exact fitting compatibility replays. No test-based tuning or training. Privileged simulator inputs; not untouched axis-family or hardware evaluation."
    section='\n## E177: prospective unseen-body estimator comparison\n\n'+paragraph+'\n\nReport: `outputs/ease_unseen_estimator_v1/report.md`; success/failure videos: `videos/index.html`.\n';review=Path('outputs/ease_iteration_review_v1')
    for p in (Path(__file__).with_name('TECHNICAL_REPORT.md'),review/'report.md'):
        text=p.read_text(encoding='utf-8')
        if '## E177:' not in text:p.write_text(text+section,encoding='utf-8')
    p=review/'WORKING_STATE.md';text=p.read_text(encoding='utf-8')
    if not text.startswith('# Latest checkpoint: E177'):p.write_text('# Latest checkpoint: E177 complete\n'+section+'\nAll runs, audits and videos complete. This cohort is now observed; do not reuse it as untouched.\n\n---\n'+text,encoding='utf-8')
    studies=read(review/'studies.json')
    if not any(x['experiment']=='E177' for x in studies):studies.append(dict(experiment='E177',folder=str(ROOT),report_exists=True,videos=len(videos)));save(review/'studies.json',studies)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'status.json',dict(status='complete',new_episodes=2*n,preflight_exact_replays=10,videos=len(videos),prospective_bodies=True,axis_families_previously_used=True,controller_frozen=True,reserved_legacy_tests_evaluated=False));save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print(paragraph,flush=True)

if __name__=='__main__':finish()
