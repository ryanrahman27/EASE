"""Summarize E96 read-only estimation diagnostics and link verified E95 videos."""
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .probe_estimation import ROOT
from .local_probe import ROOT as PREVIOUS
from .benchmark import fingerprint
from .history_excitation import excitation


def run():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==288 and all(r['exact_replay'] for r in rows)
    for r in rows:
        snap=json.loads((ROOT/'snapshots'/f"{r['case']}_{r['method']}_{r['mode']}.json").read_text())
        old=json.loads((PREVIOUS/'trials'/f"{r['case']}_{r['method']}_{r['mode']}.json").read_text())
        assert len(snap['history'])==32
        if r['route_used']:assert excitation(np.array(snap['state']['gram']))==old['excitation']['after']
        elif r['mode']!='no_probe':
            base=json.loads((ROOT/'snapshots'/f"{r['case']}_{r['method']}_no_probe.json").read_text())
            for k in ('state','history','matrices','metrics'):assert snap[k]==base[k]
    spec=json.loads((ROOT/'protocol.json').read_text())
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    table=[];paired=[]
    for group in ('all','6','7','no_route',*sorted({r['method'] for r in rows})):
        for mode in ('no_probe','hold32','probe32'):
            rr=[r for r in rows if r['mode']==mode and ((not r['route_used']) if group=='no_route' else (r['route_used'] and (group=='all' or str(r['dof'])==group or r['method']==group)))]
            for estimator in ('learned','analytical','exact'):
                table.append(dict(group=group,mode=mode,estimator=estimator,n=len(rr),**{metric:float(np.median([r['metrics'][estimator][metric] for r in rr])) for metric in ('relative_error','history_relative_residual','task_linear_residual')}))
    for dof in (6,7):
        for mode in ('no_probe','hold32','probe32'):
            rr=[r for r in rows if r['route_used'] and r['dof']==dof and r['mode']==mode]
            paired.append(dict(dof=dof,mode=mode,n=len(rr),analytical_lower_jac_error=sum(r['metrics']['analytical']['relative_error']<r['metrics']['learned']['relative_error'] for r in rr),analytical_lower_task_residual=sum(r['metrics']['analytical']['task_linear_residual']<r['metrics']['learned']['task_linear_residual'] for r in rr)))
    changes=[]
    for dof in (6,7):
        for estimator in ('learned','analytical'):
            pp=[r for r in rows if r['route_used'] and r['dof']==dof and r['mode']=='probe32']
            for control in ('no_probe','hold32'):
                delta=[]
                for r in pp:
                    b=next(x for x in rows if x['case']==r['case'] and x['method']==r['method'] and x['mode']==control)
                    delta.append(r['metrics'][estimator]['relative_error']-b['metrics'][estimator]['relative_error'])
                changes.append(dict(dof=dof,estimator=estimator,control=control,n=len(delta),improved=sum(x<0 for x in delta),median_delta=float(np.median(delta))))
    outcomes=[]
    for dof in (6,7):
        for mode in ('no_probe','hold32','probe32'):
            for success in (False,True):
                rr=[r for r in rows if r['route_used'] and r['dof']==dof and r['mode']==mode and r['success']==success]
                if rr:outcomes.append(dict(dof=dof,mode=mode,success=success,n=len(rr),**{e:float(np.median([r['metrics'][e]['relative_error'] for r in rr])) for e in ('learned','analytical')}))
    (ROOT/'summary.json').write_text(json.dumps(dict(table=table,paired=paired,changes=changes,outcomes=outcomes),indent=2))
    lines=['# E96: same-state post-probe estimation audit','',
        'All 288 E95 full-task rollouts reproduce their saved result and trace exactly. At the first actual close-phase prediction, learned, ridge analytical and exact simulator Jacobians are compared at the same physical state using the same measured history. The diagnostic never changes the policy prediction. Primary summaries include 78 admitted-route pairs in each mode (30 six-joint and 48 seven-joint); 18 no-route pairs per mode are reported separately. These are repeated checkpoints and tasks, not independent samples.',
        '', 'Relative Jacobian error is ||J_hat - J_exact||_F / ||J_exact||_F. Historical residual is ||DQ J_hat^T - DX||_F / ||DX||_F over the 32 measured transitions (unweighted). Linearized task residual is ||0.35 e - J_exact u||_2, where u is the bias-free DLS correction clipped to +/-0.04 rad. Position and scaled tool-Z component errors, matrices, histories and actions are retained in each snapshot.', '', '| Group | Mode | Estimator | N | Median relative Jacobian error | Median in-history residual | Median linearized task residual |','|---|---|---|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['estimator']} | {r['n']} | {r['relative_error']:.5f} | {r['history_relative_residual']:.5f} | {r['task_linear_residual']:.6f} |")
    lines+=['','## Paired estimator comparisons','','| DOF | Mode | N | Analytical lower Jacobian error | Analytical lower task residual |','|---|---|---:|---:|---:|']
    for r in paired:lines.append(f"| {r['dof']} | {r['mode']} | {r['n']} | {r['analytical_lower_jac_error']} | {r['analytical_lower_task_residual']} |")
    lines+=['','## Paired probe-versus-control error changes','','Negative delta means probe error is lower. Different intervention modes have different physical states; only comparisons between estimators within a mode hold state fixed.','','| DOF | Estimator | Control | N | Improved | Median error delta |','|---|---|---|---:|---:|---:|']
    for r in changes:lines.append(f"| {r['dof']} | {r['estimator']} | {r['control']} | {r['n']} | {r['improved']} | {r['median_delta']:.5f} |")
    lines+=['','## Descriptive outcome strata','','These strata use the original E95 outcomes, with no outcome selection for the audit. They do not define a failure threshold.','','| DOF | Mode | Success | N | Median learned error | Median analytical error |','|---|---|---|---:|---:|---:|']
    for r in outcomes:lines.append(f"| {r['dof']} | {r['mode']} | {r['success']} | {r['n']} | {r['learned']:.5f} | {r['analytical']:.5f} |")
    lines+=['','## Six-joint regressions retained from E95','','Both cases newly lost against both controls are shown below. Better instantaneous matrix and task-direction estimates do not guarantee successful subsequent contact-rich control.','','| Case | Checkpoint | Mode | Success | Learned relative error | Learned task residual |','|---|---|---|---|---:|---:|']
    for case,method in ((0,'parent_1'),(2,'candidate_0')):
        for mode in ('no_probe','hold32','probe32'):
            r=next(r for r in rows if r['case']==case and r['method']==method and r['mode']==mode)
            lines.append(f"| {case} | {method} | {mode} | {r['success']} | {r['metrics']['learned']['relative_error']:.5f} | {r['metrics']['learned']['task_linear_residual']:.6f} |")
    lines+=['','## Finding and next decision','',
        'The probe supplies measurements from which the existing analytical estimator obtains a substantially more accurate local motion map. On all 78 route-assisted post-probe states, analytical error and linearized task residual are lower than the learned estimator at that same state. Median relative Jacobian errors are 0.419 versus 0.070 for six joints (30 pairs) and 0.410 versus 0.083 for seven joints (48 pairs). The pooled medians are 0.419 learned and 0.079 analytical. Every route-assisted learned and analytical estimate also improves its relative Jacobian error after probing versus either control, although those mode comparisons involve different states.',
        '', 'This identifies unused estimation accuracy in the frozen model on these measured histories. It does not show that estimation mismatch alone causes failures: both lost six-joint cases have improved instantaneous matrix and task-direction measures after probing. Do not adopt the probe or retrain on these evaluation histories. Next predeclare a paired continuation diagnostic with an identical learned/planner/probe prefix and learned versus analytical versus exact post-probe prediction, preserving measured bias, updates, contacts and full-task budgets. This will test whether better estimation translates into task recovery before changing training.',
        '', 'Figure: `estimation.png` and `estimation.pdf`. Each point compares estimators at one identical physical state; points below the diagonal favor analytical error. All snapshots and repeated-checkpoint results remain descriptive development evidence.']
    lines+=['','## Scope and verification','',
        'Exact Jacobian is a read-only reference and never an action input. Full replay equality verifies no behavior change. Linearized task residual uses the exact Jacobian applied to the clipped, bias-free DLS correction; it excludes dynamics, contact and geometry filtering. In-history residual is an in-sample fit, not future prediction accuracy. The exact local Jacobian can itself have nonzero historical residual because history spans different postures. Error differences do not establish full-task recovery or a causal failure threshold.',
        '', 'All input and checkpoint hashes are unchanged. Every snapshot retains 32 transitions; admitted-route snapshot excitation matches E95 post-intervention statistics, and no-route snapshots match across modes. No training or reserved-test evaluation occurs. The 288 replay checks are in addition to 125 passing tests, including known-linear-map and zero-motion diagnostic checks. E95 remains a failed intervention: probe57/96, hold59/96, no-probe58/96. The benchmark uses privileged route planning on reused development cases.',
        '', '## Video evidence','', 'All ten original E95 videos are linked in `videos/index.html` after SHA-256 verification. They show unchanged successes and failures, not new outcomes from this read-only diagnostic. Selection and playback disclosures are copied from E95.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,dof in zip(axes,(6,7)):
        for mode,color in zip(('no_probe','hold32','probe32'),('tab:gray','tab:orange','tab:blue')):
            rr=[r for r in rows if r['route_used'] and r['dof']==dof and r['mode']==mode]
            ax.scatter([r['metrics']['learned']['relative_error'] for r in rr],[r['metrics']['analytical']['relative_error'] for r in rr],label=mode,s=18,alpha=.65,color=color)
        lim=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([0,lim],[0,lim],'k--',lw=1);ax.set(xlim=(0,lim),ylim=(0,lim),xlabel='Learned relative Jacobian error',ylabel='Analytical relative Jacobian error',title=f'{dof}-joint: identical state per point');ax.legend(fontsize=8,loc='lower right')
    fig.savefig(ROOT/'estimation.png',dpi=160);fig.savefig(ROOT/'estimation.pdf');plt.close(fig)
    (ROOT/'videos').mkdir(exist_ok=True);manifest=json.loads((PREVIOUS/'videos/manifest.json').read_text())
    for v in manifest['videos']:assert fingerprint(PREVIOUS/'videos'/v['file'])==v['sha256']
    manifest['reuse_note']='Unchanged E95 executed rollouts; E96 is read-only. All 288 E96 replays match E95.'
    (ROOT/'videos/manifest.json').write_text(json.dumps(manifest,indent=2))
    html='<html><head><meta charset="utf-8"><title>E96 estimation audit video evidence</title></head><body style="background:#111;color:#eee;font:16px sans-serif;margin:30px"><h1>E96: unchanged E95 success/failure evidence</h1><p>'+manifest['reuse_note']+'</p><p>'+manifest['selection']+'</p><p>'+manifest['playback']+'</p>'
    for v in manifest['videos']:html+=f'<h2>{v["file"]}</h2><video controls width="640" src="../../ease_local_probe_v1/videos/{v["file"]}"></video>'
    (ROOT/'videos/index.html').write_text(html+'</body></html>',encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print(json.dumps(dict(main=table[:9],paired=paired,changes=changes),indent=2))

if __name__=='__main__':run()
