"""Report E112 without selecting cohorts or suppressing regressions."""
import json
import shutil
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .unseen_progress import ROOT

def run():
    assert json.loads((ROOT/'status.json').read_text())['status']=='complete'
    rows=json.loads((ROOT/'records.json').read_text());tasks=json.loads((ROOT/'cases.json').read_text());screen=json.loads((ROOT/'screening.json').read_text())
    assert len(rows)==len(tasks)*6
    table=[];changes=[];diagnostics=[]
    for group in ('all','6','7','seed0','seed1','seed2',*[f'body{i}' for i in range(8)]):
        for mode in ('baseline','physics'):
            rr=[r for r in rows if r['mode']==mode and (group=='all' or group==str(r['dof']) or group==f"seed{r['seed']}" or group==f"body{r['body']}")]
            table.append(dict(group=group,mode=mode,n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),contact_free_failures=sum(not r['success'] and not r['collision'] for r in rr),changed_actions=sum(r['changed_actions'] for r in rr)))
    old=[r for r in rows if r['mode']=='baseline'];new=[r for r in rows if r['mode']=='physics']
    for r in new:
        ref=next(x for x in old if (x['case'],x['seed'])==(r['case'],r['seed']))
        if r['success']!=ref['success']:changes.append(dict(case=r['case'],seed=r['seed'],body=r['body'],dof=r['dof'],change='gain' if r['success'] else 'loss'))
    for r in rows:
        if r['success']:continue
        full=json.loads((ROOT/'trials'/f"{r['case']}_{r['seed']}_None_{r['mode']}.json").read_text());bad=next((v for v in full['trace'] if v['bad']),None)
        status=None
        if bad is not None:
            step=32+next(i for i,v in enumerate(full['trace']) if v['bad'])
            a=next((a for a in full['physics_audit'] if a['step']==step),None)
            if a is None:status='unmodified phase'
            elif not a['clean']:status='contact already at control boundary'
            elif a['alternatives'] and a['choice']=='original':status='no candidate meets all frozen criteria'
            else:status='original control or inspect trace'
        diagnostics.append(dict(case=r['case'],seed=r['seed'],mode=r['mode'],first_collision_phase=None if bad is None else bad['phase'],selector_at_first_collision=status,final_xy_error=float(np.linalg.norm(np.asarray(r['final_object'][:2])-tasks[r['case']]['target'])),changed_actions=r['changed_actions'],unreached_phases=[p['phase'] for p in r['phases'] if not p['reached']]))
    gates=dict(success_improves=sum(r['success'] for r in new)>sum(r['success'] for r in old),retains_all_successes=not any(c['change']=='loss' for c in changes),collisions_nonincreasing=sum(r['collision'] for r in new)<=sum(r['collision'] for r in old))
    for key,values in [('seed',range(3)),('dof',(6,7))]:gates[f'per_{key}_nondecreasing']=all(sum(r['success'] for r in new if r[key]==v)>=sum(r['success'] for r in old if r[key]==v) for v in values)
    differences=np.array([sum(r['success'] for r in new if r['body']==b)-sum(r['success'] for r in old if r['body']==b) for b in range(8)])/6
    rng=np.random.default_rng(1125000);boot=differences[rng.integers(0,8,(10000,8))].mean(1)
    interval=dict(pipeline_difference=float(differences.mean()),percentile_95=np.quantile(boot,[.025,.975]).tolist(),body_differences=differences.tolist(),clusters=8,resamples=10000)
    timing=json.loads((ROOT/'timings.json').read_text());cost={}
    for search in (False,True):
        tt=[t['ms'] for t in timing if t['mode']=='physics' and t['search']==search]
        cost['search' if search else 'check_only']=dict(n=len(tt),median_ms=float(np.median(tt)) if tt else None,p95_ms=float(np.quantile(tt,.95)) if tt else None)
    failure_counts=dict(Counter((d['first_collision_phase'] or 'no collision')+' / '+(d['selector_at_first_collision'] or 'task criteria failed') for d in diagnostics if d['mode']=='physics'))
    gain_bodies=sorted({c['body'] for c in changes if c['change']=='gain'})
    summary=dict(table=table,changes=changes,gates=gates,passes=all(gates.values()),cluster_bootstrap=interval,screened_candidates=len(screen),accepted_candidates=len(tasks),compute=cost,selector_failure_counts=failure_counts,gain_bodies=gain_bodies)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));(ROOT/'failure_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    lines=['# E112: frozen progress-aware selector on prospective unseen bodies','',f'{len(tasks)}/16 candidate placements passed endpoint IK screening; {len(rows)} complete paired rollouts across all three checkpoint seeds. Every rejection and accepted-task failure is retained. No route planner, extra task budget, weight update or selector tuning.',
        '', 'Eight new parameterizations of four E87 axis orders, two bodies per order and two placements per body. The axis layouts differ from the original training bodies but were inspected in earlier development. This is a prospective body-parameter replication, not an untouched axis-family test or the reserved final test. Procedural simulator arms; exact physics/state/geometry and IK initialization remain privileged.',
        '', '| Group | Mode | Success | Collisions | Contact-free failures | Changed actions |','|---|---|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['n']} | {r['collisions']} | {r['contact_free_failures']} | {r['changed_actions']} |")
    lines+=['','## Paired changes','',json.dumps(changes,indent=2),'','## Frozen transfer gate','',f"**{'PASS' if summary['passes'] else 'FAIL'}**",'',json.dumps(gates,indent=2),'','## Screening and uncertainty','',json.dumps(screen,indent=2),
        '', 'Pipeline success uses all48 candidate/seed pairs, with screened-out placements counted as zero for both modes. The bootstrap resamples eight whole bodies, preserving correlated placements and checkpoints; 10,000 fixed-seed resamples provide descriptive intervals, not evidence from48 independent embodiments.', '',json.dumps(interval,indent=2),'','## Compute','',json.dumps(cost,indent=2),
        '', 'Added prediction/search wall time excludes neural inference, original static filtering and normal execution. Compare search latency with the40ms control interval; this is not a hardware real-time claim.',
        '', '## Interpretation','',('The frozen selector passes the prespecified transfer gate on this prospective cohort. This supports transfer of the privileged physics-assisted component; it does not isolate learned history adaptation or demonstrate history-only control.' if summary['passes'] else 'The frozen selector fails the prespecified transfer gate. Aggregate gains, if present, do not erase paired regressions or per-DOF/per-seed declines. Preserve this result before proposing changes; this cohort is no longer untouched for later tuning.'),
        '', 'Before a paper claim, separate the history-conditioned neural component from exact-model search using analytical/specialist/history ablations, assess compute, and broaden independent embodiment evidence. No method changes were made after outcomes.',
        '',f"Recovered episodes occur on {len(gain_bodies)}/8 bodies (body IDs {gain_bodies}). The body-bootstrap interval includes zero when its lower endpoint is zero; passing the operational retention gate is not statistical confirmation of broad transfer. Each of the three observed recoveries in this run uses one action substitution. The six-joint controller makes65 substitutions without changing any task outcome; safe local progress is insufficient for those failures.",
        '', '## Remaining failures','',json.dumps(failure_counts,indent=2),
        '', 'In this cohort, all ten remaining collision episodes are six-joint: six first collide in descend/lower when no candidate satisfies every frozen criterion, and four first collide during the unmodified release phase. The seven contact-free failures are seven-joint; six belong to one body that misses descent/close tolerances. These are trace-level diagnostics, not controlled causal ablations. The next diagnostic should examine those six-joint states and release transitions without tuning this frozen evaluation result.',
        '', '## Verification and artifacts','',json.dumps(json.loads((ROOT/'verification.json').read_text()),indent=2),
        '',json.dumps(json.loads((ROOT/'adapter_verification.json').read_text()),indent=2),
        '', 'Saved protocol, source snapshots, checkpoint/input hashes, all raw traces and failures, timing measurements, and replay-verified success/failure videos. Twelve prespecified E111 compatibility runs match full serialized results. All selected predicted next states/contact unions match execution; all paired calibration/approach prefixes match. See failure_diagnostics.json for every failed rollout.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric,title in zip(axes,('success','collisions'),('Successful tasks','Tasks with forbidden contact')):
        for j,mode in enumerate(('baseline','physics')):
            vals=[next(r[metric] for r in table if r['group']==g and r['mode']==mode) for g in ('all','6','7')]
            bars=ax.bar(np.arange(3)+(j-.5)*.35,vals,width=.35,label=mode);ax.bar_label(bars,padding=3)
        ax.set_xticks(range(3),['All','Six-joint','Seven-joint']);ax.set_ylim(0,len(tasks)*3+3);ax.set_title(title);ax.legend()
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'comparison.{ext}')
    plt.close(fig);print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
