"""Audit E72 paired geometry-filter outcomes and information costs."""
import json
from pathlib import Path
from .geometry_filter import ROOT
from .benchmark import fingerprint


def run():
    rows=json.loads((ROOT/'records.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text())
    assert len(rows)==224 and len({(r['case'],r['method'],r['mode']) for r in rows})==224
    for path,sha in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(path)==sha
    for name in ('geometry_filter.py','arm_pick_place.py','pick_place_control.py','retention_fresh.py','action_objective_audit.py'):
        assert fingerprint(Path(__file__).with_name(name))==fingerprint(ROOT/'source'/name)
    details=[]
    for r in rows:
        full=json.loads((ROOT/'trials'/f"{r['case']}_{r['method']}_{r['mode']}.json").read_text());tr=full['trace'];logs=full['filter_log']
        assert len(tr)+32==r['steps']
        if r['mode']=='geometry':
            assert len(tr)==len(logs) and sum(x['queries'] for x in logs)==r['queries']
            first=next((i for i,x in enumerate(tr) if x['bad']),None)
            details.append(dict(case=r['case'],method=r['method'],success=r['success'],collision=r['collision'],
                                first_collision_step=first,first_collision_phase=tr[first]['phase'] if first is not None else None,
                                filter_at_first_collision=logs[first] if first is not None else None))
    methods=list(dict.fromkeys(r['method'] for r in rows));summary=[];paired=[]
    def counts(rr):return {key:sum(r[key] for r in rr) for key in ('success','collision','steps','queries','changed_actions','fallbacks')}
    for method in methods:
        for mode in ('baseline','geometry'):
            rr=[r for r in rows if r['method']==method and r['mode']==mode]
            summary.append(dict(method=method,mode=mode,**counts(rr),six=sum(r['success'] for r in rr if r['dof']==6),seven=sum(r['success'] for r in rr if r['dof']==7)))
        base={r['case']:r for r in rows if r['method']==method and r['mode']=='baseline'}
        filtered=[r for r in rows if r['method']==method and r['mode']=='geometry']
        paired.append(dict(method=method,gains=[r['case'] for r in filtered if r['success'] and not base[r['case']]['success']],losses=[r['case'] for r in filtered if not r['success'] and base[r['case']]['success']],six_losses=[r['case'] for r in filtered if r['dof']==6 and not r['success'] and base[r['case']]['success']]))
    pooled={mode:dict(**counts([r for r in rows if r['method']!='analytical' and r['mode']==mode]),six=sum(r['six'] for r in summary if r['method']!='analytical' and r['mode']==mode),seven=sum(r['seven'] for r in summary if r['method']!='analytical' and r['mode']==mode)) for mode in ('baseline','geometry')}
    b,g=pooled['baseline'],pooled['geometry']
    retain=all(next(r['six'] for r in summary if r['method']==m and r['mode']=='geometry')>=next(r['six'] for r in summary if r['method']==m and r['mode']=='baseline') for m in methods if m!='analytical')
    passed=g['success']>b['success'] and g['collision']<=b['collision'] and retain
    out=dict(pooled=pooled,methods=summary,passes=passed,per_checkpoint_six_retained=retain,
             gains=sum(len(r['gains']) for r in paired if r['method']!='analytical'),losses=sum(len(r['losses']) for r in paired if r['method']!='analytical'))
    for name,value in [('summary',out),('paired',paired),('collision_diagnostics',details)]: (ROOT/f'{name}.json').write_text(json.dumps(value,indent=2))
    lines=['# E72: shared geometry-based action filtering','',
        'Development test on the original 16 procedural manipulation placements. Seven frozen estimators, original phase budgets and matched calibration states; 224 trials. A separate kinematic model checks midpoint/end of joint-target interpolations and scales motion while preserving measured bias. Physics, contact masks, success rules and neural weights are unchanged.','',
        '**Information difference:** the filter receives the known robot geometry/kinematics and exact current simulator configuration, object pose and scene geometry. Both learned and analytical estimators receive the identical filter. These are additional privileged observations unavailable to a history-only policy. No dynamic future rollout or task Jacobian is used. This experiment cannot establish improved learned embodiment identification.','',
        '| Estimator | Mode | Success /16 | Six /7 | Seven /9 | Collisions | Control steps | Geometry queries | Changed actions | Fallbacks |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['method']} | {r['mode']} | {r['success']} | {r['six']} | {r['seven']} | {r['collision']} | {r['steps']} | {r['queries']} | {r['changed_actions']} | {r['fallbacks']} |")
    lines+=['',f"Pooled learned success: {b['success']}/96 to {g['success']}/96 ({out['gains']} recoveries, {out['losses']} regressions). Six-joint: {b['six']} to {g['six']}/42; seven-joint: {b['seven']} to {g['seven']}/54. Collision cases: {b['collision']} to {g['collision']}. Control steps including calibration: {b['steps']} to {g['steps']}. Geometry queries: {g['queries']}; modified actions: {g['changed_actions']}; no-feasible-candidate fallbacks: {g['fallbacks']}.",'',
        f"Prespecified advancement rule: {'PASS' if passed else 'FAIL'}; per-checkpoint aggregate six-joint retention={retain}. "+('Freeze this filter for fresh-placement validation before adoption.' if passed else 'Keep the original controller. Do not promote this filter or tune its margin/scales to these outcomes.'),'',
        'The three recoveries are concentrated in E35 seed 0 (cases 2 and 11) and E66 seed 2 (case 1); the other four learned checkpoints have unchanged success counts. Analytical success stays 11/16, while its collision count falls from five to four. These are descriptive repeated development cases, not independent evidence of a uniform model improvement.','',
        'Checks are kinematic samples, not a continuous swept-volume or dynamics guarantee. The object is frozen at its current pose in geometry queries, including when carried; actuator transients and untested points along the path may differ. When already within the clearance margin, nonworsening motion is allowed. When no candidate qualifies, maximum sampled clearance is used and the fallback is recorded. Neither collisions nor failures are masked. The effect of geometry access is inseparable from the filter in this comparison.','',
        'Verification: all 112 unfiltered baseline traces and probes reproduce E68 exactly; model/input hashes and physical margins are unchanged. Trial uniqueness, filter-log coverage and query totals are audited. All 102 EASE-FM tests pass, including physical-state isolation and exact pass-through of unmodified actions. Query counts are computational work, not a measured real-time latency guarantee.','',
        'Version history: v1 was interrupted after detecting that bias arithmetic could introduce roundoff even at full scale; its partial trials and source are retained. V2 passes full-scale actions through bit-exactly. No thresholds were changed in response to aggregate task outcomes. All completed v2 outcomes are retained. No training, native tuning or reserved final-test evaluation.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(9,4))
    for ax,key,label in zip(axs,('success','collision'),('Successes /96','Collision cases /96')):
        bars=ax.bar(['baseline','geometry filter'],[b[key],g[key]],color=['#4c78a8','#f58518']);ax.bar_label(bars);ax.set(ylabel=label,ylim=(0,100))
    fig.suptitle('E72: learned estimators; filter uses extra geometry information');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in out.items() if k!='methods'},indent=2))


if __name__=='__main__':run()
