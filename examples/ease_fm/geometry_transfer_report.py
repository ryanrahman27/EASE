"""E73 paired outcomes and slowing-control comparison."""
import json
from pathlib import Path
from .geometry_transfer import ROOT
from .benchmark import fingerprint


def run():
    rows=json.loads((ROOT/'records.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text())
    assert len(rows)==357 and len({(r['case'],r['method'],r['mode']) for r in rows})==357
    for p,h in {**spec['inputs'],**spec['checkpoints'],**spec['sources']}.items():assert fingerprint(p)==h
    assert fingerprint(Path(__file__).with_name('geometry_transfer.py'))==fingerprint(ROOT/'source'/'geometry_transfer.py')
    for r in rows:
        full=json.loads((ROOT/'trials'/f"{r['case']}_{r['method']}_{r['mode']}.json").read_text())
        assert len(full['trace'])+32==r['steps'] and len(r['phases'])==9
        if r['mode']!='geometry':assert not full['filter_log'] and r['queries']==0
        else:assert len(full['filter_log'])==len(full['trace']) and sum(x['queries'] for x in full['filter_log'])==r['queries']
    def counts(rr):
        return dict(n=len(rr),**{key:sum(r[key] for r in rr) for key in ('success','collision','steps','queries','changed_actions','fallbacks')},
                    six=sum(r['success'] for r in rr if r['dof']==6),seven=sum(r['success'] for r in rr if r['dof']==7))
    methods=list(dict.fromkeys(r['method'] for r in rows));modes=('baseline','geometry','slow')
    summary=[dict(method=m,mode=mode,**counts([r for r in rows if r['method']==m and r['mode']==mode])) for m in methods for mode in modes]
    pooled={mode:counts([r for r in rows if r['method']!='analytical' and r['mode']==mode]) for mode in modes}
    paired=[]
    for method in methods:
        for mode,reference in [('geometry','baseline'),('slow','baseline'),('geometry','slow')]:
            ref={r['case']:r for r in rows if r['method']==method and r['mode']==reference}
            rr=[r for r in rows if r['method']==method and r['mode']==mode]
            paired.append(dict(method=method,mode=mode,reference=reference,gains=[r['case'] for r in rr if r['success'] and not ref[r['case']]['success']],losses=[r['case'] for r in rr if not r['success'] and ref[r['case']]['success']],six_losses=[r['case'] for r in rr if r['dof']==6 and not r['success'] and ref[r['case']]['success']]))
    b,g,s=[pooled[m] for m in modes]
    retained=all(next(r['six'] for r in summary if r['method']==m and r['mode']=='geometry')>=next(r['six'] for r in summary if r['method']==m and r['mode']=='baseline') for m in methods if m!='analytical')
    passes=g['success']>b['success'] and g['collision']<=b['collision'] and retained
    aggregates=[]
    for mode,ref in [('geometry','baseline'),('slow','baseline'),('geometry','slow')]:
        pp=[r for r in paired if r['method']!='analytical' and r['mode']==mode and r['reference']==ref]
        aggregates.append(dict(mode=mode,reference=ref,gains=sum(len(r['gains']) for r in pp),losses=sum(len(r['losses']) for r in pp),six_losses=sum(len(r['six_losses']) for r in pp)))
    families=[dict(family=f,mode=m,**counts([r for r in rows if r['family']==f and r['mode']==m and r['method']!='analytical'])) for f in dict.fromkeys(r['family'] for r in rows) for m in modes]
    out=dict(pooled=pooled,methods=summary,paired_totals=aggregates,families=families,passes=passes,per_checkpoint_six_retained=retained,beats_fixed_slowing=g['success']>s['success'])
    (ROOT/'summary.json').write_text(json.dumps(out,indent=2));(ROOT/'paired.json').write_text(json.dumps(paired,indent=2))
    lines=['# E73: geometry-filter transfer and fixed slowing control','',
        'All 17 accepted E71 placements on six existing bodies; all seven frozen estimators and three conditions, 357 trials. The 17/18 screening record and rejected case remain unchanged. This set was not used to design the frozen E72 filter, but its baseline outcomes were already observed in E71; it is development validation, not the reserved final test or new-embodiment transfer.','',
        '| Estimator | Mode | Success /17 | Six /8 | Seven /9 | Collisions | Steps | Geometry queries |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['method']} | {r['mode']} | {r['success']} | {r['six']} | {r['seven']} | {r['collision']} | {r['steps']} | {r['queries']} |")
    lines+=['','| Learned models pooled | Success /102 | Six /48 | Seven /54 | Collisions | Steps |','|---|---:|---:|---:|---:|---:|']
    for mode,r in pooled.items():lines.append(f"| {mode} | {r['success']} | {r['six']} | {r['seven']} | {r['collision']} | {r['steps']} |")
    lines+=['',f"Prespecified geometry replication rule: {'PASS' if passes else 'FAIL'}. Per-checkpoint aggregate six-joint retention={retained}. Geometry success exceeds this fixed slowing control={g['success']>s['success']}.",'']
    for r in aggregates:lines.append(f"- {r['mode']} versus {r['reference']}: {r['gains']} paired recoveries, {r['losses']} regressions, including {r['six_losses']} six-joint regressions.")
    lines+=['',f"Geometry control uses {100*(g['steps']/b['steps']-1):+.1f}% control steps versus baseline, {g['queries']} geometry queries, {g['changed_actions']} changed actions and {g['fallbacks']} no-feasible-candidate fallbacks. Fixed slowing uses {100*(s['steps']/b['steps']-1):+.1f}% control steps and zero geometry queries.",'',
        'The filter is unchanged E72v2: known geometry and exact current simulator pose, 2 mm margin, midpoint/endpoint kinematic checks, action scales 1/.5/.25/0 with measured bias preserved. The slowing control uses a fixed .5 scale on the same motion component, preserves bias and leaves finger commands unchanged. All conditions use the original phase budgets and physics. Slowing is not matched to the filter\'s per-step action distribution or total interaction time; one constant cannot rule out all geometry-free schedules.','',
        'Geometry remains extra privileged information shared across all estimators. A result above the slowing control is not learned-identifier superiority, and a filtered task failure may reflect dynamics outside the sampled kinematic checks. These experiments do not evaluate hardware, latency compliance or a safety guarantee.','',
        'The filter changes 238 learned-model actions (about 0.9% of task steps) and performs 80,912 geometry queries, yet all learned casewise success outcomes remain identical to baseline. Its roughly 0.3% reduction in simulated control steps is not evidence of lower computation or real-time latency. The slowing comparison has five recoveries and eleven regressions, so its net loss also masks mixed casewise effects.','',
        'Verification: 119 baseline traces and calibration probes exactly reproduce E71. E72 code, model and input hashes are unchanged; physical margins remain unchanged. Geometry access is blocked in both baseline and slowing controls. Trial coverage, nine-phase trajectories and filter/query logs are audited. All 103 EASE-FM tests pass. All failures retained; no training, native tuning or reserved final-test rollout.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,key,label in zip(axs,('success','collision'),('Successful tasks /102','Collision cases /102')):
        bars=ax.bar(modes,[pooled[m][key] for m in modes],color=['#4c78a8','#f58518','#54a24b']);ax.bar_label(bars);ax.set(ylabel=label,ylim=(0,106))
    fig.suptitle('E73: frozen geometry filter versus fixed half-speed control');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in out.items() if k not in ('methods','families')},indent=2))


if __name__=='__main__':run()
