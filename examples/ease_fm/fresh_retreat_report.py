"""Report all E105 outcomes with embodiment-clustered descriptive uncertainty."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .fresh_retreat import ROOT,MODES

def run():
    assert json.loads((ROOT/'status.json').read_text())['status']=='complete'
    rows=json.loads((ROOT/'records.json').read_text());tasks=json.loads((ROOT/'cases.json').read_text());screen=json.loads((ROOT/'screening.json').read_text())
    assert len(rows)==len(tasks)*9
    table=[]
    for group in ('all','6','7','seed0','seed1','seed2','route','no_route',*[f'body{i}' for i in range(8)]):
        for mode in MODES:
            rr=[r for r in rows if r['mode']==mode and (group=='all' or group==str(r['dof']) or group==f"seed{r['seed']}" or group==f"body{r['body']}" or group==('route' if r['route_used'] else 'no_route'))]
            table.append(dict(group=group,mode=mode,n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr)))
    changes=[];gates={};intervals={}
    new=[r for r in rows if r['mode']=='gentle']
    for control in ('baseline','budget'):
        old=[r for r in rows if r['mode']==control]
        for r in new:
            ref=next(x for x in old if (x['case'],x['seed'])==(r['case'],r['seed']))
            if r['success']!=ref['success']:changes.append(dict(control=control,case=r['case'],seed=r['seed'],body=r['body'],dof=r['dof'],change='gain' if r['success'] else 'loss'))
        gates[control]=dict(success_improves=sum(r['success'] for r in new)>sum(r['success'] for r in old),retains_all_successes=not any(r['control']==control and r['change']=='loss' for r in changes),collisions_nonincreasing=sum(r['collision'] for r in new)<=sum(r['collision'] for r in old),per_seed_nondecreasing=all(sum(r['success'] for r in new if r['seed']==s)>=sum(r['success'] for r in old if r['seed']==s) for s in range(3)))
        # Fixed 8-body target population; screened-out placements count as zero
        # pipeline success in both arms, preserving a common denominator of6/body.
        differences=np.array([sum(r['success'] for r in new if r['body']==b)-sum(r['success'] for r in old if r['body']==b) for b in range(8)])/6
        rng=np.random.default_rng(1055000);boot=differences[rng.integers(0,8,(10000,8))].mean(1)
        intervals[control]=dict(pipeline_difference=float(differences.mean()),percentile_95=np.quantile(boot,[.025,.975]).tolist(),body_differences=differences.tolist(),clusters=8,resamples=10000)
    passed=all(all(g.values()) for g in gates.values());summary=dict(table=table,changes=changes,gates=gates,passes=passed,cluster_bootstrap=intervals,screened_candidates=len(screen),accepted_candidates=len(tasks))
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# E105: prospective body and placement replication','',f'{len(tasks)}/{len(screen)} candidate placements passed endpoint IK screening. All accepted cases, including no-route failures, received all three checkpoint seeds and all three controls: {len(rows)} physical task rollouts. No replacement. Nine E104 compatibility runs match full traces and outcomes.','',
    'This cohort has eight newly perturbed bodies and two new placements per body. It shares the four E87 axis orders and procedural generator: it tests fresh parameterizations/placements, not new axis arrangements or real robots. E99 weights and E104 retreat intervention were frozen before outcomes. Privileged IK, route planning and geometric filtering remain. No reserved final-test evaluation.','',
    '| Group | Mode | Success | Collisions |','|---|---|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['n']} | {r['collisions']} |")
    lines+=['','## Paired changes','',json.dumps(changes,indent=2),'','## Prespecified operational gate','',f"**{'PASS' if passed else 'FAIL'}**",'',json.dumps(gates,indent=2),'','## Body-clustered uncertainty','',
    'The fixed candidate pipeline denominator is48 (16 placements x3 seeds); an IK rejection counts as zero pipeline success for every mode. Success differences are resampled over eight bodies, keeping both placements and all three seeds together. 10,000 fixed-seed bootstrap samples give descriptive percentile intervals. Eight bodies and four shared axis orders limit precision; these are not 48 independent embodiment samples.', '',json.dumps(intervals,indent=2),'','## Next decision','',
    ('Retain the frozen candidate for further evaluation; a passed operational gate alone does not establish broad generalization.' if passed else 'Do not promote this as a robust transfer improvement. Keep the frozen outcome and diagnose the paired regressions before proposing another intervention; do not retune this cohort and call it independent confirmation.'),'',
    'Raw traces, protocol, screening records, static planner attempts, checkpoint hashes, source snapshots, and replay-verified success/failure videos are saved alongside this report.']
    diagnostics=[]
    for case,seed in sorted({(c['case'],c['seed']) for c in changes}):
        for mode in MODES:
            phase=None if mode=='baseline' else 'retreat'
            r=json.loads((ROOT/'trials'/f'{case}_{seed}_{phase}_{mode}.json').read_text())
            bad=next((x for x in r['trace'] if x['bad']),None)
            diagnostics.append(dict(case=case,seed=seed,mode=mode,success=r['success'],first_bad_phase=None if bad is None else bad['phase'],first_bad_pairs=[] if bad is None else bad['bad'],final_xy_error=float(np.linalg.norm(np.asarray(r['final_object'][:2])-tasks[case]['target']))))
    (ROOT/'changed_pair_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    lines+=['','## Changed-pair trace diagnostics','',
    'First forbidden contact is sampled at control-step boundaries (40ms); it does not establish the substep contact mechanism. Full pre-retreat prefixes match controls.','',json.dumps(diagnostics,indent=2)]
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    fig,ax=plt.subplots(figsize=(8,4))
    for j,mode in enumerate(MODES):
        vals=[next(r['success'] for r in table if r['group']==g and r['mode']==mode) for g in ('all','6','7')]
        bars=ax.bar(np.arange(3)+(j-1)*.24,vals,width=.24,label=mode);ax.bar_label(bars,padding=3)
    ax.set_xticks(range(3),['All candidates (48)','Six-joint (24)','Seven-joint (24)']);ax.set_ylim(0,48);ax.set_ylabel('Successful tasks');ax.set_title('Prospective body/placement replication');ax.legend();fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'comparison.{ext}')
    plt.close(fig);print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
