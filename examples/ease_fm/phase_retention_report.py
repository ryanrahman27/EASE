"""E70 paired retention audit and report."""
import json
from collections import Counter
from .phase_retention import ROOT


def run():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==336
    summaries=[];pairs=[]
    methods=list(dict.fromkeys(r['method'] for r in rows))
    for method in methods:
        baseline={r['case']:r for r in rows if r['method']==method and r['mode']=='baseline'}
        for mode in ('baseline','extended','gated'):
            rr=[r for r in rows if r['method']==method and r['mode']==mode]
            summaries.append(dict(method=method,mode=mode,success=sum(r['success'] for r in rr),six=sum(r['success'] for r in rr if r['dof']==6),seven=sum(r['success'] for r in rr if r['dof']==7),collisions=sum(r['collision'] for r in rr),aborts=sum(r['abort'] is not None for r in rr),steps=sum(r['steps'] for r in rr)))
            if mode!='baseline':pairs.append(dict(method=method,mode=mode,gains=[r['case'] for r in rr if r['success'] and not baseline[r['case']]['success']],losses=[r['case'] for r in rr if not r['success'] and baseline[r['case']]['success']]))
    pooled={mode:{key:sum(r[key] for r in summaries if r['method']!='analytical' and r['mode']==mode) for key in ('success','six','seven','collisions','aborts','steps')} for mode in ('baseline','extended','gated')}
    decisions={}
    for mode in ('extended','gated'):
        six_retained=all(next(r['six'] for r in summaries if r['method']==m and r['mode']==mode)>=next(r['six'] for r in summaries if r['method']==m and r['mode']=='baseline') for m in methods if m!='analytical')
        decisions[mode]=dict(per_checkpoint_six_retained=six_retained,passes=pooled[mode]['success']>pooled['baseline']['success'] and pooled[mode]['collisions']<=pooled['baseline']['collisions'] and six_retained)
    aborts=dict(Counter(r['abort'] for r in rows if r['method']!='analytical' and r['mode']=='gated' and r['abort']))
    output=dict(pooled=pooled,decisions=decisions,aborts=aborts,methods=summaries)
    (ROOT/'summary.json').write_text(json.dumps(output,indent=2));(ROOT/'paired.json').write_text(json.dumps(pairs,indent=2))
    lines=['# E70: bounded phase-transition retention','',
        'All 16 observed procedural placements, seven frozen estimators and three conditions: 336 trials. Calibration, weights, dynamics, task waypoints and forbidden-contact definitions are unchanged. This experiment isolates phase budget and transition decisions; it does not introduce a collision-clearance planner.','',
        '| Estimator | Mode | Success /16 | Six /7 | Seven /9 | Collisions | Aborts | Steps including calibration |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summaries:lines.append(f"| {r['method']} | {r['mode']} | {r['success']} | {r['six']} | {r['seven']} | {r['collisions']} | {r['aborts']} | {r['steps']} |")
    lines+=['','Extended budgets add at most 40 steps to each moving phase and 20 to close (260 maximum additional steps). Original early pose exit remains active. Gated uses identical budgets but aborts at phase end on accumulated collision, missing five final in-tolerance moving-phase samples, or missing eight final in-tolerance bilateral close samples. Aborts always count as failures. This gate does not stop a collision during a phase, nor avoid contact geometrically.','',
        '| Learned models pooled | Success /96 | Six /42 | Seven /54 | Collisions | Aborts | Steps |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode,r in pooled.items():lines.append(f"| {mode} | {r['success']} | {r['six']} | {r['seven']} | {r['collisions']} | {r['aborts']} | {r['steps']} |")
    lines+=['','Prespecified retention requires improved pooled success, no pooled collision increase, and no six-joint success loss in any checkpoint.']
    for mode,d in decisions.items():
        pp=[r for r in pairs if r['mode']==mode and r['method']!='analytical']
        lines.append(f"- {mode}: {'PASS' if d['passes'] else 'FAIL'}; {sum(len(r['gains']) for r in pp)} paired recoveries and {sum(len(r['losses']) for r in pp)} regressions; per-checkpoint six-joint retention={d['per_checkpoint_six_retained']}.")
    lines+=['','Retention refers to per-checkpoint aggregate six-joint counts, not preservation of every successful case. Extended candidate_0 loses familiar six-joint case 4 while recovering unseen six-joint case 12. Five extended recoveries and three losses therefore remain a mixed result despite passing this aggregate rule. Extended control uses 33,250 versus 30,107 total learned control steps, a 10.4% increase.','',
        'Learned gated aborts: '+', '.join(f'{k}={v}' for k,v in aborts.items())+'.','',
        'Verification: all 112 baseline traces and calibration probes reproduce E68 exactly. Each of 112 gated traces exactly matches the corresponding extended trajectory up to its stopping point. Model/input hashes are unchanged; 99 EASE-FM tests pass. All failures and stopped trajectories are retained. No training, native evaluation or reserved final-test evaluation.','',
        'These dependent repeated seeds and small, IK-screened development placements do not establish population-level transfer. Fewer collision cases can result from ending a trial earlier; task success and paired regressions remain essential. A retention pass would justify further validation, not immediate deployment or a paper-ready claim.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axs=plt.subplots(1,2,figsize=(10,4));modes=list(pooled);x=np.arange(3)
    for ax,key,label in zip(axs,('success','collisions'),('Successful tasks /96','Trials with forbidden contact /96')):
        bars=ax.bar(x,[pooled[m][key] for m in modes],color=['#4c78a8','#f58518','#54a24b']);ax.bar_label(bars,padding=3)
        ax.set(xticks=x,xticklabels=modes,ylabel=label,ylim=(0,100))
    fig.suptitle('E70: frozen learned estimators, observed procedural placements');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(pooled=pooled,decisions=decisions,aborts=aborts),indent=2))


if __name__=='__main__':run()
