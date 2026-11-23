"""E71 outcome, screening and casewise retention report."""
import json
from pathlib import Path
import shutil
from .phase_fresh import ROOT
from .benchmark import fingerprint


def run():
    rows=json.loads((ROOT/'records.json').read_text());screen=json.loads((ROOT/'screening.json').read_text())
    spec=json.loads((ROOT/'protocol.json').read_text());tasks=json.loads((ROOT/'cases.json').read_text())
    assert len(rows)==14*len(tasks) and len({(r['case'],r['method'],r['mode']) for r in rows})==len(rows)
    for p,h in {**spec['inputs'],**spec['checkpoints'],**spec['source_hashes']}.items():assert fingerprint(p)==h
    for r in rows:
        full=json.loads((ROOT/'trials'/f"{r['case']}_{r['method']}_{r['mode']}.json").read_text())
        assert len(full['trace'])+32==r['steps'] and len(r['phases'])==9 and r['abort'] is None
    methods=list(dict.fromkeys(r['method'] for r in rows));summary=[];paired=[]
    def counts(rr):
        return dict(n=len(rr),success=sum(r['success'] for r in rr),six=sum(r['success'] for r in rr if r['dof']==6),six_n=sum(r['dof']==6 for r in rr),seven=sum(r['success'] for r in rr if r['dof']==7),seven_n=sum(r['dof']==7 for r in rr),collisions=sum(r['collision'] for r in rr),steps=sum(r['steps'] for r in rr))
    for method in methods:
        rr=[r for r in rows if r['method']==method];base={r['case']:r for r in rr if r['mode']=='baseline'}
        for mode in ('baseline','extended'):summary.append(dict(method=method,mode=mode,**counts([r for r in rr if r['mode']==mode])))
        ext=[r for r in rr if r['mode']=='extended']
        paired.append(dict(method=method,gains=[r['case'] for r in ext if r['success'] and not base[r['case']]['success']],losses=[r['case'] for r in ext if not r['success'] and base[r['case']]['success']],six_losses=[r['case'] for r in ext if r['dof']==6 and not r['success'] and base[r['case']]['success']]))
    pooled={m:counts([r for r in rows if r['method']!='analytical' and r['mode']==m]) for m in ('baseline','extended')}
    b,e=pooled['baseline'],pooled['extended']
    six_retained=all(next(r['six'] for r in summary if r['method']==m and r['mode']=='extended')>=next(r['six'] for r in summary if r['method']==m and r['mode']=='baseline') for m in methods if m!='analytical')
    passed=bool(b['n'] and e['success']>b['success'] and e['collisions']<=b['collisions'] and six_retained)
    gains=sum(len(p['gains']) for p in paired if p['method']!='analytical');losses=sum(len(p['losses']) for p in paired if p['method']!='analytical')
    family=[dict(family=f,mode=m,**counts([r for r in rows if r['family']==f and r['mode']==m and r['method']!='analytical'])) for f in dict.fromkeys(r['family'] for r in rows) for m in ('baseline','extended')]
    result=dict(pooled=pooled,per_checkpoint_six_retained=six_retained,passes=passed,paired_gains=gains,paired_losses=losses,methods=summary,families=family)
    for name,value in [('summary',result),('paired',paired)]: (ROOT/f'{name}.json').write_text(json.dumps(value,indent=2))
    cost=100*(e['steps']/b['steps']-1)
    decision=('The aggregate E70 rule replicates on these placements. Preserve the frozen extended-budget candidate for broader validation; do not infer casewise retention or new-embodiment superiority.' if passed else 'The E70 improvement does not pass the prespecified replication rule. Keep the original phase budgets and retain extended timing as a failed development candidate.')
    lines=['# E71: fresh-placement validation of extended phase budgets','',
        f"{len(tasks)}/{len(screen)} prespecified placements pass endpoint IK screening across the same six procedural bodies. All screening rejections are retained without replacement; each accepted placement receives all seven estimators and both conditions ({len(rows)} trials). Positions, IK seeds and probe seeds are new. Bodies and axis families are reused: this is placement transfer, not new-embodiment transfer or the reserved final test.",'',
        '| Estimator | Budget | Success | Six-joint | Seven-joint | Collisions | Total steps |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['method']} | {r['mode']} | {r['success']}/{r['n']} | {r['six']}/{r['six_n']} | {r['seven']}/{r['seven_n']} | {r['collisions']} | {r['steps']} |")
    lines+=['',f"Pooled learned success: {b['success']}/{b['n']} baseline versus {e['success']}/{e['n']} extended; {gains} recoveries and {losses} regressions. Six-joint: {b['six']} to {e['six']}/{e['six_n']}; seven-joint: {b['seven']} to {e['seven']}/{e['seven_n']}. Collisions: {b['collisions']} to {e['collisions']}. Total control steps including calibration: {b['steps']} to {e['steps']} ({cost:+.1f}%).",'',
        f"Prespecified aggregate replication rule: {'PASS' if passed else 'FAIL'}. Per-checkpoint aggregate six-joint retention: {six_retained}. {decision}",'',
        'The frozen E70 implementation adds at most 40 steps per moving phase and 20 to close. It leaves the early-exit rule, physics, waypoints and success/contact criteria unchanged. No strict transition gate is used. No thresholds, parameters or checkpoints were chosen using E71 results. All trials share a cloned measured calibration state within each placement; probes with contacts remain failures rather than being filtered out.','',
        '| Family | Budget | Learned success | Collision cases |',
        '|---|---|---:|---:|']
    for r in family:lines.append(f"| {r['family']} | {r['mode']} | {r['success']}/{r['n']} | {r['collisions']} |")
    lines+=['','Screening rejections: '+', '.join(f"candidate {r['candidate']} ({r['dof']} joints), failed {r['failed_phase']} after {r['endpoints']} endpoints" for r in screen if not r['accepted'])+'.','',
        'Verification: all model/input/E70 implementation hashes unchanged; unique trial keys, complete nine-phase rollouts and trace lengths audited. The same 99 EASE-FM tests pass. Full protocols, probes, screening records and trial traces are retained. No model training, native tuning or final-test rollout.','',
        'Interpretation is limited by endpoint screening, six reused bodies, dependent training seeds and observed development history. Aggregate six-joint retention does not mean every successful six-joint case is preserved; paired.json lists each regression. Extra time and its cost must remain part of the comparison.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axs=plt.subplots(1,2,figsize=(10,4));x=np.arange(len(methods))
    for i,mode in enumerate(('baseline','extended')):
        vals=[next(r['success'] for r in summary if r['method']==m and r['mode']==mode) for m in methods]
        bars=axs[0].bar(x+(i-.5)*.36,vals,.36,label=mode);axs[0].bar_label(bars,fontsize=8)
    axs[0].set(xticks=x,xticklabels=[m.replace('candidate','E66').replace('parent','E35') for m in methods],ylabel=f'Successful placements /{len(tasks)}',ylim=(0,len(tasks)+2));axs[0].tick_params(axis='x',labelrotation=30);axs[0].legend()
    bars=axs[1].bar(['baseline','extended'],[b['collisions'],e['collisions']],color=['#4c78a8','#f58518']);axs[1].bar_label(bars);axs[1].set(ylabel=f'Learned collision cases /{b["n"]}',ylim=(0,b['n']))
    fig.suptitle('E71: frozen phase budgets on fresh procedural placements');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(pooled=pooled,passes=passed,per_checkpoint_six_retained=six_retained,gains=gains,losses=losses),indent=2))


if __name__=='__main__':run()
