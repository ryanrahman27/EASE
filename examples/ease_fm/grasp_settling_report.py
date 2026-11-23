"""Audit and summarize the prespecified E68 comparison."""
import json
from pathlib import Path
from .benchmark import fingerprint
from .grasp_settling import ROOT


def run():
    rows=json.loads((ROOT/'records.json').read_text())
    protocol=json.loads((ROOT/'protocol.json').read_text())
    assert len(rows)==336 and len({(r['case'],r['method'],r['mode']) for r in rows})==336
    for p,h in {**protocol['inputs'],**protocol['checkpoints']}.items(): assert fingerprint(p)==h
    summary=[];paired=[]
    for method in dict.fromkeys(r['method'] for r in rows):
        for mode in ('baseline','fixed','contact'):
            rr=[r for r in rows if r['method']==method and r['mode']==mode]
            summary.append(dict(method=method,mode=mode,success=sum(r['success'] for r in rr),n=len(rr),
                six=sum(r['success'] for r in rr if r['dof']==6),seven=sum(r['success'] for r in rr if r['dof']==7),
                collisions=sum(r['collision'] for r in rr),steps=sum(r['steps'] for r in rr),
                added_dwell=sum(sum(e['steps'] for e in r['events']) for r in rr)))
            if mode!='baseline':
                baseline={r['case']:r for r in rows if r['method']==method and r['mode']=='baseline'}
                paired.append(dict(method=method,mode=mode,
                    gains=[r['case'] for r in rr if r['success'] and not baseline[r['case']]['success']],
                    losses=[r['case'] for r in rr if not r['success'] and baseline[r['case']]['success']]))
    for r in rows:
        full=json.loads((ROOT/'trials'/f"{r['case']}_{r['method']}_{r['mode']}.json").read_text())
        assert len(full['trace'])+32==r['steps']
        assert len(r['events'])==(0 if r['mode']=='baseline' else 2)
        for e in r['events']:
            assert 8<=e['steps']<=20
            if r['mode']=='fixed': assert e['steps']==20
            if e['steps']<20: assert e['stable_contact']
    pooled={mode:{k:sum(r[k] for r in summary if r['method']!='analytical' and r['mode']==mode) for k in ('success','six','seven','collisions','steps','added_dwell')} for mode in ('baseline','fixed','contact')}
    b,f,c=[pooled[m] for m in ('baseline','fixed','contact')]
    advance=c['success']>max(b['success'],f['success']) and c['six']>=b['six'] and c['collisions']<=b['collisions']
    legacy_checks=0
    for folder,offset in [('ease_pick_place_control_v2',0),('ease_pick_place_unseen_control_v1',12)]:
        old=json.loads((Path('outputs')/folder/'records.json').read_text())
        for r in old:
            method=r['method'].replace('neural','parent')
            current=json.loads((ROOT/'trials'/f"{r['case']+offset}_{method}_baseline.json").read_text())
            assert current['trace']==r['trace']
            assert current['success']==r['success']
            legacy_checks+=1
    assert legacy_checks==64
    audit=dict(trials=len(rows),hashes_unchanged=True,trace_lengths_verified=True,exact_historical_baselines=legacy_checks,advance_native=advance,pooled_learned=pooled,test_evaluated=False)
    for name,value in [('summary',summary),('paired',paired),('audit',audit)]:
        (ROOT/f'{name}.json').write_text(json.dumps(value,indent=2))
    lines=['# E68: frozen-estimator grasp settling','',
        'Development comparison on 12 familiar and four previously IK-screened unseen procedural placements. Seven frozen estimators; shared calibration and simulator state. All 336 outcomes retained. Contact sensing is additional simulator-provided information. No training or reserved final-test evaluation.','',
        '| Estimator | Mode | Success /16 | Six-joint /7 | Seven-joint /9 | Collisions | Total control steps |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['method']} | {r['mode']} | {r['success']} | {r['six']} | {r['seven']} | {r['collisions']} | {r['steps']} |")
    lines+=['',f"Pooled learned success: baseline {b['success']}/96, fixed-duration {f['success']}/96, contact-ended {c['success']}/96. Six-joint: {b['six']}, {f['six']}, {c['six']} /42; collisions: {b['collisions']}, {f['collisions']}, {c['collisions']}.", '',
        f"Contact-ended waiting has {sum(len(r['gains']) for r in paired if r['mode']=='contact' and r['method']!='analytical')} paired recoveries and {sum(len(r['losses']) for r in paired if r['mode']=='contact' and r['method']!='analytical')} paired regressions. Total learned control steps (including calibration) increase from {b['steps']} to {c['steps']} ({100*(c['steps']/b['steps']-1):.1f}%). E35 parents lose one aggregate success, while E66 candidates gain one; neither subgroup is a new independently selected deployment model.", '',
        f"Prespecified native advancement gate: {'PASS' if advance else 'FAIL'}. "+('Proceed to native retention before adoption.' if advance else 'Retain the original sequencer; no native expansion of this intervention.'),'',
        'The intervention holds the measured XYZ reference before lift and transport, while keeping upright tool-Z control, closed gripper, measured bias and online statistics updates. Fixed control waits 20 steps; contact control ends after 8 consecutive bilateral endpoint contacts or 20 steps. Neither rejects unstable grasps; original task outcomes remain the criterion. Maximum extra dwell is 40 control steps (1.6 seconds).','',
        'These are dependent seed/case repeats on a small, observed development fixture. Pooled counts are descriptive. Fixed waiting is a duration control, not exact per-trial time matching. This tests bounded settling, not active regrasp recovery or all contact-aware controllers. Native robot performance cannot be inferred from this procedural experiment.','',
        'Audits verify all checkpoint/input hashes, unique trial keys, trace lengths, dwell bounds, and exact trace reproduction of all 64 historical analytical/E35 baselines. All 95 regression tests pass. Per-case gains and losses are saved in paired.json; full traces remain in trials/.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    methods=list(dict.fromkeys(r['method'] for r in summary)); x=np.arange(len(methods))
    fig,ax=plt.subplots(figsize=(10,4))
    for i,mode in enumerate(('baseline','fixed','contact')):
        yy=[next(r['success'] for r in summary if r['method']==m and r['mode']==mode) for m in methods]
        bars=ax.bar(x+(i-1)*.25,yy,.25,label=mode)
        ax.bar_label(bars,padding=2,fontsize=8)
    ax.set(xticks=x,xticklabels=methods,ylim=(0,18),ylabel='Successful placements /16',title='E68: bounded settling on observed procedural manipulation cases')
    ax.legend(ncol=3,loc='lower left'); fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':run()
