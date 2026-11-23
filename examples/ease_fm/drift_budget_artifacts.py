"""E121 reports, independent ledger checks and reproducible media."""
import json
import shutil
from pathlib import Path
import numpy as np
from .drift_budget import ROOT,OLD,setup,evaluate,fingerprint,BUDGET


def build():
    p=Path(__file__).with_name('bounded_history_videos.py')
    s=p.read_text().replace('bounded_history_correction import ROOT','drift_budget import ROOT').replace('E120','E121').replace('Bounded correction / Retention','Cumulative correction budget / Retention')
    needle='    selection=list(dict.fromkeys(selection));manifest='
    extra="    for d in json.loads((ROOT/'diagnosis/summary.json').read_text()):\n        for family in ('reference','unbounded','candidate'):selection.append((d['case'],f\"{family}_{d['seed']}\",'frozen_bias'))\n"
    assert needle in s;s=s.replace(needle,extra+needle)
    s=s.replace('Outcome-selected diagnostic examples.','Includes all four E120 regressions under E118, E120 and E121. Outcome-selected diagnostic examples.')
    Path(__file__).with_name('drift_budget_videos.py').write_text(s)


def report():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==240
    tasks=json.loads((ROOT/'cases.json').read_text());lookup={(r['case'],r['method']):r for r in rows};aggregate=[];changes=[]
    for method in ('reference','unbounded','candidate','analytical'):
        for split in ('all','fitting','held'):
            for dof in ('all',6,7):
                rr=[r for r in rows if r['method'].startswith(method) and (split=='all' or tasks[r['case']]['validation']==(split=='held')) and (dof=='all' or r['dof']==dof)]
                aggregate.append(dict(method=method,split=split,dof=str(dof),n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr)))
    for baseline in ('reference','unbounded'):
        for seed in range(3):
            for case in range(24):
                a=lookup[case,f'{baseline}_{seed}'];b=lookup[case,f'candidate_{seed}']
                if a['success']!=b['success']:changes.append(dict(baseline=baseline,case=case,seed=seed,gain=b['success'],split='held' if tasks[case]['validation'] else 'fitting',dof=b['dof']))
    diagnostic=json.loads((ROOT/'diagnosis/summary.json').read_text())
    recovery=[dict(case=d['case'],seed=d['seed'],recovered=lookup[d['case'],f"candidate_{d['seed']}"]['success']) for d in diagnostic]
    oldsummary=json.loads((OLD/'summary.json').read_text());gains=[dict(case=c['case'],seed=p['seed'],retained=lookup[c['case'],f"candidate_{p['seed']}"]['success']) for p in oldsummary['paired'] for c in p['changes'] if c['gain']]
    gate=all(r['recovered'] for r in recovery) and all(r['retained'] for r in gains) and not any(not r['gain'] for r in changes)
    summary=dict(aggregate=aggregate,paired_changes=changes,regressions=recovery,prior_gains=gains,retention_gate_passed=gate)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    totals=[r for r in aggregate if r['split']==r['dof']=='all'];headline='; '.join(f"{r['method']}: {r['success']}/{r['n']}, {r['collisions']} collisions" for r in totals if r['method']!='analytical')
    lines=['# E121: Accumulated command residual and episode budget','',headline+'.', '',
        'reference = frozen E118; unbounded = E120 (per-step bound only); candidate = E121 (per-step plus signed cumulative budget). The unbounded label is inherited tooling terminology: E120 is bounded per step but has no cumulative budget.', '',
        'Both neural estimators remain frozen. Preserve the E120 25% action residual and 0.001-rad/joint/step cap; additionally keep the signed sum of applied residuals within +/-0.004 rad/joint for the entire task. No resets at phase changes. Opposite residuals can unwind the sum. Calibration and the explicit probe are unchanged and do not spend the budget. This does not bound joint-state drift, path length, contact or physical safety. No physics selector, analytical fallback, privileged Jacobian or additional fitting.', '',
        'All24 placements on four training bodies, repeated across three checkpoints. Sixteen fitting placements and eight held placements; these are development tasks, not untouched tests or unseen bodies. One fixed budget was tested, without a sweep or seed selection.', '',
        '## Drift diagnosis','',
        'The four successful E118 trajectories were replayed exactly. State/action comparisons align phase and within-phase step. If E118 already left a phase, a paired state is unavailable; we do not substitute a different phase. Cumulative residual includes the action at the first collision; recorded q/XYZ drift is immediately before that action. These observations locate drift but do not establish that command accumulation alone caused the collision.', '',
        '| Case / seed | First collision | Signed residual max (mrad) | Pre-action XYZ drift (mm) | Recovered |','|---|---|---:|---:|---|']
    for d,r in zip(diagnostic,recovery):
        c=d['first_collision'];xyz=c.get('xyz_drift_mm');lines.append(f"| {d['case']} / {d['seed']} | {c['phase']} step {c['phase_step']} | {1000*c['signed_residual_max_rad']:.2f} | {xyz if xyz is not None else 'unavailable'} | {r['recovered']} |")
    lines+=['','## All episode results','','| Method | Split | DOF | Success | Collision episodes |','|---|---|---|---:|---:|']
    for r in aggregate:lines.append(f"| {r['method']} | {r['split']} | {r['dof']} | {r['success']}/{r['n']} | {r['collisions']} |")
    lines+=['',f"Recovered {sum(r['recovered'] for r in recovery)}/4 regressions; retained {sum(r['retained'] for r in gains)}/6 E120 gains. Strict retention gate {'PASSED' if gate else 'FAILED'}.",'','## Every paired change','','```json',json.dumps(changes,indent=2),'```','',
        'All weights, trajectories, command/history logs, signed-budget ledgers and outcomes are saved. Do not promote a higher aggregate score if it loses previously successful cases. A failure of this intervention would show that bounding signed residual alone is insufficient; it would not rule out state/history feedback amplification or phase-timing effects.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    for i,(m,label) in enumerate((('reference','Frozen E118'),('unbounded','Per-step cap E120'),('candidate','Episode budget E121'))):
        rr=[next(r for r in aggregate if r['method']==m and r['split']==sp and r['dof']=='all') for sp in ('fitting','held')];x=np.arange(2)+(i-1)*.23;ax.bar(x,[100*r['success']/r['n'] for r in rr],.21,label=label)
        for xx,r in zip(x,rr):ax.text(xx,100*r['success']/r['n']+2,f"{r['success']}/{r['n']}",ha='center')
    ax.set_xticks([0,1],['Fitting placements','Held placements']);ax.set_ylim(0,110);ax.set_ylabel('Task success (%)');ax.legend(ncol=3,loc='upper center');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(11,11))
    for row,d in enumerate(diagnostic):
        case,seed=d['case'],d['seed'];baseline=json.loads((ROOT/'diagnosis'/f'{case}_{seed}_baseline_actions.json').read_text());index={};counts={}
        for a in baseline:
            key=a['phase'];k=counts.get(key,0);counts[key]=k+1;index[key,k]=a
        logs=json.loads((ROOT/'actions'/f'{case}_candidate_{seed}_frozen_bias.json').read_text());counts={};cum=np.zeros(7);series=[]
        for i,a in enumerate(logs):
            key=a['phase'];k=counts.get(key,0);counts[key]=k+1
            if a['active']:cum+=np.array(a['nominal_action'])-a['base_action']
            b=index.get((key,k));series.append(dict(step=i,phase=key,phase_step=k,signed_residual_max_rad=float(np.max(np.abs(cum))),xyz_drift_mm=None if b is None else float(np.linalg.norm(np.array(a['position_before'])-b['position_before'])*1000)))
        (ROOT/'diagnosis'/f'{case}_{seed}_budget_drift.json').write_text(json.dumps(series,indent=2))
        old=json.loads((ROOT/'diagnosis'/f'{case}_{seed}_drift.json').read_text())
        for label,ss in (('E120',old),('E121',series)):
            axes[row,0].plot([r['step'] for r in ss],[1000*r['signed_residual_max_rad'] for r in ss],label=label)
            axes[row,1].plot([r['step'] for r in ss],[r.get('xyz_drift_mm') if r.get('xyz_drift_mm') is not None else np.nan for r in ss],label=label)
        axes[row,0].axhline(4,color='black',ls=':',lw=1);axes[row,0].set_ylabel(f'Case {case}, seed {seed}\nResidual max (mrad)');axes[row,1].set_ylabel('Matched-phase XYZ drift (mm)')
        for ax in axes[row]:ax.axvline(d['first_collision']['step'],color='gray',ls='--',lw=1);ax.set_xlabel('Own control step');ax.legend()
    fig.suptitle('Command budget and physical drift are different quantities\nDashed line: E120 first collision; curves align to baseline by phase and phase-step');fig.tight_layout();fig.savefig(ROOT/'drift.png',dpi=150);plt.close(fig)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');s=p.read_text(encoding='utf-8');marker='## E121: cumulative correction budget';assert marker not in s;s=s.replace('Revision 108 · 14 November 2026','Revision 109 · 14 November 2026',1);s+='\n'+marker+'\n\n'+headline+f". Recovered {sum(r['recovered'] for r in recovery)}/4 regressions; retained {sum(r['retained'] for r in gains)}/6 gains. Gate {gate}.\n\nFour exact baseline replays, phase-aligned drift diagnosis and72 fixed-budget evaluations. No unseen bodies or physics selector. See `outputs/ease_drift_budget_v1/report.md`.\n";p.write_text(s,encoding='utf-8')
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(headline);print(json.dumps(dict(recovery=recovery,gains=gains,changes=changes,gate=gate),indent=2))


def verify():
    tasks,models,hashes=setup();rows=json.loads((ROOT/'records.json').read_text());checks=[]
    for seed in range(3):
        for outcome in (True,False):
            r=next((r for r in rows if r['method']==f'candidate_{seed}' and r['success']==outcome),None)
            if r is None:continue
            case=r['case'];stem=f'{case}_candidate_{seed}_frozen_bias';old=json.loads((ROOT/'trials'/f'{stem}.json').read_text());states=np.load(ROOT/'states'/f'{stem}.npz')['states'].copy()
            new,_,_=evaluate(case,tasks[case],f'candidate_{seed}',models[f'reference_{seed}'],models[f'correction_{seed}']);assert new==old;np.testing.assert_array_equal(states,np.load(ROOT/'states'/f'{stem}.npz')['states']);checks.append(stem);print('exact',stem,flush=True)
    steps=0;peak=0.
    for p in (ROOT/'actions').glob('*_candidate_*_frozen_bias.json'):
        logs=json.loads(p.read_text());cumulative=np.zeros(len(logs[0]['base_action']));ledger=json.loads(p.with_name(p.name.replace('_frozen_bias','_budget')).read_text());j=0
        for r in logs:
            if not r['active']:continue
            a=np.array(r['base_action']);b=np.array(r['correction_action']);raw=np.clip(a+np.clip(.25*(b-a),-.001,.001),-.04,.04)-a
            expected=np.clip(a+np.clip(raw,-BUDGET-cumulative,BUDGET-cumulative),-.04,.04)
            np.testing.assert_array_equal(expected,r['nominal_action']);np.testing.assert_array_equal(cumulative,ledger[j]['before']);cumulative+=expected-a;np.testing.assert_array_equal(cumulative,ledger[j]['after']);j+=1
            assert np.max(np.abs(cumulative))<=BUDGET+1e-12;assert np.max(np.abs(expected-a))<=.001+1e-12;steps+=1;peak=max(peak,float(np.max(np.abs(cumulative))))
        assert j==len(ledger)
    for p,h in hashes.items():assert fingerprint(p)==h
    (ROOT/'verification.json').write_text(json.dumps(dict(exact_replays=checks,action_steps_checked=steps,max_cumulative_rad=peak,unchanged_weights=True),indent=2))


def finish():
    spec=json.loads((ROOT/'protocol.json').read_text());v=json.loads((ROOT/'verification.json').read_text());s=json.loads((ROOT/'summary.json').read_text());m=json.loads((ROOT/'videos/manifest.json').read_text())
    for p,h in {**spec['checkpoints'],**spec['inputs']}.items():assert fingerprint(p)==h
    for r in m['videos']:assert fingerprint(ROOT/'videos'/r['file'])==r['sha256']
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete',new_episodes=72,baseline_replays=4,verification_replays=len(v['exact_replays']),videos=len(m['videos']),retention_gate_passed=s['retention_gate_passed'],unseen_evaluated=False),indent=2))
    p=ROOT/'report.md';p.write_text(p.read_text()+f"\nVerification: {len(v['exact_replays'])} exact candidate replays; {v['action_steps_checked']} action steps independently checked against both limits; {len(m['videos'])} trace-verified videos, including all four regression comparisons.\n\nThe episode budget recovered three regressions but retained only one of six E120 gains. Versus E120 it gained four successes and lost ten, reducing52/72 to46/72 and raising collision episodes18 to25. Reject this intervention. Preserve E118 as the frozen reference and E120 as the better correction candidate. The residual ledger remains bounded while substantial physical drift persists; see drift.png. A global signed-command budget does not solve state/history feedback or selectively retain useful corrections.\n")
    p=ROOT/'videos/index.html';html=p.read_text();html=html.replace('</h1>','</h1><p>reference = E118 frozen baseline; unbounded = E120 per-step cap (no episode budget); candidate = E121 cumulative budget.</p><p><a href="../drift.png">Accumulated residual and physical drift plots</a></p>',1);p.write_text(html)
    (ROOT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2));print((ROOT/'status.json').read_text())


if __name__=='__main__':
    import sys
    {'build':build,'report':report,'verify':verify,'finish':finish}[sys.argv[1]]()
