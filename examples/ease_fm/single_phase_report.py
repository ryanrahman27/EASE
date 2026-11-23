import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .single_phase import ROOT,MOTION,fingerprint


def run():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==324
    spec=json.loads((ROOT/'protocol.json').read_text())
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    tables=[];changes=[];gates={}
    for phase in MOTION:
        selected=[r for r in rows if r['mode']=='baseline' or r['affected_phase']==phase]
        for group in ('all','train','validation','6','7','seed0','seed1','seed2'):
            for mode in ('baseline','budget','gentle'):
                rr=[r for r in selected if r['mode']==mode and (group=='all' or group==str(r['dof']) or group==f"seed{r['seed']}" or group==('validation' if r['validation'] else 'train'))]
                tables.append(dict(phase=phase,group=group,mode=mode,n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),steps=sum(r['steps'] for r in rr)))
        gates[phase]={}
        for control in ('baseline','budget'):
            for r in selected:
                if r['mode']!='gentle':continue
                ref=next(x for x in selected if x['case']==r['case'] and x['seed']==r['seed'] and x['mode']==control)
                if r['success']!=ref['success']:changes.append(dict(phase=phase,case=r['case'],seed=r['seed'],dof=r['dof'],validation=r['validation'],control=control,change='gain' if r['success'] else 'loss',collision=r['collision']))
            checks={}
            for split,val in [('train',False),('validation',True)]:
                new=[r for r in selected if r['mode']=='gentle' and r['validation']==val];old=[r for r in selected if r['mode']==control and r['validation']==val]
                checks[split]=dict(retains_all_successes=not any(x['phase']==phase and x['control']==control and x['validation']==val and x['change']=='loss' for x in changes),collisions_nonincreasing=sum(r['collision'] for r in new)<=sum(r['collision'] for r in old))
                if split=='train':checks[split].update(success_improves=sum(r['success'] for r in new)>sum(r['success'] for r in old),per_seed_nondecreasing=all(sum(r['success'] for r in new if r['seed']==s)>=sum(r['success'] for r in old if r['seed']==s) for s in range(3)))
            gates[phase][control]=checks
        gates[phase]['passes']=all(all(v.values()) for g in gates[phase].values() for v in g.values())
    passing=[p for p in MOTION if gates[p]['passes']]
    (ROOT/'summary.json').write_text(json.dumps(dict(table=tables,changes=changes,gates=gates,passing=passing),indent=2))
    lines=['# E104: isolated phase ramps with matched time controls','',
        '324 complete rollouts:36 training-body task/seed pairs x one unchanged baseline plus four phases x two controls. The baseline is executed once per pair and reused in each phase comparison. All36 baseline results match E103 exactly. Every intervention trace matches baseline before its selected phase. E99 retained weights, original E66 prefix/probe, four training bodies, task criteria and privileged geometry filter stay fixed. No route planner, weight training or contact feedback is added.',
        '', 'For each phase separately, budget-only allows20 extra maximum control steps; gentle uses the same allowance and the fixed20-step E103 cubic ramp of action minus measured bias. All other phase budgets and commands are unchanged. Original commands resume exactly from step20 onward. The24 training /12 held-placement pairs are unchanged. All four phases are evaluated and reported without ramp-duration or seed selection.',
        '', '## Outcomes','','| Phase | Group | Mode | Success | Collisions | Steps |','|---|---|---|---:|---:|---:|']
    for r in tables:lines.append(f"| {r['phase']} | {r['group']} | {r['mode']} | {r['success']}/{r['n']} | {r['collisions']} | {r['steps']} |")
    lines+=['','## All paired gains and losses','','| Phase | Reference | Case | Seed | DOF | Held placement | Change | Collision |','|---|---|---:|---:|---:|---|---|---|']
    for r in changes:lines.append(f"| {r['phase']} | {r['control']} | {r['case']} | {r['seed']} | {r['dof']} | {r['validation']} | {r['change']} | {r['collision']} |")
    lines+=['','## Prespecified gates','', 'Each phase must strictly improve training success over both controls, preserve every success and each training seed count, and not increase collisions; held placements must retain every success with nonincreasing collisions. Passing phases only are eligible for transfer.']
    for phase in MOTION:lines+=['',f"- {phase}: **{'PASS' if gates[phase]['passes'] else 'FAIL'}**. {gates[phase]}"]
    lines+=['','## Verification and scope','', '131 existing tests pass. All36 full baseline replays and288 pre-intervention phase prefixes verify; all324 prefixes through closing match the original reference. Input/checkpoint hashes are unchanged. All trial traces and40ms contact/object diagnostics are saved; these do not resolve every contact transient. Videos are full-result replay verified with frame counts and hashes. Figure phase_comparison.png / .pdf; gallery videos/index.html. This is a training-body development ablation, not a new learned policy or independent generalization result. No final test or inference-time weight adaptation.']
    if (ROOT/'decision.txt').exists():lines+=['','## Decision','',(ROOT/'decision.txt').read_text().strip()]
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');x=np.arange(4)
    for ax,group,limit in zip(axes,('train','validation'),(27,15)):
        for i,mode in enumerate(('baseline','budget','gentle')):
            values=[next(r['success'] for r in tables if r['phase']==p and r['group']==group and r['mode']==mode) for p in MOTION];bars=ax.bar(x+(i-1)*.25,values,.25,label=mode);ax.bar_label(bars,padding=2)
        ax.set(xticks=x,xticklabels=MOTION,ylim=(0,limit),title=group,ylabel='Successes');ax.legend(fontsize=8)
    fig.savefig(ROOT/'phase_comparison.png',dpi=160);fig.savefig(ROOT/'phase_comparison.pdf');plt.close(fig)
    print(json.dumps(dict(main=[r for r in tables if r['group'] in ('all','train','validation')],passing=passing),indent=2))


if __name__=='__main__':run()
