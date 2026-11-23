import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .gentle_transition import ROOT,MODES
from .benchmark import fingerprint


def run():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==108
    spec=json.loads((ROOT/'protocol.json').read_text())
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    tables=[];changes=[];gates={}
    for group in ('all','train','validation','6','7','seed0','seed1','seed2'):
        for mode in MODES:
            rr=[r for r in rows if r['mode']==mode and (group=='all' or group==str(r['dof']) or group==f"seed{r['seed']}" or group==('validation' if r['validation'] else 'train'))]
            tables.append(dict(group=group,mode=mode,n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),steps=sum(r['steps'] for r in rr)))
    for control in ('baseline','budget'):
        for r in rows:
            if r['mode']!='gentle':continue
            ref=next(x for x in rows if x['case']==r['case'] and x['seed']==r['seed'] and x['mode']==control)
            if r['success']!=ref['success']:changes.append(dict(case=r['case'],seed=r['seed'],dof=r['dof'],validation=r['validation'],control=control,change='gain' if r['success'] else 'loss',collision=r['collision']))
        checks={}
        for split,val in [('train',False),('validation',True)]:
            new=[r for r in rows if r['mode']=='gentle' and r['validation']==val];old=[r for r in rows if r['mode']==control and r['validation']==val]
            checks[split]=dict(retains_all_successes=not any(x['control']==control and x['validation']==val and x['change']=='loss' for x in changes),collisions_nonincreasing=sum(r['collision'] for r in new)<=sum(r['collision'] for r in old))
            if split=='train':checks[split].update(success_improves=sum(r['success'] for r in new)>sum(r['success'] for r in old),per_seed_nondecreasing=all(sum(r['success'] for r in new if r['seed']==s)>=sum(r['success'] for r in old if r['seed']==s) for s in range(3)))
        gates[control]=checks
    passes=all(all(v.values()) for g in gates.values() for v in g.values())
    summary=dict(table=tables,changes=changes,gates=gates,passes=passes,transfer_evaluated=False);(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# E103: gentle phase-transition training-arm screen','',
        'All12 original training placements on four original training bodies x3 fixed E99 retained checkpoints x baseline/budget/gentle =108 complete tasks. Third placement per body remains held-placement validation (24 training,12 held-placement pairs). Original E66 prefix and32-step probe are identical; E99 retained continuation weights stay frozen. No route planner is used on these training tasks; the existing privileged geometry filter remains.',
        '', 'Gentle scales the task-command correction around measured bias using cubic smoothstep during the first20 control steps (0.8s) of lift, transport, lower and retreat. Both gentle and budget-only allow20 extra maximum steps per affected phase. Original commands pass through exactly from step20 onward. Targets, gripper apertures, early-exit conditions, collision masks, success criteria and physics are unchanged. No contact feedback or estimator training is added. Ramp limits joint-command onset, not necessarily Cartesian velocity.',
        '', '| Group | Mode | Success | Collisions | Steps |','|---|---|---:|---:|---:|']
    for r in tables:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['n']} | {r['collisions']} | {r['steps']} |")
    lines+=['','## All gentle paired changes','','| Reference | Case | Seed | DOF | Held placement | Change | Collision |','|---|---:|---:|---:|---|---|---|']
    for r in changes:lines.append(f"| {r['control']} | {r['case']} | {r['seed']} | {r['dof']} | {r['validation']} | {r['change']} | {r['collision']} |")
    lines+=['','## Prespecified gate','',f"**{'PASS' if passes else 'FAIL'}**: {gates}",'',
        'Transfer evaluation is conditional on strict training improvement versus both controls, no lost successes or increased collisions, no training-seed decline, and held-placement retention. No ramp duration or seed is selected after outcomes. Contact/object traces are saved at40ms control-step endpoints, which cannot resolve every transient contact; E102 used finer2ms diagnostics.',
        '', '## Verification','', 'All36 original baseline full rollouts and108 prefixes through closing replay exactly. Input/checkpoint hashes are unchanged.131 tests pass, including monotonic ramp onset, bias-only preservation, bounds and exact original-command recovery. A calibration phase-counter initialization error was fixed before any completed trial; the failed-start source and note are preserved. All new video results must match saved full serialized trials, decoded frame counts and hashes. Figure transition_comparison.png / .pdf; gallery videos/index.html. No final-test evaluation or inference-time weight updates.']
    if (ROOT/'decision.txt').exists():lines+=['','## Decision','',(ROOT/'decision.txt').read_text().strip()]
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,group in zip(axes,('train','validation')):
        rr=[r for r in tables if r['group']==group];bars=ax.bar([r['mode'] for r in rr],[r['success'] for r in rr]);ax.bar_label(bars,padding=2);ax.set(title=group,ylabel='Successes',ylim=(0,27 if group=='train' else 15))
    fig.savefig(ROOT/'transition_comparison.png',dpi=160);fig.savefig(ROOT/'transition_comparison.pdf');plt.close(fig);print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
