"""E115 complete-factorial report, distinguishing new and reused trials."""
import json,shutil
from pathlib import Path
import numpy as np
from .frozen_bias_core import ROOT

def run():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==256
    modes=('online','frozen_response','frozen_all','frozen_bias');summary=[];pairs=[];bybody=[]
    lookup={(r['case'],r['method'],r['mode']):r for r in rows};assert len(lookup)==256
    for method in ('learned','analytical'):
        for mode in modes:
            for dof in ('all',6,7):
                group=[r for r in rows if r['method'].startswith(method) and r['mode']==mode and (dof=='all' or r['dof']==dof)]
                summary.append(dict(method=method,mode=mode,dof=str(dof),n=len(group),success=sum(r['success'] for r in group),collision=sum(r['collision'] for r in group),grasped=sum(r['grasped'] for r in group),lifted=sum(r['lifted'] for r in group),mean_steps=float(np.mean([r['steps'] for r in group]))))
    for method in ('analytical','learned_0','learned_1','learned_2'):
        for control in ('online','frozen_response','frozen_all','analytical_frozen_bias'):
            if method=='analytical' and control=='analytical_frozen_bias':continue
            for dof in ('all',6,7):
                changes=[];n=0
                for case in range(16):
                    a=lookup[case,method,'frozen_bias'];b=lookup[case,'analytical','frozen_bias'] if control=='analytical_frozen_bias' else lookup[case,method,control]
                    if dof!='all' and a['dof']!=dof:continue
                    n+=1
                    if a['success']!=b['success']:changes.append(dict(case=case,body=a['body'],dof=a['dof'],gain=bool(a['success'])))
                pairs.append(dict(method=method,control=control,dof=str(dof),n=n,gains=sum(c['gain'] for c in changes),losses=sum(not c['gain'] for c in changes),changes=changes))
        for body in sorted(set(r['body'] for r in rows)):
            for mode in modes:
                group=[r for r in rows if r['method']==method and r['body']==body and r['mode']==mode]
                bybody.append(dict(method=method,body=body,mode=mode,dof=group[0]['dof'],success=sum(r['success'] for r in group),n=len(group)))
    (ROOT/'summary.json').write_text(json.dumps(dict(aggregate=summary,paired=pairs,by_body=bybody),indent=2))
    totals=lambda method,mode:next(r for r in summary if r['method']==method and r['mode']==mode and r['dof']=='all')
    learned=totals('learned','frozen_bias');analytical=totals('analytical','frozen_bias')
    headline=f"Motion-only adaptation succeeds in {learned['success']}/48 learned checkpoint-task runs and {analytical['success']}/16 analytical tasks. Learned both-updates is27/48 and neither-updates is30/48; analytical references are12/16 and13/16 respectively."
    if learned['success']<=27:headline+=' Freezing bias while retaining learned response updates does not improve aggregate success over updating both.'
    if learned['success']<=30:headline+=' It also does not establish a benefit of learned online response updating over the fully calibrated frozen control.'
    lines=['# E115: Motion adaptation with frozen command bias','',headline,'',
        '64 new episodes complete the response/bias update factorial. The192 E114 controls are reused and explicitly not counted as new experiments. Same16 previously studied unseen-body development tasks, eight bodies, four axis families and two placements per body. Learned counts pool three frozen checkpoint seeds on the same16 tasks; analytical counts are16 distinct tasks. No physics selector, geometry filter, route planner or privileged Jacobian.','',
        'All conditions receive32 measured home-probe transitions. The new frozen_bias condition retains that calibrated bias exactly while updating the motion estimate from subsequent measured transitions, including the existing recent32-history switch and local32-step probe. It uses one E99 retained checkpoint throughout. Bias is checked for exact equality at every update. It is not zero bias.','',
        '| Estimator | Condition | DOF | Success | Collision episodes |','|---|---|---|---:|---:|']
    for r in summary:lines.append(f"| {r['method']} | {r['mode']} | {r['dof']} | {r['success']}/{r['n']} | {r['collision']} |")
    lines+=['','`online`: both updates. `frozen_response`: bias only. `frozen_all`: neither. `frozen_bias`: motion only.','',
        '## Paired changes: motion-only versus each control','',
        '| Estimator | Control | DOF | Gains | Losses |','|---|---|---|---:|---:|']
    for p in pairs:lines.append(f"| {p['method']} | {p['control']} | {p['dof']} | {p['gains']} | {p['losses']} |")
    lines+=['','## Interpretation','',
        'Motion-only versus neither isolates response/history updating with calibrated bias held fixed. Motion-only versus both isolates bias updating with response updating active. Both versus bias-only tests response updating under active bias adaptation. These effects may interact; pooled success differences do not imply universal benefit across bodies or checkpoints.',
        '', 'All policies use measured joint/tool pose, a common reset configuration and fixed task-phase goals. Later trajectories and probe commands can diverge across conditions; no additional interaction budget is granted. This is a reused development-set attribution test, not a new confirmatory test, native-arm result or flow-matching comparison. No weights are trained or promoted.',
        '', '## Verification and saved artifacts','',
        'Twelve old-condition episodes (cases0/8, analytical/learned_0, all three controls) are re-executed through the extended evaluator and match every old trace and state exactly. All64 new initial estimators/calibrations match E114. Bias invariance is asserted on every new task update. Checkpoint and source-result hashes remain unchanged. Complete traces, states, models, body/seed outcomes and outcome-selected success/failure videos are saved.']
    text='\n'.join(lines)+'\n';(ROOT/'report.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,dof in zip(axes,('6','7')):
        for i,method in enumerate(('learned','analytical')):
            vals=[next(r for r in summary if r['method']==method and r['mode']==mode and r['dof']==dof) for mode in modes]
            x=np.arange(4)+(i-.5)*.34;ax.bar(x,[r['success']/r['n']*100 for r in vals],.32,label=method)
            for xx,r in zip(x,vals):ax.text(xx,r['success']/r['n']*100+1,f"{r['success']}/{r['n']}",ha='center',fontsize=9)
        ax.set_title(f'{dof}-joint unseen development bodies');ax.set_xticks(range(4),['Both','Bias only','Neither','Motion only']);ax.set_ylim(0,110);ax.legend();ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Complete task success (%)');fig.suptitle('E115 · Which updates help? · No privileged action selector');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    report=Path(__file__).with_name('TECHNICAL_REPORT.md');s=report.read_text(encoding='utf-8');marker='## E115: complete motion/bias adaptation factorial';assert marker not in s
    s=s.replace('Revision 102 · 14 November 2026','Revision 103 · 14 November 2026',1)
    s+='\n'+marker+'\n\n64 new motion-only episodes;192 reused E114 controls. Bias fixed at measured post-calibration values, response history updates unchanged. No action selector or geometry filter.\n\n'
    for r in summary:
        if r['dof']=='all':s+=f"- {r['method']} / {r['mode']}: {r['success']}/{r['n']} successful, {r['collision']} collisions.\n"
    s+='\nFull paired changes, body/seed breakdown and development-set scope in `outputs/ease_frozen_bias_v1/report.md`.\n';report.write_text(s,encoding='utf-8')
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(json.dumps([r for r in summary if r['dof']=='all'],indent=2))

if __name__=='__main__':run()
