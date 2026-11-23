import json,shutil
from pathlib import Path
import numpy as np
from .candidate_history_repair import ROOT

def run():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==276;tasks=json.loads((ROOT/'cases.json').read_text());lookup={(r['case'],r['method']):r for r in rows};aggregate=[];paired=[];legacy=[]
    for method in ('reference','candidate','analytical'):
        for group in ('all','fitting','held','original_train','original_held','broad_train','fresh_held'):
            for dof in ('all',6,7):
                g=[r for r in rows if r['method'].startswith(method) and (group=='all' or (group=='fitting' and not tasks[r['case']]['validation']) or (group=='held' and tasks[r['case']]['validation']) or tasks[r['case']]['group']==group) and (dof=='all' or r['dof']==dof)]
                aggregate.append(dict(method=method,group=group,dof=str(dof),n=len(g),success=sum(r['success'] for r in g),collision=sum(r['collision'] for r in g)))
    for seed in range(3):
        changes=[]
        for case in range(24):
            a=lookup[case,f'candidate_{seed}'];b=lookup[case,f'reference_{seed}']
            if a['success']!=b['success']:changes.append(dict(case=case,dof=a['dof'],group=tasks[case]['group'],fitting=not tasks[case]['validation'],gain=a['success']))
            if not a['success'] and (lookup[case,f'older_{seed}']['success'] or (case<12 and lookup[case,f'parent_{seed}']['success'])):legacy.append(dict(case=case,seed=seed,group=tasks[case]['group']))
        paired.append(dict(seed=seed,reference=sum(lookup[c,f'reference_{seed}']['success'] for c in range(24)),candidate=sum(lookup[c,f'candidate_{seed}']['success'] for c in range(24)),gains=sum(c['gain'] for c in changes),losses=sum(not c['gain'] for c in changes),fitting_gains=sum(c['gain'] and c['fitting'] for c in changes),fitting_losses=sum(not c['gain'] and c['fitting'] for c in changes),changes=changes))
    overall=lambda name:next(r for r in aggregate if r['method']==name and r['group']==r['dof']=='all');old=overall('reference');new=overall('candidate')
    gate=new['success']>old['success'] and new['collision']<=old['collision'] and not legacy and all(p['losses']==0 for p in paired)
    summary=dict(aggregate=aggregate,paired=paired,older_successes_still_lost=legacy,retention_gate_passed=gate)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# E119: Corrective learning on candidate-reached histories','',f"Task success {old['success']}/72 to {new['success']}/72; collisions {old['collision']} to {new['collision']}. Strict retention gate {'PASSED' if gate else 'FAILED'}.",'',
        'All48 E118 fitting trajectories are exactly replayed, including successful and failed episodes, and their pre-update histories are appended to the E118 dataset. Full traces and states match the stored candidates. No candidate histories from the eight held placements are added to fitting. Existing held histories are used only for fixed offline comparisons. Same24 training-body tasks; no unseen morphology evaluation.',
        '', 'Each seed starts its E118 checkpoint and receives a fixed1500-update corrective run. Batches contain own failed-task histories, own successful-task histories, older fitting histories and synthetic replay. Successful cases preserve current E118 predictions/actions. Failed cases target a successful older estimator on that same measured history when available, otherwise analytical ridge. This is target supervision, not rollout reset or an inference-time teacher/selector.',
        '', 'The previous conflicting analytical matrix/response objectives are removed from protected successful contexts. Matrix and action targets agree on which reference is preserved. Measured-response loss applies only to unprotected histories. All fitting rows must be clean up to the current transition; all failed outcomes and their full trajectories remain recorded. Explicit probe actions are excluded from the nominal DLS action objective.',
        '', '| Method | Group | DOF | Success | Collision episodes |','|---|---|---|---:|---:|']
    for r in aggregate:lines.append(f"| {r['method']} | {r['group']} | {r['dof']} | {r['success']}/{r['n']} | {r['collision']} |")
    lines+=['','## Paired recovery and retention','',json.dumps(paired,indent=2),'','Unresolved successes from older E117/E99 references:',json.dumps(legacy,indent=2),'',
        '## Scope and decision','',
        'The correction is evaluated on every case and checkpoint, not only the selected fitting failures. Fitting recovery is in-sample; original/fresh held placements are the generalization checks on these same training bodies. Targets from successful prior estimators need not be optimal on off-trajectory histories; their effect must be judged by complete episodes.',
        '', 'No task-phase, bias, controller, collision geometry, physics selector or inference-time estimator architecture changes. No best-checkpoint/seed selection; all three final checkpoints are retained. Promotion requires increased overall success, no lost current success, no collision increase and retention of older successes. Unseen-body testing stays deferred.',
        '', 'All204 inherited trajectories,48 exact fitting-history replays and72 new candidate episodes remain traceable. Checkpoints, contexts, targets/provenance, states, XMLs and outcome-labeled success/failure videos are saved.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4));groups=('original_train','broad_train','original_held','fresh_held')
    for i,method in enumerate(('reference','candidate')):
        vals=[next(r for r in aggregate if r['method']==method and r['group']==g and r['dof']=='all') for g in groups];x=np.arange(4)+(i-.5)*.34;ax.bar(x,[r['success']/r['n']*100 for r in vals],.32,label=method)
        for xx,r in zip(x,vals):ax.text(xx,r['success']/r['n']*100+2,f"{r['success']}/{r['n']}",ha='center')
    ax.set_xticks(range(4),['Original fitted','Shifted fitted','Original held','Fresh held']);ax.set_ylim(0,115);ax.legend(loc='upper center',ncol=2);ax.set_ylabel('Task success (%)');ax.set_title('E119 · Correcting candidate-reached histories');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');s=p.read_text(encoding='utf-8');marker='## E119: candidate-reached corrective histories';assert marker not in s;s=s.replace('Revision 106 · 14 November 2026','Revision 107 · 14 November 2026',1)
    s+='\n'+marker+'\n\n'+lines[2]+'\n\n48 exact fitting-history replays, three fixed1500-update corrective runs and72 candidate evaluations. No held-history optimization or unseen evaluation. Full paired gains/losses and legacy retention: `outputs/ease_candidate_history_v1/report.md`.\n';p.write_text(s,encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print(json.dumps(dict(overall=[old,new],paired=paired,gate=gate),indent=2))

if __name__=='__main__':run()
