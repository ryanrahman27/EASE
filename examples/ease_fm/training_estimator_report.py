"""E117 matched offline errors and full-task retention report."""
import json,shutil
from pathlib import Path
import numpy as np
from .training_estimator_repair import ROOT

def run():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==84
    aggregate=[];paired=[];metrics={};lookup={(r['case'],r['method']):r for r in rows}
    for family in ('learned','candidate','analytical'):
        for split in ('all','train','held'):
            for dof in ('all',6,7):
                g=[r for r in rows if r['method'].startswith(family) and (split=='all' or (r['case']%3==2)==(split=='held')) and (dof=='all' or r['dof']==dof)]
                aggregate.append(dict(method=family,split=split,dof=str(dof),n=len(g),success=sum(r['success'] for r in g),collision=sum(r['collision'] for r in g)))
    for seed in range(3):
        m=json.loads((ROOT/f'seed_{seed}/training.json').read_text());metrics[str(seed)]={k:m[k] for k in ('initial','final','policy_sha256')}
        changes=[]
        for case in range(12):
            a=lookup[case,f'candidate_{seed}'];b=lookup[case,f'learned_{seed}']
            if a['success']!=b['success']:changes.append(dict(case=case,dof=a['dof'],held=case%3==2,gain=a['success']))
        paired.append(dict(seed=seed,parent_success=sum(lookup[c,f'learned_{seed}']['success'] for c in range(12)),candidate_success=sum(lookup[c,f'candidate_{seed}']['success'] for c in range(12)),gains=sum(c['gain'] for c in changes),losses=sum(not c['gain'] for c in changes),changes=changes))
    old=next(r for r in aggregate if r['method']=='learned' and r['split']==r['dof']=='all');new=next(r for r in aggregate if r['method']=='candidate' and r['split']==r['dof']=='all')
    gate=new['success']>old['success'] and new['collision']<=old['collision'] and all(p['losses']==0 and p['candidate_success']>=p['parent_success'] for p in paired)
    summary=dict(aggregate=aggregate,paired=paired,held_history_metrics=metrics,retention_gate_passed=gate)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# E117: Training-body estimator diagnosis and repair','',f"Candidate full-task success: {new['success']}/36 versus parent {old['success']}/36; collision episodes {new['collision']} versus {old['collision']}. Strict retention gate {'PASSED' if gate else 'FAILED'}. All three final checkpoints are retained; no seed/checkpoint selection.",'',
        '48 selector-free training-body trajectories provide pre-update histories, next measured motion and analytical reference actions. Three E99 parents and one analytical driver use identical calibration. First two placements per body are training, third is held out for every driver; all bodies themselves are training bodies. No unseen-body data enters training or checkpoint selection. All clean training contexts are eligible for motion supervision; probe actions are excluded from the action objective because the probe overrides nominal DLS commands.','',
        'Three fixed1500-update fine-tunes use ridge-matrix targets, same-history analytical-action targets and measured-response loss, plus synthetic replay. The network architecture and inference remain inverse-free; analytical solves generate training labels only. Bias remains frozen at calibrated values, with existing recent32-history update and phase budgets. No physics selector, geometry filter or planner.','',
        '## Matched held-placement history errors','', '| Seed | Metric | Parent | Candidate |','|---|---|---:|---:|']
    for seed,m in metrics.items():
        for key in ('matrix_mse','action_rmse','xyz_error'):
            lines.append(f"| {seed} | {key} | {m['initial']['all'][key]:.6g} | {m['final']['all'][key]:.6g} |")
    lines+=['','Action RMSE is joint-target radians against analytical DLS on identical histories/goals/bias (probe excluded); XYZ error is metres against the next observed tool transition. Both models use the same fixed held histories. These offline metrics do not establish closed-loop improvement.','',
        '## Full episodes','', '| Method | Placement split | DOF | Success | Collision episodes |','|---|---|---|---:|---:|']
    for r in aggregate:lines.append(f"| {r['method']} | {r['split']} | {r['dof']} | {r['success']}/{r['n']} | {r['collision']} |")
    lines+=['','## Paired outcomes','',json.dumps(paired,indent=2),'',
        '## Scope and next decision','',
        'This is a training-body repair experiment with held placements on those same bodies. It does not evaluate new morphology transfer. The strict gate requires pooled task improvement, no parent success lost, no seed decline, and no collision increase. No unseen or reserved final-test evaluation is performed here. Offline improvement alone cannot promote a checkpoint. If retention fails, preserve the parents and diagnose changed trajectories before further tuning.',
        '', 'Every baseline and candidate outcome, including collisions and failures, is retained with full traces, states and scene XML. Source and checkpoint hashes, row provenance and independent target checks are saved. Videos are outcome-selected illustrations.']
    text='\n'.join(lines)+'\n';(ROOT/'report.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for j,key in enumerate(('action_rmse','xyz_error')):
        for i,label in enumerate(('initial','final')):axes[j].bar(np.arange(3)+(i-.5)*.34,[metrics[str(s)][label]['all'][key]*1000 for s in range(3)],.32,label='parent' if i==0 else 'candidate')
        axes[j].set_xticks(range(3),['Seed0','Seed1','Seed2']);axes[j].set_ylabel('mrad' if key=='action_rmse' else 'mm');axes[j].set_title('Matched action error' if key=='action_rmse' else 'Matched motion prediction error');axes[j].legend()
    fig.suptitle('E117 · Held placements on training bodies');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');s=p.read_text(encoding='utf-8');marker='## E117: training-body learned-estimator repair';assert marker not in s
    s=s.replace('Revision 104 · 14 November 2026','Revision 105 · 14 November 2026',1);s+='\n'+marker+'\n\n'+lines[2]+'\n\n48 matched training-body drivers; three fixed1500-update final checkpoints;36 new candidate evaluations without selector or geometry filter. No unseen-body fitting or evaluation. Full offline metrics, paired task outcomes and provenance: `outputs/ease_estimator_repair_v1/report.md`.\n';p.write_text(s,encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
