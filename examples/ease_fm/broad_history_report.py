import json,shutil
from pathlib import Path
import numpy as np
from .broad_history_repair import ROOT

def run():
    rows=json.loads((ROOT/'records.json').read_text());tasks=json.loads((ROOT/'cases.json').read_text());assert len(rows)==204
    lookup={(r['case'],r['method']):r for r in rows};aggregate=[];paired=[];old_losses=[]
    for method in ('reference','candidate','analytical'):
        for group in ('all','original','original_train','original_held','broad_train','fresh_held'):
            for dof in ('all',6,7):
                g=[r for r in rows if r['method'].startswith(method) and (group=='all' or (group=='original' and r['case']<12) or tasks[r['case']]['group']==group) and (dof=='all' or r['dof']==dof)]
                aggregate.append(dict(method=method,group=group,dof=str(dof),n=len(g),success=sum(r['success'] for r in g),collision=sum(r['collision'] for r in g)))
    for seed in range(3):
        changes=[]
        for case in range(24):
            a=lookup[case,f'candidate_{seed}'];b=lookup[case,f'reference_{seed}']
            if a['success']!=b['success']:changes.append(dict(case=case,dof=a['dof'],group=tasks[case]['group'],gain=a['success']))
            if case<12 and lookup[case,f'parent_{seed}']['success'] and not a['success']:old_losses.append(dict(case=case,seed=seed,group=tasks[case]['group']))
        paired.append(dict(seed=seed,reference=sum(lookup[c,f'reference_{seed}']['success'] for c in range(24)),candidate=sum(lookup[c,f'candidate_{seed}']['success'] for c in range(24)),gains=sum(c['gain'] for c in changes),losses=sum(not c['gain'] for c in changes),changes=changes))
    score=lambda name:next(r for r in aggregate if r['method']==name and r['group']==r['dof']=='all')
    a=score('candidate');b=score('reference');gate=a['success']>b['success'] and a['collision']<=b['collision'] and not old_losses and all(p['losses']==0 for p in paired)
    metrics={str(s):{k:json.loads((ROOT/f'seed_{s}/training.json').read_text())[k] for k in ('initial','final','policy_sha256')} for s in range(3)}
    (ROOT/'summary.json').write_text(json.dumps(dict(aggregate=aggregate,paired=paired,older_parent_successes_lost=old_losses,retention_gate_passed=gate,matched_held_metrics=metrics),indent=2))
    lines=['# E118: Broader training histories and phase retention','',f"Candidate {a['success']}/72 versus E117 reference {b['success']}/72; collisions {a['collision']} versus {b['collision']}. Strict retention gate {'PASSED' if gate else 'FAILED'}. No unseen-body evaluation performed.",'',
        'Four existing training bodies,24 task placements: eight original fitted, four original held, eight shifted fitted and four fresh shifted held. Shifts (+/-12mm x,+/-8mm y) and calibration seeds are fixed before rollout outcomes. No reachability/success filtering. New histories use E117 three-checkpoint and analytical drivers. The original14,906-row context dataset is retained; no held placement enters training.',
        '', 'Three fixed2000-update final checkpoints start from E117. Batches combine original histories, shifted descent/lowering histories, shifted other phases and synthetic replay. Descent/lowering action error has3x weight. Successful reference actions/matrices are preserved on fitting cases; on original fitting cases where E117 failed but E99 succeeded, E99 supplies the preservation target. Other rows use analytical action targets. This is a combined data/objective change, not an isolated loss ablation.',
        '', 'Policy inference remains a learned inverse-free motion estimator with frozen calibrated bias and measured response-history updates. No physics selector, geometry filter, planner or privileged task Jacobian. Original and fresh held tasks are kept separate; all checkpoints and failures remain saved.',
        '', '| Method | Placement group | DOF | Success | Collision episodes |','|---|---|---|---:|---:|']
    for r in aggregate:lines.append(f"| {r['method']} | {r['group']} | {r['dof']} | {r['success']}/{r['n']} | {r['collision']} |")
    lines+=['','## Paired changes','',json.dumps(paired,indent=2),'','Older E99 successes still lost:',json.dumps(old_losses,indent=2),'',
        '## Matched held-history diagnostics','', '| Seed | Group | Action RMSE before/after (mrad) | XYZ error before/after (mm) |','|---|---|---:|---:|']
    for seed,m in metrics.items():
        for group in ('original','fresh'):
            x=m['initial'][group];y=m['final'][group];lines.append(f"| {seed} | {group} | {x['action_rmse']*1000:.3f} / {y['action_rmse']*1000:.3f} | {x['xyz_error']*1000:.3f} / {y['xyz_error']*1000:.3f} |")
    phases=json.loads((ROOT/'held_phase_metrics.json').read_text())
    lines+=['','## Descent and lowering on matched held histories','', '| DOF | Phase | Reference/candidate XYZ error (mm) | Reference/candidate action error (mrad) |','|---|---|---:|---:|']
    for dof in (6,7):
        for phase in ('descend','lower'):
            values={family:{k:float(np.mean([r[k] for r in phases if r['dof']==dof and r['phase']==phase and r['method'].startswith(family)])) for k in ('xyz_mm','action_mrad')} for family in ('reference','candidate')}
            x=values['reference'];y=values['candidate'];lines.append(f"| {dof} | {phase} | {x['xyz_mm']:.3f} / {y['xyz_mm']:.3f} | {x['action_mrad']:.3f} / {y['action_mrad']:.3f} |")
    lines+=['','Offline diagnostics use identical clean held histories for reference and candidate, with the analytical same-history action as comparator. The phase table combines original/fresh held histories and averages the three checkpoint means. Probe override actions are excluded. Full-task outcomes remain the primary gate; better offline means do not prove retained successes.','',
        '## Decision and scope','',
        'The strict gate requires increased overall reference success, zero lost reference successes, no collision increase, and retention of the older E99 successes on the original tasks. Zero losses also prevents joint-count or seed success declines. If it fails, retain both experimental candidates and earlier references without promotion. Regardless of the outcome, unseen-body testing remains deferred as requested. The fresh held set tests new placements on familiar training bodies, not new morphology.',
        '', 'All84 inherited original results,48 new reference trajectories and72 new candidate trajectories remain saved. Context provenance, full state sequences, XML scenes, final checkpoints and outcome-labeled success/failure videos are retained.']
    text='\n'.join(lines)+'\n';(ROOT/'report.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4));groups=('original_train','original_held','broad_train','fresh_held')
    for i,method in enumerate(('reference','candidate')):
        vals=[next(r for r in aggregate if r['method']==method and r['group']==g and r['dof']=='all') for g in groups];x=np.arange(4)+(i-.5)*.34;ax.bar(x,[r['success']/r['n']*100 for r in vals],.32,label=method)
        for xx,r in zip(x,vals):ax.text(xx,r['success']/r['n']*100+2,f"{r['success']}/{r['n']}",ha='center')
    ax.set_xticks(range(4),['Original fitted','Original held','Shifted fitted','Fresh held']);ax.set_ylim(0,115);ax.set_ylabel('Task success (%)');ax.legend(loc='upper center',ncol=2);ax.set_title('E118 · Training bodies only · No physics selector');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');s=p.read_text(encoding='utf-8');marker='## E118: broadened histories and descent/lowering retention';assert marker not in s;s=s.replace('Revision 105 · 14 November 2026','Revision 106 · 14 November 2026',1)
    s+='\n'+marker+'\n\n'+lines[2]+'\n\nEight added fitting placements and four fresh held placements on existing training bodies,48 new reference rollouts, three fixed2000-update runs and72 candidate rollouts. Complete paired outcomes and held-placement diagnostics: `outputs/ease_broad_history_v1/report.md`.\n';p.write_text(s,encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print(json.dumps(dict(overall=[b,a],paired=paired,older_parent_losses=old_losses,gate=gate),indent=2))

if __name__=='__main__':run()
