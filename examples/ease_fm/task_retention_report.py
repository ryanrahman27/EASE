"""E99 matched task-data/action-retention report and scientific figure."""
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .task_retention_training import ROOT,CONDITIONS,prior
from .benchmark import fingerprint


def run():
    assert json.loads((ROOT/'status.json').read_text())['status']=='complete';rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==144
    spec=json.loads((ROOT/'protocol.json').read_text());ep=json.loads((ROOT/'evaluation_protocol.json').read_text());audit=json.loads((ROOT/'data_audit.json').read_text())
    for p,h in {**spec['inputs'],**spec['parents'],**ep['checkpoints'],**ep['parents']}.items():assert fingerprint(p)==h
    assert fingerprint(ROOT/'task_contexts.npz')==audit['contexts_sha256'] and fingerprint(ROOT/'provenance.json')==audit['provenance_sha256']
    table=[];changes=[];gates={};training=[]
    for seed in range(3):
        paired=[]
        for condition in CONDITIONS:
            m=json.loads((ROOT/condition/f'seed_{seed}/training.json').read_text());training.append(m);paired.append(m['batch_sha256'])
        assert paired[0]==paired[1]
    for group in ('all','6','7','candidate_0','candidate_1','candidate_2','route','no_route'):
        for mode in ('parent',*CONDITIONS):
            rr=[r for r in rows if r['mode']==mode and (group=='all' or str(r['dof'])==group or r['method']==group or (group=='route' and r['route_used']) or (group=='no_route' and not r['route_used']))]
            table.append(dict(group=group,mode=mode,n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),steps=sum(r['steps'] for r in rr)))
    for condition,control in [('data_only','parent'),('retained','parent'),('retained','data_only')]:
        for r in rows:
            if r['mode']!=condition:continue
            b=next(x for x in rows if x['case']==r['case'] and x['method']==r['method'] and x['mode']==control)
            if b['success']!=r['success']:
                full=json.loads((ROOT/'trials'/f"{r['case']}_{r['method']}_{condition}.json").read_text());first=next((t['phase'] for t in full['trace'] if t['bad']),None)
                changes.append(dict(case=r['case'],method=r['method'],dof=r['dof'],condition=condition,control=control,change='gain' if r['success'] else 'loss',collision=r['collision'],first_collision=first))
    lookup=lambda group,mode:next(r for r in table if r['group']==group and r['mode']==mode)
    for condition in CONDITIONS:
        a,b=lookup('all',condition),lookup('all','parent');cc=[r for r in changes if r['condition']==condition and r['control']=='parent']
        gate=dict(improves_success=a['success']>b['success'],nonincreasing_collisions=a['collisions']<=b['collisions'],all_parent_successes_retained=not any(r['change']=='loss' for r in cc),per_seed_nondecreasing=all(lookup(f'candidate_{s}',condition)['success']>=lookup(f'candidate_{s}','parent')['success'] for s in range(3)))
        gate['passes']=all(gate.values());gates[condition]=gate
    trials=[json.loads(p.read_text()) for p in (ROOT/'training_trials').glob('*.json')];assert len(trials)==36
    training_outcomes={split:dict(n=sum(t['validation']==val for t in trials),success=sum(t['validation']==val and t['success'] for t in trials),collisions=sum(t['validation']==val and t['collision'] for t in trials)) for split,val in [('train',False),('validation',True)]}
    (ROOT/'summary.json').write_text(json.dumps(dict(table=table,changes=changes,gates=gates,training_outcomes=training_outcomes,training=[{k:v for k,v in m.items() if k!='losses'} for m in training]),indent=2))
    lines=['# E99: training-task histories and matched action retention','',
        f"Collect36 complete parent-driven trajectories on12 existing training placements across four original training bodies. All cases and failures are retained. Save{audit['rows']} pre-action contexts; fitting uses only post-probe continuation contexts: {audit['train']} training and{audit['validation']} validation. Third placement per body is validation across every driver, giving24/12 disjoint trajectories. Reuse E98 local-motion data on all eight training bodies (12288 train/4096 validation). No evaluation-arm histories enter training.",
        '',f"Parent data-collection outcomes: training {training_outcomes['train']}; validation {training_outcomes['validation']}. These are source-driver outcomes, not trained-student training-task scores. Full phase traces and calibration observations are saved. Original geometry filtering and32-step probe are used; there is no training route planner. Existing IK homes are privileged setup.",
        '', 'Both conditions start from original E66 checkpoints. Each of2000 identical paired batches contains64 synthetic,32 local-motion and32 training-task contexts. AdamW lr2e-5, gradient clip1; analytical ridge prediction target and synthetic parent anchor are identical. Only retained adds0.01 times the mean squared clipped joint-command difference from the frozen parent, divided by0.04 squared, on task contexts. This uses the current measured bias, task error, DLS damping0.015 and gain0.35. It compares commands before geometry filtering and physics; it is not an executed-contact loss. Parent commands from unsuccessful trajectories are also retained, a possible recovery/retention tradeoff.',
        '', 'All six fixed final checkpoints are evaluated without checkpoint/seed selection. Paired batch hashes match within every seed. Local/task targets are independently checked against NumPy solves. Validation holds out trajectories/placements on the same training bodies.',
        '', '| Seed | Condition | Local MSE before / after | Task MSE before / after | Held task action drift | Synthetic MSE before / after |','|---|---|---:|---:|---:|---:|']
    for m in training:
        a,b=m['initial'],m['final'];lines.append(f"| {m['seed']} | {m['condition']} | {a['local_mse']:.5g} / {b['local_mse']:.5g} | {a['task_mse']:.5g} / {b['task_mse']:.5g} | {b['task_action_drift']:.5g} | {a['synthetic_mse']:.5g} / {b['synthetic_mse']:.5g} |")
    lines+=['','## Full-task evaluation','', 'All16 development tasks x3 seeds x parent/data_only/retained =144 complete rollouts. Same frozen parent approach and probe; only post-probe neural weights change on admitted routes. Every parent replay matches E97, and every handoff/prefix matches the saved references. No-route behavior remains unchanged. Evaluation remains phase-specific and uses privileged route planning.',
        '', '| Group | Mode | Success | Collisions | Total steps |','|---|---|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['n']} | {r['collisions']} | {r['steps']} |")
    lines+=['','## All paired changes','','| Condition | Control | Case | Seed | DOF | Change | Collision | First collision phase |','|---|---|---:|---|---:|---|---|---|']
    for r in changes:lines.append(f"| {r['condition']} | {r['control']} | {r['case']} | {r['method']} | {r['dof']} | {r['change']} | {r['collision']} | {r['first_collision']} |")
    lines+=['','## Retention gates','']
    for c,g in gates.items():lines.append(f"- {c}: **{'PASS' if g['passes'] else 'FAIL'}**. Success improves={g['improves_success']}; collisions nonincreasing={g['nonincreasing_collisions']}; every parent success retained={g['all_parent_successes_retained']}; every seed nondecreasing={g['per_seed_nondecreasing']}.")
    for condition,control in [('data_only','parent'),('retained','parent'),('retained','data_only')]:
        cc=[r for r in changes if r['condition']==condition and r['control']==control];lines.append(f"- {condition} versus {control}: {sum(r['change']=='gain' for r in cc)} gains, {sum(r['change']=='loss' for r in cc)} losses.")
    lines+=['','## Interpretation and verification','',
        'At the fixed coefficient, retained reaches32/48 successes with16 collisions versus parent29/48 with18 and data_only31/48 with17. Retained gains six parent failures but loses three parent successes: case10 seed0 (first collision in retreat), case11 seed2 (lift), and case13 seed1 (transport), all seven-joint. Seed2 declines12/16 to11/16. Both adoption gates fail; no checkpoint replaces the parent. Six-joint retained improves13/24 to14/24 without losing a parent success, but data_only reaches15/24. The action-retention term trades four recoveries against three losses relative to data_only, so its pooled gain is not universal.',
        '', 'Next decision: audit the three retained regressions at matched physical states, measuring parent/student command differences before and after geometry filtering and their accumulation by phase. First use read-only queries without changing executed actions; then predeclare any training-only remedy and repeat the full paired evaluation. Do not tune on these three trajectories or select a successful seed. On-policy student histories on training bodies are a candidate remedy for distribution shift, which this experiment has not established causally.',
        '',
        'The matched comparison isolates the added action-retention term at the fixed coefficient. E98 comparisons also change the data mixture, so they do not isolate task-data addition alone. Lower command drift on training-body validation does not guarantee retention on new-body full tasks. No failed gate is overridden by a favorable pooled score, seed selection or post-hoc DOF switch.',
        '', '129 tests pass, including controller-formula agreement for6/7 joints and finite action-loss gradients. All48 parent full rollouts and144 identical handoffs verify; both trained branches preserve all nine no-route cases. Dataset, parent, paired-batch and final checkpoint hashes are verified. No target-arm training, inference weight updates or reserved-test evaluation occurs. This is not a whole-policy or original-reaching-suite retention claim.',
        '', '## Videos and figures','', 'Gallery: `outputs/ease_task_retention_training_v1/videos/index.html`. Prespecified case2 candidate_0 and cases10/15 candidate_2, all three modes; first new regression for each condition is appended with its parent if absent, plus a missing outcome category if needed. Outcome selection is disclosed. Full serialized video replays, decoded frame counts and SHA-256 hashes are checked. Scientific figure: `retention.png` and `retention.pdf`.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained');x=np.arange(3);width=.24
    for i,mode in enumerate(('parent',*CONDITIONS)):
        rr=[lookup(f'candidate_{s}',mode) for s in range(3)];bars=axes[0].bar(x+(i-1)*width,[r['success'] for r in rr],width,label=mode);axes[0].bar_label(bars,padding=2)
    axes[0].set(xticks=x,xticklabels=['seed0','seed1','seed2'],ylim=(0,19),ylabel='Full-task successes /16',title='Fixed final checkpoints');axes[0].legend(fontsize=8)
    for i,condition in enumerate(CONDITIONS):
        rr=[next(m for m in training if m['seed']==s and m['condition']==condition) for s in range(3)];axes[1].bar(x+(i-.5)*.35,[m['final']['task_action_drift'] for m in rr],.35,label=condition)
    axes[1].set(xticks=x,xticklabels=['seed0','seed1','seed2'],ylabel='Normalized pre-filter action MSE',title='Training-body held-placement drift');axes[1].legend(fontsize=8)
    fig.savefig(ROOT/'retention.png',dpi=160);fig.savefig(ROOT/'retention.pdf');plt.close(fig)
    print(json.dumps(dict(main=table[:18],gates=gates,training_outcomes=training_outcomes),indent=2))

if __name__=='__main__':run()
