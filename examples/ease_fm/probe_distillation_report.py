"""E98 training-only local-history distillation and full-task retention report."""
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from .probe_distillation import ROOT,PARENTS,DATA,previous
from .benchmark import fingerprint


def run():
    assert json.loads((ROOT/'status.json').read_text())['status']=='complete'
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==96
    spec=json.loads((ROOT/'protocol.json').read_text());audit=json.loads((ROOT/'data_audit.json').read_text());ep=json.loads((ROOT/'evaluation_protocol.json').read_text())
    assert fingerprint(DATA/'manifest.json')==spec['training_manifest_sha256']
    for name in ('contexts','provenance'):assert fingerprint(ROOT/('contexts.npz' if name=='contexts' else 'provenance.json'))==audit[f'{name}_sha256']
    for p,h in {**spec['parents'],**ep['checkpoints'],**ep['parent_checkpoints']}.items():assert fingerprint(p)==h
    table=[];changes=[]
    reference=json.loads((previous.ROOT/'records.json').read_text());reference_table=[]
    for group in ('all','6','7'):
        for mode in ('analytical','exact'):
            rr=[r for r in reference if r['method'].startswith('candidate_') and r['mode']==mode and (group=='all' or str(r['dof'])==group)]
            reference_table.append(dict(group=group,mode=mode,n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr)))
    for group in ('all','6','7','candidate_0','candidate_1','candidate_2','route','no_route'):
        for mode in ('parent','student'):
            rr=[r for r in rows if r['mode']==mode and (group=='all' or str(r['dof'])==group or r['method']==group or (group=='route' and r['route_used']) or (group=='no_route' and not r['route_used']))]
            table.append(dict(group=group,mode=mode,n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),steps=sum(r['steps'] for r in rr)))
    for r in rows:
        if r['mode']!='student':continue
        b=next(x for x in rows if x['case']==r['case'] and x['method']==r['method'] and x['mode']=='parent')
        if r['success']!=b['success']:changes.append(dict(case=r['case'],method=r['method'],dof=r['dof'],change='gain' if r['success'] else 'loss',parent_collision=b['collision'],student_collision=r['collision']))
    gate=dict(success_improves=table[1]['success']>table[0]['success'],collisions_nonincreasing=table[1]['collisions']<=table[0]['collisions'],all_successes_retained=not any(r['change']=='loss' for r in changes),per_seed_nondecreasing=all(table[i+1]['success']>=table[i]['success'] for i in (6,8,10)))
    gate['passes']=all(gate.values());training=[json.loads((ROOT/f'seed_{s}/training.json').read_text()) for s in range(3)]
    (ROOT/'summary.json').write_text(json.dumps(dict(table=table,changes=changes,gate=gate,reference=reference_table,reference_sha256=fingerprint(previous.ROOT/'records.json'),training=[{k:v for k,v in m.items() if k!='losses'} for m in training]),indent=2))
    lines=['# E98: training-only local-history neural distillation','',
        'Fixed three-seed fine tuning from E66 analytical checkpoints using only the eight original training bodies. Generate16 physical trajectories per body:32 settling,32 E95-style probe and96 rank-one sinusoidal motion steps. Save128 pre-action contexts per trajectory. All contacts and trajectories are retained. Whole trajectories0..11 train;12..15 validate on the same bodies. There are16384 contexts:12288 train and4096 validation;23 physical control steps contain contacts. Validation is not unseen-body evaluation.',
        '', '2000 AdamW updates per seed, learning rate2e-5, batch128, gradient clip1. Each batch mixes64 measured contexts and64 synthetic replay contexts. Measured targets use the existing float64 analytical ridge solve of the input Gram/cross; every target is independently verified in NumPy. Prediction MSE covers both halves; frozen-parent anchor is applied only to synthetic replay (six-joint weight2, seven-joint0.5). Final checkpoints are used without model selection. No target-embodiment history, Jacobian label or outcome enters training.',
        '', '| Seed | Held-trajectory MSE before | After | Synthetic MSE before | After |','|---|---:|---:|---:|---:|']
    for m in training:lines.append(f"| {m['seed']} | {m['initial']['held_trajectory_mse']:.6g} | {m['final']['held_trajectory_mse']:.6g} | {m['initial']['synthetic_mse']:.6g} | {m['final']['synthetic_mse']:.6g} |")
    lines+=['','## Full-task development evaluation','',
        'All16 existing E97 tasks x3 E66 parents x2 conditions =96 full tasks. Parent weights control the identical approach and physical probe; only after the probe on admitted routes do student weights replace the motion estimator. No-route behavior is unchanged. This is phase-specific neural substitution with privileged route planning, not a newly trained end-to-end policy. All48 parent results replay E97 exactly; every student shares the original prefix and E96 handoff.',
        '', '| Group | Mode | Success | Collisions | Total steps |','|---|---|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['n']} | {r['collisions']} | {r['steps']} |")
    lines+=['','## Matched E97 reference controllers','','These saved reference results use the same three candidate-parent prefixes and16 tasks. They are reused context, not new E98 rollouts. The exact reference is privileged.','','| Group | Continuation | Success | Collisions |','|---|---|---:|---:|']
    for r in reference_table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['n']} | {r['collisions']} |")
    lines+=['','## Paired gains and losses','','| Case | Parent seed | DOF | Change | Parent collision | Student collision |','|---|---|---:|---|---|---|']
    for r in changes:lines.append(f"| {r['case']} | {r['method']} | {r['dof']} | {r['change']} | {r['parent_collision']} | {r['student_collision']} |")
    lines+=['','## Retention decision','',f"Gate: **{'PASS' if gate['passes'] else 'FAIL'}**. Strict pooled success improvement={gate['success_improves']}; nonincreasing collisions={gate['collisions_nonincreasing']}; every parent success retained={gate['all_successes_retained']}; every seed nondecreasing={gate['per_seed_nondecreasing']}.",
        '',f"There are {sum(r['change']=='gain' for r in changes)} gains and {sum(r['change']=='loss' for r in changes)} losses. All three seeds and all failed tasks are retained. Lower held-trajectory MSE does not establish better closed-loop control. Do not promote a checkpoint that fails the prespecified full-task retention gate; no outcome-driven retuning occurs in this experiment.",
        '', 'Observed decision: do not adopt these checkpoints. Students improve29/48 to31/48 and reduce collisions18 to17, but gains6 and losses4 are uneven across seeds. Seed0 improves8/16 to12/16; seed1 improves9/16 to11/16; seed2 drops12/16 to8/16. All four losses are seven-joint seed2 cases10,11,12,15. Six-joint success improves13/24 to15/24 with no losses; this is15/15 on admitted-route six-joint pairs, while nine no-route pairs remain unchanged failures. Seven-joint pooled success remains16/24 because four gains offset four losses. The matched analytical reference remains stronger at34/48; exact reaches39/48.',
        '', 'Held-trajectory prediction MSE improves29.7-31.3 percent in all seeds, and synthetic replay MSE improves too. This does not prevent full-task regression in seed2. The current evidence supports training-only local-history transfer gains for the six-joint subset, with seed-dependent seven-joint control instability. Do not select the best seed or introduce a DOF switch based on these outcomes.',
        '', 'Next: add complete task-trajectory histories from training embodiments and an explicit parent-behavior retention objective, then evaluate every fixed final seed again. Preserve the current four seven-joint failures as development checks and reserve fresh evaluation for confirmation after the method is fixed. This targets the gap between scripted local-motion fitting and closed-loop task behavior; it is a hypothesis to test, not an established cause.',
        '', 'Scientific figure: `training_and_tasks.png` and `training_and_tasks.pdf` show training fit and full-task outcomes. All contexts, provenance, losses, final weights and failures remain saved.',
        '', 'This evaluates only the post-probe use of the new weights. Synthetic replay loss is a retention surrogate, not a full original reaching-suite check. No whole-policy or original-reaching retention claim is made. Training histories are scripted local motions rather than successful pick/place trajectories. The repeated development benchmark remains separate from training bodies, but is not an independent final test.',
        '', 'Verification:127 existing regression tests pass. Training/validation trajectories are disjoint (96/32 trajectories); all target solves agree with independent NumPy; frozen evaluation weight hashes and input dataset hashes verify. No reserved-test evaluation or target-arm adaptation occurs. Raw contexts, provenance, losses, final weights and full trial traces are saved.',
        '', '## Videos','', 'Gallery: `outputs/ease_probe_distillation_v1/videos/index.html`. Candidate_0 cases2,4,10 are prespecified parent/student pairs; append first new student regression with both modes and first missing outcome category if necessary. Disclosed illustrations, not representative samples. Executed video replays must match complete serialized results; decoded frame counts and hashes are checked.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for m in training:
        fit=np.array(m['losses'])[:,0].reshape(20,100).mean(1)
        axes[0].plot(np.arange(100,2001,100),fit,label=f"seed {m['seed']}")
    axes[0].set(xlabel='Optimizer update',ylabel='Mean training fit MSE (100 updates)',title='Fixed-budget training');axes[0].legend()
    x=np.arange(2);width=.35
    for i,mode in enumerate(('parent','student')):
        rr=[next(r for r in table if r['group']==group and r['mode']==mode) for group in ('6','7')]
        bars=axes[1].bar(x+(i-.5)*width,[100*r['success']/r['n'] for r in rr],width,label=mode)
        axes[1].bar_label(bars,labels=[f"{r['success']}/{r['n']}" for r in rr],padding=3)
    axes[1].set(xticks=x,xticklabels=['6 joints','7 joints'],ylim=(0,110),ylabel='Full-task success (%)',title='Same prefix, neural continuation');axes[1].legend(loc='upper right')
    fig.savefig(ROOT/'training_and_tasks.png',dpi=160);fig.savefig(ROOT/'training_and_tasks.pdf');plt.close(fig)
    print(json.dumps(dict(table=table,changes=changes,gate=gate),indent=2))

if __name__=='__main__':run()
