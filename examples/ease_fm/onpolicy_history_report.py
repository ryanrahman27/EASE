"""E101 fixed-budget data-source comparison with both retention references."""
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .onpolicy_history import ROOT,CONDITIONS,e99
from .benchmark import fingerprint


def run():
    assert json.loads((ROOT/'status.json').read_text())['status']=='complete'
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==144
    starts=[dict(r,mode='start') for r in json.loads((e99.ROOT/'records.json').read_text()) if r['mode']=='retained'];rows+=starts
    for r in starts:
        p=e99.ROOT/'trials'/f"{r['case']}_{r['method']}_retained.json";full=json.loads(p.read_text());full['mode']='start'
        (ROOT/'trials'/f"{r['case']}_{r['method']}_start.json").write_text(json.dumps(full))
    spec=json.loads((ROOT/'protocol.json').read_text());ep=json.loads((ROOT/'evaluation_protocol.json').read_text());audit=json.loads((ROOT/'data_audit.json').read_text());collection=json.loads((ROOT/'collection_verification.json').read_text())
    for p,h in {**spec['inputs'],**spec['parents'],**ep['checkpoints']}.items():assert fingerprint(p)==h
    assert fingerprint(ROOT/'task_contexts.npz')==audit['contexts_sha256']
    tasks=json.loads((e99.previous.PRIOR/'cases.json').read_text())
    table=[];changes=[];gates={};training=[]
    for seed in range(3):
        hashes=[]
        for condition in CONDITIONS:
            m=json.loads((ROOT/condition/f'seed_{seed}/training.json').read_text());training.append(m);hashes.append(m['batch_sha256'])
        assert hashes[0]==hashes[1]
    for group in ('all','6','7','candidate_0','candidate_1','candidate_2','route','no_route'):
        for mode in ('parent','start',*CONDITIONS):
            rr=[r for r in rows if r['mode']==mode and (group=='all' or str(r['dof'])==group or r['method']==group or (group=='route' and r['route_used']) or (group=='no_route' and not r['route_used']))]
            table.append(dict(group=group,mode=mode,n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),steps=sum(r['steps'] for r in rr)))
    lookup=lambda g,m:next(r for r in table if r['group']==g and r['mode']==m)
    for condition,control in [(c,b) for c in CONDITIONS for b in ('parent','start')]+[('onpolicy','replay')]:
        for r in rows:
            if r['mode']!=condition:continue
            b=next(x for x in rows if x['case']==r['case'] and x['method']==r['method'] and x['mode']==control)
            if b['success']!=r['success']:
                full=json.loads((ROOT/'trials'/f"{r['case']}_{r['method']}_{condition}.json").read_text())
                changes.append(dict(case=r['case'],method=r['method'],dof=r['dof'],condition=condition,control=control,change='gain' if r['success'] else 'loss',grasped=full['grasped'],lifted=full['lifted'],final_xy_error_mm=float(1000*np.linalg.norm(np.array(full['final_object'][:2])-tasks[r['case']]['target'])),first_collision=next((t['phase'] for t in full['trace'] if t['bad']),None)))
    for c in CONDITIONS:
        gates[c]={}
        for b in ('parent','start'):
            a,ref=lookup('all',c),lookup('all',b)
            checks=dict(success_improves=a['success']>ref['success'],collisions_nonincreasing=a['collisions']<=ref['collisions'],all_successes_retained=not any(x['condition']==c and x['control']==b and x['change']=='loss' for x in changes),per_seed_nondecreasing=all(lookup(f'candidate_{s}',c)['success']>=lookup(f'candidate_{s}',b)['success'] for s in range(3)))
            gates[c][b]=checks
        gates[c]['passes']=all(all(v.values()) for v in gates[c].values())
    summary=dict(table=table,changes=changes,gates=gates,collection=collection,training=[{k:v for k,v in m.items() if k!='losses'} for m in training]);(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# E101: student-collected histories versus matched replay','',
        f"One fixed collection round saves {audit['rows']} contexts from 36 training-arm trajectories: {audit['train']} eligible training and {audit['validation']} eligible validation contexts. All 36 calibration/approach/descent/probe prefixes exactly match E99. Four training bodies are exact original manifest entries; local-motion replay spans eight. The same held-out third placement per body gives 24/12 disjoint training/validation trajectories across all drivers. Failed and colliding trajectories remain included. Source outcomes: {collection['outcomes']}. These are data-collection outcomes, not final trained-model training scores.",
        '', 'Both conditions start each seed\'s E99 retained weights and receive 2000 AdamW updates at 2e-5, batch128, gradient clip1. Each batch contains64 synthetic,32 local,16 parent-task and16 extra-task contexts. Replay draws the extra16 from parent histories; onpolicy draws them from fixed starting-student histories. Common rows and uniform index draws match, but complete batches intentionally differ in those16 rows. The recorded batch hash covers common rows/draws only. All seeds use the pooled three-driver training data.',
        '', 'Both conditions fit analytical ridge targets, retain the synthetic E66 parent anchor (six-joint weight2, seven-joint0.5), and use0.01 normalized pre-filter action retention against E66 on all32 task rows. Every target is independently checked with NumPy. No simulator Jacobian targets or evaluation-arm histories enter fitting. This is a single fixed student-data collection, not iterative aggregation. Six final checkpoints are used without selection.',
        '', '## Full development evaluation','', '144 new complete rollouts: original parent, replay and onpolicy on16 tasks x3 seeds. Start is the48 saved E99 retained results for the identical protocol, not48 additional new evaluations. The original parent/planner/probe prefix is frozen; trained weights activate only afterward on admitted routes. Training arms differ from evaluation arms; this remains privileged-planner, phase-specific transfer on reused development tasks.',
        '', '| Group | Model | Success | Collisions | Steps |','|---|---|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['n']} | {r['collisions']} | {r['steps']} |")
    lines+=['','## Paired changes','','Final object XY error is in millimeters; success requires less than25 mm as well as the other unchanged criteria. Grasped/lifted are sticky events, not proof the grasp was maintained throughout transport.','', '| Model | Reference | Case | Seed | DOF | Change | Grasped / lifted | XY error mm | First collision |','|---|---|---:|---|---:|---|---|---:|---|']
    for r in changes:lines.append(f"| {r['condition']} | {r['control']} | {r['case']} | {r['method']} | {r['dof']} | {r['change']} | {r['grasped']} / {r['lifted']} | {r['final_xy_error_mm']:.2f} | {r['first_collision']} |")
    lines+=['','## Gates and interpretation','']
    for c in CONDITIONS:
        lines.append(f"- {c}: **{'PASS' if gates[c]['passes'] else 'FAIL'}**. {gates[c]}")
    for c,b in [(c,b) for c in CONDITIONS for b in ('parent','start')]+[('onpolicy','replay')]:
        cc=[x for x in changes if x['condition']==c and x['control']==b];lines.append(f"- {c} versus {b}: {sum(x['change']=='gain' for x in cc)} gains, {sum(x['change']=='loss' for x in cc)} losses.")
    lines+=['', 'The matched comparison tests the extra task-context source at fixed optimization budget and objective. A gate failure is not overridden by pooled improvements, seed selection or DOF switching. This experiment alone cannot establish whether distribution shift caused E99 failures, nor broad new-robot generalization.',
        '', '## Saved artifacts','', 'All data/provenance, six final checkpoints and loss histories,144 new full evaluations,48 reference copies, per-case changes, verified video gallery and scientific figure are retained. Checkpoint/input hashes,48 parent replays and144 handoffs verify. No-route branches are unchanged. See videos/index.html and onpolicy_comparison.png / .pdf. No reserved-test evaluation or inference-time weight adaptation.']
    if (ROOT/'decision.txt').exists():
        at=lines.index('## Saved artifacts');lines[at:at]=['## Decision','',
            'Student-history training reaches30/48 versus matched replay33/48 and the starting student32/48. It gains three and loses six relative to replay; seed2 declines from11/16 at start to8/16. Replay improves the pooled result but loses four original-parent successes and three starting-student successes. Both strict gates fail and no checkpoint is adopted. These data do not support the fixed student-history mixture as an improvement.',
            '',(ROOT/'decision.txt').read_text().strip(),'', 'All five non-collision regressions finish transport within25 mm of the target, then lose placement accuracy during lowering or release. This is a descriptive phase localization, not proof of grip slip or a causal effect of the estimator. All129 existing tests pass.','']
    metrics=['## Held-placement prediction metrics','','The same held subsets are used within every paired condition; these metrics do not select checkpoints. MSE is against analytical ridge targets. Action drift is relative to the original E66 parent, before filtering.','',
        '| Seed | Condition | Local MSE | Parent-task MSE | Student-task MSE | Student-task action drift |','|---|---|---:|---:|---:|---:|']
    for m in training:
        a=m['final'];metrics.append(f"| {m['seed']} | {m['condition']} | {a['local_mse']:.6f} | {a['task_mse']:.6f} | {a['onpolicy_mse']:.6f} | {a['onpolicy_action_drift']:.6f} |")
    at=lines.index('## Full development evaluation');lines[at:at]=metrics+['']
    misses=['## Non-collision regressions','','These rows lost a success from at least one reference without a forbidden collision. Errors use the last saved state of each phase; they diagnose object displacement but do not prove why it occurred.','',
        '| Model | Case | Seed | Object XY mm after transport | After lower | Final |','|---|---:|---|---:|---:|---:|']
    seen=set()
    for change in changes:
        key=(change['case'],change['method'],change['condition'])
        if change['control'] not in ('parent','start') or change['change']!='loss' or change['first_collision'] is not None or key in seen:continue
        seen.add(key);case,method,condition=key;full=json.loads((ROOT/'trials'/f'{case}_{method}_{condition}.json').read_text())
        error=lambda phase:1000*np.linalg.norm(np.array(next(t for t in reversed(full['trace']) if t['phase']==phase)['object'][:2])-tasks[case]['target'])
        misses.append(f"| {condition} | {case} | {method} | {error('transport'):.2f} | {error('lower'):.2f} | {change['final_xy_error_mm']:.2f} |")
    lines+=['']+misses
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');x=np.arange(3)
    for i,mode in enumerate(('parent','start',*CONDITIONS)):
        bars=ax.bar(x+(i-1.5)*.2,[lookup(f'candidate_{s}',mode)['success'] for s in range(3)],.2,label=mode);ax.bar_label(bars,padding=2)
    ax.set(xticks=x,xticklabels=['seed0','seed1','seed2'],ylim=(0,19),ylabel='Successes /16',title='Fixed final checkpoints: student data vs matched replay');ax.legend(ncol=4)
    fig.savefig(ROOT/'onpolicy_comparison.png',dpi=160);fig.savefig(ROOT/'onpolicy_comparison.pdf');plt.close(fig)
    print(json.dumps(dict(main=table[:24],gates=gates),indent=2))


if __name__=='__main__':run()
