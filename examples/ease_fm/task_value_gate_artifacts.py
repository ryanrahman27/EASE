"""E124 learned gate report, fitting isolation audit, and media."""
import json,shutil
from pathlib import Path
import numpy as np
import torch
from .task_value_gate import ROOT,COLLECT,OLD,setup,load_gate,evaluate,fingerprint,features,PHASES

def build():
    s=Path(__file__).with_name('bounded_history_videos.py').read_text().replace('bounded_history_correction import ROOT','task_value_gate import ROOT').replace('E120','E124').replace('Bounded correction / Retention','Learned task-value gate / Retention')
    needle='    selection=list(dict.fromkeys(selection));manifest='
    extra="    from .task_value_gate import OLD\n    for p in json.loads((OLD/'summary.json').read_text())['paired']:\n        for c in p['changes']:\n            for family in ('unbounded','candidate'):selection.append((c['case'],f\"{family}_{p['seed']}\",'frozen_bias'))\n"
    assert needle in s;s=s.replace(needle,extra+needle).replace('Outcome-selected diagnostic examples.','Includes all original E120 gains/regressions paired with the learned gate. Outcome-selected diagnostic examples.')
    Path(__file__).with_name('task_value_gate_videos.py').write_text(s)

def report():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==240;tasks=json.loads((ROOT/'cases.json').read_text());lookup={(r['case'],r['method']):r for r in rows};aggregate=[];changes=[]
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
    targeted=[]
    for p in json.loads((OLD/'summary.json').read_text())['paired']:
        for c in p['changes']:targeted.append(dict(case=c['case'],seed=p['seed'],previous_gain=c['gain'],success=lookup[c['case'],f"candidate_{p['seed']}"]['success']))
    retained=sum(x['success'] for x in targeted if x['previous_gain']);recovered=sum(x['success'] for x in targeted if not x['previous_gain']);gate=retained==6 and recovered==4 and not any(not c['gain'] for c in changes)
    summary=dict(aggregate=aggregate,paired_changes=changes,targeted_pairs=targeted,gains_retained=retained,regressions_recovered=recovered,retention_gate_passed=gate)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));training=json.loads((ROOT/'training.json').read_text());headline='; '.join(f"{r['method']}: {r['success']}/{r['n']}, {r['collisions']} collisions" for r in aggregate if r['split']==r['dof']=='all' and r['method']!='analytical')
    lines=['# E124: Fitting-only paired task outcomes and learned gate','',headline+'.','',
        'reference = frozen E118; unbounded = E120 per-step correction; candidate = E124 learned task-value gate. The filename unbounded means no task-value gate, not unlimited E120 actions.', '',
        '## Data and learning protocol','',
        f"Expanded to all16 fitting placements x3 estimator checkpoints:48 pairs,48 correction-on references and432 independent phase-entry off-pulse continuations. Reused70 fitting E123 trajectories;410 new collection trajectories. All432 intervention snapshots/prefixes checked. Of432 paired labels, {training['rows']} start without prior collision and are fitted; contaminated branches remain saved but excluded. Held rows in fitting:0.",'',
        'One shared MLP (139-64-32-2, tanh) predicts final success for correction-on and a short off-pulse followed by E120. Inputs: phase, joint mask, measured joint position/bias, position-goal error, history Gram/cross matrices, both proposed actions and local-history flag. No robot/case/seed identifier, simulator integration state, collision flag, future data or outcome is an input. Features are normalized using fitting rows only. Fixed seed124000,1500 full-batch AdamW updates, BCE supervision, no held-based threshold/checkpoint selection. Both motion estimators remain frozen.', '',
        'At each phase entry before an intervention, suppress E120 only if predicted off-success minus on-success exceeds0.10. Suppression lasts up to8 steps within that phase. Then resume E120 for the remainder of the episode. At most one off-pulse; this matches the label continuation and avoids composing untrained interventions. Explicit probe unchanged. No physics branching or geometry selector at inference.', '',
        'Training labels come from full-task outcomes, not prediction error or immediate position progress. Repeated phase labels within a placement are dependent: this is48 fitting checkpoint/placement trajectories, not hundreds of independent robots. Fitting performance is in-sample. Eight held placements were excluded from optimization; they are reused development cases, not an untouched final test. All bodies are training bodies.', '',
        f"Training diagnostics: head accuracy {training['training_head_accuracy']:.3f}; {training['training_switches']} predicted off choices on fitted rows, of which {training['training_switch_positive_advantage']} have positive observed advantage and {training['training_switch_negative_advantage']} have negative observed advantage. These are in-sample checks, not validation results.",'',
        '## Complete episodes','','| Method | Split | DOF | Success | Collision episodes |','|---|---|---|---:|---:|']
    for r in aggregate:lines.append(f"| {r['method']} | {r['split']} | {r['dof']} | {r['success']}/{r['n']} | {r['collisions']} |")
    lines+=['',f'Retained {retained}/6 E120 gains; recovered {recovered}/4 E120 regressions. Strict retention gate {gate}.','','## Every paired outcome change','','```json',json.dumps(changes,indent=2),'```','',
        'All collection episodes, intervention snapshots, training rows and provenance, gate weights, normalization, evaluation trajectories, decisions and success/failure videos are preserved. E120 remains an immutable baseline. Promote only after retention and held-placement checks; a better fitted score alone is not evidence of generalization.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    for i,(method,label) in enumerate((('reference','Frozen E118'),('unbounded','E120'),('candidate','Learned task-value gate'))):
        rr=[next(r for r in aggregate if r['method']==method and r['split']==sp and r['dof']=='all') for sp in ('fitting','held')];x=np.arange(2)+(i-1)*.23;ax.bar(x,[100*r['success']/r['n'] for r in rr],.21,label=label)
        for xx,r in zip(x,rr):ax.text(xx,100*r['success']/r['n']+2,f"{r['success']}/{r['n']}",ha='center')
    ax.set_xticks([0,1],['Fitting placements','Held placements']);ax.set_ylim(0,110);ax.set_ylabel('Task success (%)');ax.legend(ncol=3,loc='upper center');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');s=p.read_text(encoding='utf-8');marker='## E124: learned task-value gate';assert marker not in s;s=s.replace('Revision 111 · 14 November 2026','Revision 112 · 14 November 2026',1);s+='\n'+marker+'\n\n'+headline+f'. Retained{retained}/6 gains; recovered{recovered}/4 regressions; gate{gate}.\n\n480 fitting collection episodes (410 new), clean paired-outcome supervision, one shared fixed MLP and72 full-task evaluations. No held optimization or unseen bodies. Full report: `outputs/ease_task_value_gate_v1/report.md`.\n';p.write_text(s,encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print(headline);print(json.dumps(dict(retained=retained,recovered=recovered,changes=changes,gate=gate),indent=2))

def verify():
    tasks,models,hashes=setup();gate,mean,scale=load_gate();meta=json.loads((ROOT/'gate_rows.json').read_text());f=np.load(ROOT/'gate_dataset.npz');assert len(meta)==len(f['x']);assert all(not tasks[r['case']]['validation'] for r in meta)
    np.testing.assert_array_equal(f['mean'],f['x'].mean(0));np.testing.assert_array_equal(f['scale'],np.maximum(f['x'].std(0),1e-6))
    for i,r in enumerate(meta):
        case,seed,phase=r['case'],r['seed'],r['phase'];branch=json.loads((COLLECT/'branches'/f'{case}_on_{seed}.json').read_text());entry=branch['entries'][phase];a=json.loads((COLLECT/'actions'/f'{case}_on_{seed}_frozen_bias.json').read_text())[entry['step']];s=np.load(COLLECT/'snapshots'/f'{case}_on_{seed}_{phase}.npz')
        x=features(len(a['q_before']),s['gram'],s['cross'],s['bias'],a['q_before'],a['position_before'],s['goal'],phase,entry['metadata']['local'],a['base_action'],a['correction_action']);np.testing.assert_array_equal(x,f['x'][i])
        labels=[]
        for method in (f'on_{seed}',f'off_{phase}_{seed}'):labels.append(float(json.loads((COLLECT/'trials'/f'{case}_{method}_frozen_bias.json').read_text())['success']))
        np.testing.assert_array_equal(labels,f['y'][i])
    rows=json.loads((ROOT/'records.json').read_text());checked=0;matching=0;switches=0
    for r in rows:
        if not r['method'].startswith('candidate_'):continue
        case,seed=r['case'],int(r['method'].split('_')[-1]);d=json.loads((ROOT/'actions'/f"{case}_{r['method']}_gate.json").read_text());assert sum(x['selected'] for x in d['decisions'])<=1;switches+=d['chosen'] is not None
        for x in d['decisions']:
            with torch.no_grad():prob=gate(torch.tensor(np.clip((np.array(x['features'],np.float32)-mean)/scale,-10,10))).sigmoid().numpy()
            assert float(prob[0])==x['on_value'] and float(prob[1])==x['off_value'];assert bool(prob[1]-prob[0]>.1)==x['selected']
        actions=json.loads((ROOT/'actions'/f"{case}_{r['method']}_frozen_bias.json").read_text());counts={}
        for a in actions:
            phase=a['phase'];k=counts.get(phase,0);counts[phase]=k+1
            if not a['active']:continue
            base=np.array(a['base_action']);correction=np.array(a['correction_action']);expected=base if phase==d['chosen'] and k<8 else np.clip(base+np.clip(.25*(correction-base),-.001,.001),-.04,.04);np.testing.assert_array_equal(expected,a['nominal_action']);checked+=1
        if not tasks[case]['validation']:
            method=f'on_{seed}' if d['chosen'] is None else f"off_{d['chosen']}_{seed}"
            old=json.loads((COLLECT/'trials'/f'{case}_{method}_frozen_bias.json').read_text());old['method']=r['method'];new=json.loads((ROOT/'trials'/f"{case}_{r['method']}_frozen_bias.json").read_text());assert old==new
            np.testing.assert_array_equal(np.load(COLLECT/'states'/f'{case}_{method}_frozen_bias.npz')['states'],np.load(ROOT/'states'/f"{case}_{r['method']}_frozen_bias.npz")['states']);matching+=1
    exact=[]
    for outcome in (True,False):
        r=next((r for r in rows if r['method'].startswith('candidate_') and tasks[r['case']]['validation'] and r['success']==outcome),None)
        if r is None:continue
        case,seed=r['case'],int(r['method'][-1]);stem=f"{case}_{r['method']}_frozen_bias";old=json.loads((ROOT/'trials'/f'{stem}.json').read_text());states=np.load(ROOT/'states'/f'{stem}.npz')['states'].copy();new,_=evaluate(case,tasks[case],seed,models,gate,mean,scale);assert old==new;np.testing.assert_array_equal(states,np.load(ROOT/'states'/f'{stem}.npz')['states']);exact.append(stem)
    for p,h in hashes.items():assert fingerprint(p)==h
    assert fingerprint(ROOT/'gate.pt')==json.loads((ROOT/'training.json').read_text())['policy_sha256']
    (ROOT/'verification.json').write_text(json.dumps(dict(fitting_only_rows=len(meta),features_labels_verified=True,normalization_fitting_only=True,action_steps_checked=checked,switches=switches,fitting_continuations_exact=matching,held_exact_replays=exact,unchanged_weights=True),indent=2));print('verified',matching,'fitting continuations,',checked,'actions',flush=True)

def finish():
    p=json.loads((ROOT/'protocol.json').read_text());v=json.loads((ROOT/'verification.json').read_text());s=json.loads((ROOT/'summary.json').read_text());m=json.loads((ROOT/'videos/manifest.json').read_text())
    for path,h in {**p['inputs'],**p['checkpoints'],**p['reused_inputs']}.items():assert fingerprint(path)==h
    for r in m['videos']:assert fingerprint(ROOT/'videos'/r['file'])==r['sha256']
    for path in (ROOT/'source').glob('*.py'):assert fingerprint(path)==fingerprint(Path(__file__).with_name(path.name))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete',collection_episodes=480,new_collection_episodes=410,new_evaluation_episodes=72,retention_gate_passed=s['retention_gate_passed'],videos=len(m['videos']),held_training_rows=0,unseen_evaluated=False),indent=2))
    path=ROOT/'report.md';path.write_text(path.read_text()+f"\nVerification: all{v['fitting_only_rows']} clean training rows/features/labels and fitting-only normalization checked; {v['action_steps_checked']} action decisions verified; all{v['fitting_continuations_exact']} fitting evaluations exactly match their corresponding collected on/off continuation; {len(v['held_exact_replays'])} held episodes exactly replayed; {len(m['videos'])} trace-verified success/failure videos. E120 weights/inputs unchanged.\n")
    lines=['\n## Decision and held-placement errors','',
        'Keep E120 as the baseline; do not promote this gate. The learned gate recovers all four targeted fitting regressions and raises fitting success37/48 to46/48 without losing a fitting success. Held success falls15/24 to12/24: one gain and four losses, including all three originally held E120 gains. Overall58/72 versus52/72 therefore does not satisfy retention. These are placement-generalization errors on training bodies, not an unseen-body result.', '',
        'The sigmoid outputs are uncalibrated learned scores. Several harmful held switches receive large predicted advantages, so the result does not support simply treating a high margin as reliability. Broaden paired fitting-state coverage and test robustness without tuning a threshold to these held outcomes.', '',
        '| Held case/seed | Change vs E120 | Selected phase | Predicted on | Predicted off |','|---|---|---|---:|---:|']
    for change in s['paired_changes']:
        if change['baseline']!='unbounded' or change['split']!='held':continue
        d=json.loads((ROOT/'actions'/f"{change['case']}_candidate_{change['seed']}_gate.json").read_text());choice=next(x for x in d['decisions'] if x['selected'])
        lines.append(f"| {change['case']}/{change['seed']} | {'gain' if change['gain'] else 'loss'} | {d['chosen']} | {choice['on_value']:.4f} | {choice['off_value']:.4f} |")
    path.write_text(path.read_text()+'\n'.join(lines)+'\n')
    path=ROOT/'videos/index.html';path.write_text(path.read_text().replace('</h1>','</h1><p>reference = frozen E118; unbounded = E120; candidate = learned task-value gate. At most one short off-pulse, then E120 resumes. Training uses fitting placements only.</p>',1))
    # Video finalization hashes the immutable collection/evaluation artifacts.
    # Only these metadata files change afterwards; update their entries.
    manifest=json.loads((ROOT/'artifact_hashes.json').read_text())
    for name in ('report.md','status.json','videos/index.html','source/task_value_gate_artifacts.py'):manifest[name]=fingerprint(ROOT/name)
    assert set(manifest)=={str(path.relative_to(ROOT)) for path in ROOT.rglob('*') if path.is_file() and path.name!='artifact_hashes.json'}
    (ROOT/'artifact_hashes.json').write_text(json.dumps(manifest,indent=2));print((ROOT/'status.json').read_text())

if __name__=='__main__':
    import sys
    {'build':build,'report':report,'verify':verify,'finish':finish}[sys.argv[1]]()
