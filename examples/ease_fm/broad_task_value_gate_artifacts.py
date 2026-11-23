"""E125 learned gate report, fitting isolation audit, and media."""
import json,shutil
from pathlib import Path
import numpy as np
import torch
from .broad_task_value_gate import ROOT,COLLECT,OLD,setup,load_gate,evaluate,fingerprint,features,PHASES

def build():
    s=Path(__file__).with_name('bounded_history_videos.py').read_text().replace('bounded_history_correction import ROOT','broad_task_value_gate import ROOT').replace('E120','E125').replace('Bounded correction / Retention','Learned task-value gate / Retention')
    needle='    selection=list(dict.fromkeys(selection));manifest='
    extra="    from .broad_task_value_gate import OLD\n    for p in json.loads((OLD/'summary.json').read_text())['paired']:\n        for c in p['changes']:\n            for family in ('unbounded','candidate'):selection.append((c['case'],f\"{family}_{p['seed']}\",'frozen_bias'))\n"
    assert needle in s;s=s.replace('len(rows)==240','len(rows)==312');s=s.replace(needle,extra+needle).replace('Outcome-selected diagnostic examples.','Includes all original E120 gains/regressions paired with the learned gate. Outcome-selected diagnostic examples.')
    s=s.replace('fresh held placements','unchanged held development placements')
    s=s.replace('<img width="1000" src="../comparison.png">','<img width="1000" src="../comparison.png"><h2>Placement coverage</h2><p>Exact held placements remain excluded from fitting; the new fitting data cover a broader spatial region.</p><img width="1000" src="../placement_coverage.png">')
    Path(__file__).with_name('broad_task_value_gate_videos.py').write_text(s)

def report():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==312;tasks=json.loads((ROOT/'cases.json').read_text());lookup={(r['case'],r['method']):r for r in rows};aggregate=[];changes=[]
    for method in ('reference','unbounded','previousgate','candidate','analytical'):
        for split in ('all','fitting','held'):
            for dof in ('all',6,7):
                rr=[r for r in rows if r['method'].startswith(method) and (split=='all' or tasks[r['case']]['validation']==(split=='held')) and (dof=='all' or r['dof']==dof)]
                aggregate.append(dict(method=method,split=split,dof=str(dof),n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr)))
    for baseline in ('reference','unbounded','previousgate'):
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
    lines=['# E125: Broader fitting placements and learned task-value gate','',headline+'.','',
        'reference = frozen E118; unbounded = E120 per-step correction; previousgate = E124 gate; candidate = E125 broadened-data gate. The filename unbounded means no task-value gate, not unlimited E120 actions.', '',
        '## Data and learning protocol','',
        f"Expanded fitting placements16 to48 by adding32 randomized variants. All original24 placements, including8 held, remain unchanged. Across144 fitting checkpoint/placement pairs,144 on references and1296 off-pulse continuations are saved:480 reused E124 episodes and960 new episodes. All1296 intervention snapshots/prefixes checked. {training['rows']} clean paired labels enter fitting. Held rows in fitting:0.",'',
        'Placement seed125000: two variants per original fitting placement; object XY uniform in[0.275,0.425] x[-0.10,0.10] meters. Destination preserves the source displacement plus independent +/-20mm jitter. Same home and new calibration seeds. Coverage was widened after inspecting placement geometry and before any new outcomes were collected; the unused local-jitter catalog is archived in design_history/. Exact held placements and labels remain excluded, but negative-Y spatial support is now included in fitting. This changes the spatial extrapolation setting. No IK/success screening or held-outcome-driven threshold/model selection.', '',
        'One shared MLP (139-64-32-2, tanh) predicts final success for correction-on and a short off-pulse followed by E120. Inputs: phase, joint mask, measured joint position/bias, position-goal error, history Gram/cross matrices, both proposed actions and local-history flag. No robot/case/seed identifier, simulator integration state, collision flag, future data or outcome is an input. Features are normalized using fitting rows only. Fixed seed124000,1500 full-batch AdamW updates, BCE supervision, no held-based threshold/checkpoint selection. Both motion estimators remain frozen.', '',
        'At each phase entry before an intervention, suppress E120 only if predicted off-success minus on-success exceeds0.10. Suppression lasts up to8 steps within that phase. Then resume E120 for the remainder of the episode. At most one off-pulse; this matches the label continuation and avoids composing untrained interventions. Explicit probe unchanged. No physics branching or geometry selector at inference.', '',
        'Training labels come from full-task outcomes, not prediction error or immediate position progress. Repeated phase labels within a placement are dependent: this is144 fitting checkpoint/placement trajectories, not hundreds of independent robots. Fitting performance is in-sample. Eight held placements were excluded from optimization; they are reused development cases, not an untouched final test. All bodies are training bodies.', '',
        f"Training diagnostics: head accuracy {training['training_head_accuracy']:.3f}; {training['training_switches']} predicted off choices on fitted rows, of which {training['training_switch_positive_advantage']} have positive observed advantage and {training['training_switch_negative_advantage']} have negative observed advantage. These are in-sample checks, not validation results.",'',
        '## Complete episodes','','| Method | Split | DOF | Success | Collision episodes |','|---|---|---|---:|---:|']
    for r in aggregate:lines.append(f"| {r['method']} | {r['split']} | {r['dof']} | {r['success']}/{r['n']} | {r['collisions']} |")
    label_audit=json.loads((ROOT/'label_audit.json').read_text())
    lines+=['','## Fitting-label coverage','', '[Placement coverage figure](placement_coverage.png). Original and added fitting outcomes below are descriptive; multiple phase pairs share an episode.', '', '| Source | DOF | Clean pairs | E120 on success | Off helps | Off hurts | Unchanged |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in label_audit['groups']:lines.append(f"| {r['source']} | {r['dof']} | {r['clean_pairs']} | {r['on_success']}/{r['on_episodes']} | {r['off_helps']} | {r['off_hurts']} | {r['unchanged']} |")
    lines+=['',f'Retained {retained}/6 E120 gains; recovered {recovered}/4 E120 regressions. Strict retention gate {gate}.','','## Every paired outcome change','','```json',json.dumps(changes,indent=2),'```','',
        'All collection episodes, intervention snapshots, training rows and provenance, gate weights, normalization, evaluation trajectories, decisions and success/failure videos are preserved. E120 remains an immutable baseline. Promote only after retention and held-placement checks; a better fitted score alone is not evidence of generalization.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    for i,(method,label) in enumerate((('reference','Frozen E118'),('unbounded','E120'),('previousgate','E124 gate'),('candidate','E125 broader data'))):
        rr=[next(r for r in aggregate if r['method']==method and r['split']==sp and r['dof']=='all') for sp in ('fitting','held')];x=np.arange(2)+(i-1.5)*.20;ax.bar(x,[100*r['success']/r['n'] for r in rr],.18,label=label)
        for xx,r in zip(x,rr):ax.text(xx,100*r['success']/r['n']+2,f"{r['success']}/{r['n']}",ha='center')
    ax.set_xticks([0,1],['Fitting placements','Held placements']);ax.set_ylim(0,125);ax.set_ylabel('Task success (%)');ax.legend(ncol=4,loc='upper center');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');s=p.read_text(encoding='utf-8');marker='## E125: learned task-value gate';assert marker not in s;s=s.replace('Revision 112 · 14 November 2026','Revision 113 · 14 November 2026',1);s+='\n'+marker+'\n\n'+headline+f'. Retained{retained}/6 gains; recovered{recovered}/4 regressions; gate{gate}.\n\n1440 fitting collection episodes (960 new), clean paired-outcome supervision, one shared fixed MLP and72 full-task evaluations. No held optimization or unseen bodies. Full report: `outputs/ease_broad_task_value_gate_v1/report.md`.\n';p.write_text(s,encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
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
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete',collection_episodes=1440,new_collection_episodes=960,new_evaluation_episodes=72,retention_gate_passed=s['retention_gate_passed'],videos=len(m['videos']),held_training_rows=0,unseen_evaluated=False),indent=2))
    path=ROOT/'report.md';path.write_text(path.read_text()+f"\nVerification: all{v['fitting_only_rows']} clean training rows/features/labels and fitting-only normalization checked; {v['action_steps_checked']} action decisions verified; all{v['fitting_continuations_exact']} fitting evaluations exactly match their corresponding collected on/off continuation; {len(v['held_exact_replays'])} held episodes exactly replayed; {len(m['videos'])} trace-verified success/failure videos. E120 weights/inputs unchanged.\n")
    lines=['\n## Decision and held-placement errors','',
        f"E120 remains unchanged. Retained {s['gains_retained']}/6 original E120 gains and recovered {s['regressions_recovered']}/4 original regressions. Strict retention check (including E118, E120 and E124 successful episodes): {s['retention_gate_passed']}. See the full paired changes before interpreting aggregate improvement.", '',
        'E125 preserves all46 E124 fitting successes. Relative to E124 on held placements, it restores8/0 and23/1 but loses11/1: a net gain of one episode. Relative to E120, it adds nine fitting successes but loses held case20 at seeds0 and2, with no held gains. Both held losses are seven-joint episodes. The overall59/72 result therefore does not meet the requested no-loss generalization criterion; E125 is not promoted over E120.', '',
        'The harmful switches now occur at approach (20/0) and descend (20/2), with predicted off-success near1 despite subsequent failure. Broader fitting coverage recovers two previously lost E120 gains, but perfect fitting classification still does not imply reliable held decisions. This is descriptive evidence, not a demonstrated cause or a calibrated confidence estimate. The next proposed test is placement-grouped validation on fitting data for a conservative abstaining gate; no additional threshold or model was selected on these held outcomes.', '',
        'Compare the fixed original fitting and held placements against both E120 and the smaller-data E124 gate. The architecture, training seed, update count and threshold are unchanged; normalization is refit using expanded fitting data only. Sigmoid scores remain uncalibrated. No unseen embodiment claims are made.', '',
        '| Held case/seed | Change vs E120 | Selected phase | Predicted on | Predicted off |','|---|---|---|---:|---:|']
    for change in s['paired_changes']:
        if change['baseline']!='unbounded' or change['split']!='held':continue
        d=json.loads((ROOT/'actions'/f"{change['case']}_candidate_{change['seed']}_gate.json").read_text());choice=next(x for x in d['decisions'] if x['selected'])
        lines.append(f"| {change['case']}/{change['seed']} | {'gain' if change['gain'] else 'loss'} | {d['chosen']} | {choice['on_value']:.4f} | {choice['off_value']:.4f} |")
    path.write_text(path.read_text()+'\n'.join(lines)+'\n')
    path=ROOT/'videos/index.html';path.write_text(path.read_text().replace('</h1>','</h1><p>reference = frozen E118; unbounded = E120; previousgate = E124 gate; candidate = E125 broader-data gate. At most one short off-pulse, then E120 resumes. Training uses fitting placements only.</p>',1))
    # Video finalization hashes the immutable collection/evaluation artifacts.
    # Only these metadata files change afterwards; update their entries.
    manifest=json.loads((ROOT/'artifact_hashes.json').read_text())
    for name in ('report.md','status.json','videos/index.html','source/broad_task_value_gate_artifacts.py'):manifest[name]=fingerprint(ROOT/name)
    assert set(manifest)=={str(path.relative_to(ROOT)) for path in ROOT.rglob('*') if path.is_file() and path.name!='artifact_hashes.json'}
    (ROOT/'artifact_hashes.json').write_text(json.dumps(manifest,indent=2));print((ROOT/'status.json').read_text())

if __name__=='__main__':
    import sys
    {'build':build,'report':report,'verify':verify,'finish':finish}[sys.argv[1]]()
