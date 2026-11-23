"""E123 causal pulse comparisons, exact-state audit, and outcome media."""
import json,shutil
from pathlib import Path
import numpy as np
from .matched_correction import ROOT,OLD,PHASES,setup,evaluate,fingerprint

def report():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==100
    pairs=json.loads((ROOT/'pairs.json').read_text());lookup={(r['case'],r['method']):r for r in rows};comparisons=[]
    for pair in pairs:
        case,seed=pair['case'],pair['seed'];on=lookup[case,f'on_{seed}'];b=json.loads((ROOT/'branches'/f'{case}_on_{seed}.json').read_text())
        for phase in PHASES:
            off=lookup[case,f'off_{phase}_{seed}'];branch=json.loads((ROOT/'branches'/f'{case}_off_{phase}_{seed}.json').read_text());i=branch['intervention'];start=branch['metrics'][i];effect=int(on['success'])-int(off['success'])
            c=dict(case=case,seed=seed,group=pair['group'],original_gain=pair['e120_gain'],phase=phase,clean_start=start['clean_before'],off_steps=branch['off_steps'],on_success=on['success'],off_success=off['success'],success_effect=effect,on_collision=on['collision'],off_collision=off['collision'],collision_effect=int(off['collision'])-int(on['collision']),initial_signature=b['signatures'][i])
            for horizon in (1,8):
                a=next((x for x in b['metrics'] if x['phase']==phase and x['phase_step']==horizon-1),None);d=next((x for x in branch['metrics'] if x['phase']==phase and x['phase_step']==horizon-1),None)
                c[f'horizon_{horizon}']=None if a is None or d is None else dict(position_advantage_mm=1000*(d['position_error_after']-a['position_error_after']),rotation_advantage_rad=d['rotation_error_after']-a['rotation_error_after'],object_goal_advantage_mm=1000*(d['object_goal_xy_error']-a['object_goal_xy_error']),on_collision=a['collision_seen'],off_collision=d['collision_seen'])
            comparisons.append(c)
    counts=dict(helps=sum(c['success_effect']==1 for c in comparisons),hurts=sum(c['success_effect']==-1 for c in comparisons),unchanged=sum(c['success_effect']==0 for c in comparisons),clean_starts=sum(c['clean_start'] for c in comparisons))
    phase_summary=[]
    for phase in PHASES:
        cc=[c for c in comparisons if c['phase']==phase and c['clean_start']];phase_summary.append(dict(phase=phase,n=len(cc),helps=sum(c['success_effect']==1 for c in cc),hurts=sum(c['success_effect']==-1 for c in cc),unchanged=sum(c['success_effect']==0 for c in cc)))
    discordant=[dict(case=c['case'],seed=c['seed'],phase=c['phase'],success_effect=c['success_effect'],first_step_position_advantage_mm=c['horizon_1']['position_advantage_mm']) for c in comparisons if c['clean_start'] and c['success_effect'] and c['horizon_1'] is not None and abs(c['horizon_1']['position_advantage_mm'])>.001 and np.sign(c['horizon_1']['position_advantage_mm'])!=c['success_effect']]
    summary=dict(counts=counts,by_phase_clean=phase_summary,comparisons=comparisons,first_step_task_disagreements=discordant,e120_unchanged=True)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# E123: Correction-on/off continuations from identical states','',f"Across90 phase-entry pulse comparisons: correction-on helps final success in{counts['helps']}, hurts in{counts['hurts']}, and leaves final success unchanged in{counts['unchanged']}. {counts['clean_starts']}/90 starts have no prior forbidden collision. E120 remains unchanged.",'',
        'Scope: six E120 gains and four E120 regressions relative to E118. These ten selected checkpoint/placement pairs are development diagnostics, including previously exposed held placements. They are not an unbiased success-rate benchmark. There is no new training, selector or unseen-body evaluation.', '',
        'Intervention: at each of all nine task-phase entries, compare E120 continuously on with E120 residual off for up to eight control actions in that phase (0.32 seconds), then resume E120. The base action is still computed from the branch\'s current measured history; bias remains frozen after calibration. Explicit probing is unchanged. Episodes run to completion. Phases shorter than eight actions have shorter pulses and no matched eighth-step metric.', '',
        'Identical-state guarantee: replay the deterministic E120 prefix, asserting exact per-step fingerprints through the intervention. Fingerprints include full MuJoCo integration state (warmstart, controls and time included), Gram/cross statistics, frozen bias, recent history, local-history flag, task flags, phase step, probe counter, goal and probe anchor. Saved integration snapshots are compared independently. This is not a qpos-only reset. All ten correction-on traces and state sequences exactly match E120.', '',
        'Interpretation: help means on succeeds and off fails; hurt means off succeeds and on fails. Unchanged final success does not imply identical motion or no effect. These are effects of short phase-entry pulses, not labels for every individual action or proof of a deployable gate. Long-term differences include feedback/history and phase timing induced by the intervention. Contaminated starts cannot erase an earlier forbidden collision and are separately marked.', '',
        '## Clean-start comparisons by phase','','| Phase | Pairs | Helps | Hurts | Unchanged |','|---|---:|---:|---:|---:|']
    for r in phase_summary:lines.append(f"| {r['phase']} | {r['n']} | {r['helps']} | {r['hurts']} | {r['unchanged']} |")
    lines+=['','## Every comparison','','Position advantage = off minus on goal error; positive favors correction-on. One-step and eighth-step metrics are kept separate from whole-task outcomes. No unmatched phase duration is silently substituted.','','| Case/seed | Phase | Clean start | Off actions | On success | Off success | Position advantage at step1 (mm) | Step8 (mm) |','|---|---|---|---:|---|---|---:|---:|']
    for c in comparisons:
        show=lambda h:'n/a' if c[f'horizon_{h}'] is None else f"{c[f'horizon_{h}']['position_advantage_mm']:.4f}"
        lines.append(f"| {c['case']}/{c['seed']} | {c['phase']} | {c['clean_start']} | {c['off_steps']} | {c['on_success']} | {c['off_success']} | {show(1)} | {show(8)} |")
    lines+=['','## Local progress versus final outcome','',f'{len(discordant)} clean-start outcome-changing comparisons have first-step position-error advantage opposite to the final-success effect (excluding magnitudes <=0.001 mm). This compares one-step position progress with an up-to-eight-step intervention, not a one-action causal label. Local progress is not automatically a task-value signal.','', '```json',json.dumps(discordant,indent=2),'```','',
        'All100 full episodes, every state/command trace, intervention snapshot, history fingerprint and outcome are saved. Videos include successes and failures and paired examples of helpful/harmful pulses. Offline simulator intervention outcomes are diagnostic labels only; no simulator lookahead is added to E120.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    values=np.array([[next(c['success_effect'] for c in comparisons if c['case']==p['case'] and c['seed']==p['seed'] and c['phase']==phase) for phase in PHASES] for p in pairs])
    fig,ax=plt.subplots(figsize=(11,5));ax.imshow(values,vmin=-1,vmax=1,cmap=ListedColormap(['#d26450','#eeeeee','#48a37b']),aspect='auto')
    for i,p in enumerate(pairs):
        for j,phase in enumerate(PHASES):
            c=next(c for c in comparisons if c['case']==p['case'] and c['seed']==p['seed'] and c['phase']==phase);ax.text(j,i,('+' if values[i,j]>0 else '-' if values[i,j]<0 else '0')+(' *' if not c['clean_start'] else ''),ha='center',va='center')
    ax.set_xticks(range(9),PHASES,rotation=30,ha='right');ax.set_yticks(range(10),[f"{p['case']}/{p['seed']} ({'E120 gain' if p['e120_gain'] else 'E120 loss'})" for p in pairs]);ax.set_title('Effect of retaining E120 correction during a short phase-entry pulse\n+ helps final success; − hurts; 0 unchanged; * prior collision');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');s=p.read_text(encoding='utf-8');marker='## E123: matched-state correction pulses';assert marker not in s;s=s.replace('Revision 110 · 14 November 2026','Revision 111 · 14 November 2026',1);s+='\n'+marker+'\n\n'+lines[2]+'\n\nTen exact E120 references and90 phase-entry off-pulse continuations, full integration/history equality asserted. E120 unchanged. Selected development diagnostics, not new benchmark success rates. Details: `outputs/ease_matched_correction_v1/report.md`.\n';p.write_text(s,encoding='utf-8')
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(json.dumps(dict(counts=counts,by_phase_clean=phase_summary,flips=[c for c in comparisons if c['success_effect']]),indent=2))

def verify():
    tasks,models,hashes=setup();summary=json.loads((ROOT/'summary.json').read_text());rows=json.loads((ROOT/'records.json').read_text());checks=0
    action_checks=0
    for r in rows:
        case,method=r['case'],r['method'];actions=json.loads((ROOT/'actions'/f'{case}_{method}_frozen_bias.json').read_text());branch=json.loads((ROOT/'branches'/f'{case}_{method}.json').read_text());assert len(actions)==len(branch['metrics'])
        for a,b in zip(actions,branch['metrics']):
            assert b['off']==(b['phase']==r['off_phase'] and b['phase_step']<8)
            if not a['active']:continue
            base=np.array(a['base_action']);correction=np.array(a['correction_action']);expected=base if b['off'] else np.clip(base+np.clip(.25*(correction-base),-.001,.001),-.04,.04)
            np.testing.assert_array_equal(expected,a['nominal_action']);action_checks+=1
    for c in summary['comparisons']:
        case,seed,phase=c['case'],c['seed'],c['phase'];a=json.loads((ROOT/'branches'/f'{case}_on_{seed}.json').read_text());b=json.loads((ROOT/'branches'/f'{case}_off_{phase}_{seed}.json').read_text());i=b['intervention'];assert a['signatures'][:i+1]==b['signatures'][:i+1]
        aa=np.load(ROOT/'snapshots'/f'{case}_on_{seed}_{phase}.npz');bb=np.load(ROOT/'snapshots'/f'{case}_off_{phase}_{seed}_{phase}.npz');assert aa.files==bb.files
        assert fingerprint(ROOT/'states'/f'{case}_on_{seed}_frozen_bias.xml')==fingerprint(ROOT/'states'/f'{case}_off_{phase}_{seed}_frozen_bias.xml')
        for k in aa.files:np.testing.assert_array_equal(aa[k],bb[k])
        assert a['entries'][phase]==b['entries'][phase]
        sa=np.load(ROOT/'states'/f'{case}_on_{seed}_frozen_bias.npz')['states'];sb=np.load(ROOT/'states'/f'{case}_off_{phase}_{seed}_frozen_bias.npz')['states'];np.testing.assert_array_equal(sa[:i],sb[:i]);checks+=1
    selection=[]
    for outcome in (True,False):
        r=next(r for r in rows if r['off_phase'] is None and r['success']==outcome);selection.append(r)
    for effect in (1,-1):
        c=next((c for c in summary['comparisons'] if c['success_effect']==effect),None)
        if c:selection.append(next(r for r in rows if r['case']==c['case'] and r['seed']==c['seed'] and r['off_phase']==c['phase']))
    exact=[]
    for r in selection:
        case,seed,method=r['case'],r['seed'],r['method'];stem=f'{case}_{method}_frozen_bias';old=json.loads((ROOT/'trials'/f'{stem}.json').read_text());states=np.load(ROOT/'states'/f'{stem}.npz')['states'].copy();branch=json.loads((ROOT/'branches'/f'{case}_{method}.json').read_text())
        new,b=evaluate(case,tasks[case],seed,models[f'reference_{seed}'],models[f'correction_{seed}'],r['off_phase']);assert new==old and b==branch;np.testing.assert_array_equal(states,np.load(ROOT/'states'/f'{stem}.npz')['states']);exact.append(stem);print('exact',stem,flush=True)
    for path,h in hashes.items():assert fingerprint(path)==h
    (ROOT/'verification.json').write_text(json.dumps(dict(independent_snapshot_prefix_checks=checks,action_steps_checked=action_checks,exact_replays=exact,unchanged_weights=True),indent=2))

def finish():
    p=json.loads((ROOT/'protocol.json').read_text());v=json.loads((ROOT/'verification.json').read_text());m=json.loads((ROOT/'videos/manifest.json').read_text())
    for path,h in {**p['inputs'],**p['checkpoints']}.items():assert fingerprint(path)==h
    for r in m['videos']:assert fingerprint(ROOT/'videos'/r['file'])==r['sha256']
    for path in (ROOT/'source').glob('*.py'):assert fingerprint(path)==fingerprint(Path(__file__).with_name(path.name))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete',e120_unchanged=True,on_episodes=10,off_pulse_episodes=90,snapshot_pairs_verified=90,verification_replays=len(v['exact_replays']),videos=len(m['videos']),unseen_evaluated=False),indent=2))
    s=json.loads((ROOT/'summary.json').read_text());rescues={}
    for c in s['comparisons']:
        if c['success_effect']==-1:rescues.setdefault(f"{c['case']}/{c['seed']}",[]).append(c['phase'])
    path=ROOT/'report.md';path.write_text(path.read_text()+f"\nFinal audit: all90 snapshot/prefix pairs independently verified; {v['action_steps_checked']} action choices independently checked; {len(v['exact_replays'])} full episodes independently reexecuted exactly; {len(m['videos'])} trace-verified videos. Original E120 input/model hashes remain unchanged.\n\nAll four selected E120 regressions have at least one successful off-pulse continuation. Rescuing phases by case/seed:\n\n```json\n"+json.dumps(rescues,indent=2)+'\n```\n\nThis does not constitute a repaired policy: choosing these phases by known outcome would be an oracle. The next useful step is broader matched-state intervention data on fitting placements and a task-value gate evaluated separately on held placements. The same phase contains both beneficial and harmful pulses, and local position progress can disagree with final success. Keep E120 unchanged.\n')
    (ROOT/'artifact_hashes.json').write_text(json.dumps({str(path.relative_to(ROOT)):fingerprint(path) for path in ROOT.rglob('*') if path.is_file() and path.name!='artifact_hashes.json'},indent=2));print((ROOT/'status.json').read_text())

if __name__=='__main__':
    import sys
    {'report':report,'verify':verify,'finish':finish}[sys.argv[1]]()
