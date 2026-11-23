"""Report and verify all E120 outcomes; preserve failed and successful runs."""
import json
import shutil
from pathlib import Path
import numpy as np
from .bounded_history_correction import ROOT, setup, evaluate, fingerprint, CAP, ALPHA


def report():
    rows=json.loads((ROOT/'records.json').read_text()); assert len(rows)==240
    tasks=json.loads((ROOT/'cases.json').read_text()); lookup={(r['case'],r['method']):r for r in rows}
    aggregate=[]; paired=[]
    for method in ('reference','unbounded','candidate','analytical'):
        for group in ('all','fitting','held','original_train','original_held','broad_train','fresh_held'):
            for dof in ('all',6,7):
                part=[r for r in rows if r['method'].startswith(method) and (group=='all' or (group=='fitting' and not tasks[r['case']]['validation']) or (group=='held' and tasks[r['case']]['validation']) or tasks[r['case']]['group']==group) and (dof=='all' or r['dof']==dof)]
                aggregate.append(dict(method=method,group=group,dof=str(dof),n=len(part),success=sum(r['success'] for r in part),collision=sum(r['collision'] for r in part)))
    for seed in range(3):
        changes=[]
        for case in range(24):
            a=lookup[case,f'candidate_{seed}']; b=lookup[case,f'reference_{seed}']
            if a['success']!=b['success']:changes.append(dict(case=case,group=tasks[case]['group'],fitting=not tasks[case]['validation'],gain=a['success'],collision=a['collision']))
        paired.append(dict(seed=seed,reference=sum(lookup[c,f'reference_{seed}']['success'] for c in range(24)),candidate=sum(lookup[c,f'candidate_{seed}']['success'] for c in range(24)),gains=sum(x['gain'] for x in changes),losses=sum(not x['gain'] for x in changes),changes=changes))
    total=lambda method:next(r for r in aggregate if r['method']==method and r['group']==r['dof']=='all')
    a,b=total('reference'),total('candidate'); gate=b['success']>a['success'] and b['collision']<=a['collision'] and not any(p['losses'] for p in paired)
    divergence=json.loads((ROOT/'diagnosis/first_divergence.json').read_text())
    for d in divergence:
        stems=[f"{d['case']}_{family}_{d['seed']}_frozen_bias" for family in ('reference','unbounded')]
        logs=[json.loads((ROOT/'actions'/f'{stem}.json').read_text()) for stem in stems]
        for key in ('q_before','position_before','gram','cross'):
            assert logs[0][0][key]==logs[1][0][key]
        d['identical_initial_observation_and_history']=True
        d['first_executed_command_difference']=next((dict(step=i,baseline_phase=a['phase'],corrected_phase=b['phase'],max_rad=float(np.max(np.abs(np.array(a['executed_command'])-b['executed_command'])))) for i,(a,b) in enumerate(zip(*logs)) if np.max(np.abs(np.array(a['executed_command'])-b['executed_command']))>1e-10),None)
        states=[np.load(ROOT/'states'/f'{stem}.npz')['states'] for stem in stems]
        d['first_post_step_state_difference']=next((i for i,(a,b) in enumerate(zip(*states)) if not np.array_equal(a,b)),None)
    (ROOT/'diagnosis/first_divergence.json').write_text(json.dumps(divergence,indent=2))
    recovered=[dict(case=d['case'],seed=d['seed'],recovered=lookup[d['case'],f"candidate_{d['seed']}"]['success']) for d in divergence]
    summary=dict(aggregate=aggregate,paired=paired,lost_success_retention=recovered,retention_gate_passed=gate)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# E120: First action divergence and bounded correction','',f"Frozen E118: {a['success']}/72 successes, {a['collision']} collision episodes. Bounded E119 correction: {b['success']}/72 successes, {b['collision']} collision episodes. Strict retention gate {'PASSED' if gate else 'FAILED'}.",'',
        'E118 and E119 weights are frozen. Both propose DLS actions on the same currently measured history. Execute E118 action plus 25% of the E119-minus-E118 difference, capped at 0.001 rad per joint per step, with the original 0.04-rad action limit. Calibration and explicit probing are unchanged. This is an action-space residual from two learned estimators, not a bound on the response matrix, accumulated state drift, or physical safety. No training, parameter sweep, checkpoint selection, physics selector, geometry filter or privileged Jacobian.', '',
        'All 24 placements on four training bodies are evaluated for all three seeds (72 checkpoint/placement episodes, not 72 independent tasks). There are 16 fitting and eight held placements. These placements have been used in prior development; held means excluded from estimator fitting, not an untouched final test. No unseen-body rollout.', '',
        '## First divergence on the seven previously lost successes','',
        'Both full trajectories are exactly replayed for every lost-success pair. At each actually reached state, proposals from both models use identical observations, Gram/cross statistics and calibrated bias. Numerical divergence threshold: 1e-10 rad; material threshold: 1e-4 rad. Step indices are zero-based after calibration. First divergence identifies where behavior starts separating, not the sole cause of the later failure.', '',
        '| Seed | Case | First differing action | First >0.1 mrad | First collision under E119 | Restored by bound |','|---|---|---|---|---|---|']
    for d,r in zip(divergence,recovered):
        show=lambda key:'none' if d[key] is None else f"{d[key]['step']} / {d[key]['phase']}"
        lines.append(f"| {d['seed']} | {d['case']} | {show('first_numerical')} | {show('first_material')} | {show('first_collision')} | {r['recovered']} |")
    lines+=['','## Full episode comparisons','','reference = frozen E118; unbounded = E119; candidate = bounded correction. Analytical results are reused once per placement, not duplicated for neural seeds.','','| Method | Split | DOF | Success | Collisions |','|---|---|---|---:|---:|']
    for r in aggregate: lines.append(f"| {r['method']} | {r['group']} | {r['dof']} | {r['success']}/{r['n']} | {r['collision']} |")
    lines+=['','## Paired changes from frozen E118','', '```json',json.dumps(paired,indent=2),'```','',
        'All three seeds and all outcomes are retained. An improved total is insufficient for promotion if any baseline success is lost. The correction bound limits instantaneous action changes; it cannot ensure that later observations or phase timing match the frozen baseline. Full actions, histories, simulator states and labeled success/failure replays are preserved.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4));groups=('fitting','held')
    for i,(method,label) in enumerate((('reference','Frozen E118'),('unbounded','E119'),('candidate','Bounded correction'))):
        vals=[next(r for r in aggregate if r['method']==method and r['group']==g and r['dof']=='all') for g in groups];x=np.arange(2)+(i-1)*.23
        ax.bar(x,[r['success']/r['n']*100 for r in vals],.21,label=label)
        for xx,r in zip(x,vals):ax.text(xx,r['success']/r['n']*100+2,f"{r['success']}/{r['n']}",ha='center')
    ax.set_xticks(range(2),['Fitting placements','Held placements']);ax.set_ylim(0,110);ax.set_ylabel('Task success (%)');ax.legend(loc='upper center',ncol=3);ax.set_title('E120 · Frozen baseline with bounded action correction');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');s=p.read_text(encoding='utf-8');marker='## E120: bounded correction to frozen estimator';assert marker not in s
    s=s.replace('Revision 107 · 14 November 2026','Revision 108 · 14 November 2026',1);s+='\n'+marker+'\n\n'+lines[2]+'\n\nSeven lost-success pairs exactly replayed; fixed25% action residual capped at0.001 rad/joint/step, all72 episodes. Frozen weights, no selector or unseen-body evaluation. See `outputs/ease_bounded_history_v1/report.md`.\n';p.write_text(s,encoding='utf-8')
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print(json.dumps(summary,indent=2))


def verify():
    tasks,models,hashes=setup();rows=json.loads((ROOT/'records.json').read_text());checks=[]
    zero,_=evaluate(0,tasks[0],'reference_0',models['reference_0'],models['reference_0'],'bounded',save=False)
    assert zero==json.loads((ROOT/'trials/0_reference_0_frozen_bias.json').read_text())
    for seed in range(3):
        for success in (True,False):
            r=next((r for r in rows if r['method']==f'candidate_{seed}' and r['success']==success),None)
            if r is None:continue
            case=r['case'];stem=f'{case}_candidate_{seed}_frozen_bias';old=json.loads((ROOT/'trials'/f'{stem}.json').read_text());states=np.load(ROOT/'states'/f'{stem}.npz')['states'].copy();logs=json.loads((ROOT/'actions'/f'{stem}.json').read_text())
            result,newlogs=evaluate(case,tasks[case],f'candidate_{seed}',models[f'reference_{seed}'],models[f'correction_{seed}'],'bounded')
            assert result==old and newlogs==logs;np.testing.assert_array_equal(states,np.load(ROOT/'states'/f'{stem}.npz')['states']);checks.append(stem);print('verified',stem,flush=True)
    steps=0;maximum=0.;saturated=0
    for p in (ROOT/'actions').glob('*_candidate_*'):
        for r in json.loads(p.read_text()):
            if not r['active']:continue
            base=np.array(r['base_action']);correction=np.array(r['correction_action']);action=np.array(r['nominal_action']);expected=np.clip(base+np.minimum(np.maximum(ALPHA*(correction-base),-CAP),CAP),-.04,.04)
            np.testing.assert_array_equal(expected,action);delta=np.max(np.abs(action-base));assert delta<=CAP+1e-12;maximum=max(maximum,float(delta));saturated+=bool(np.any(np.abs(ALPHA*(correction-base))>CAP));steps+=1
    for p,h in hashes.items():assert fingerprint(p)==h
    (ROOT/'verification.json').write_text(json.dumps(dict(exact_replays=checks,zero_correction_exact_baseline=True,independently_checked_action_steps=steps,max_action_change_rad=maximum,steps_hitting_bound=saturated,checkpoint_hashes_unchanged=True),indent=2))


def finish():
    v=json.loads((ROOT/'verification.json').read_text());m=json.loads((ROOT/'videos/manifest.json').read_text());s=json.loads((ROOT/'summary.json').read_text());p=json.loads((ROOT/'protocol.json').read_text())
    for path,h in {**p['inputs'],**p['checkpoints']}.items():assert fingerprint(path)==h
    from .bounded_history_correction import OLD
    inherited_checks=0
    for row in json.loads((ROOT/'records.json').read_text()):
        if row['method'].startswith('candidate_'):continue
        original=row['method'].replace('unbounded_','candidate_')
        a=json.loads((OLD/'trials'/f"{row['case']}_{original}_frozen_bias.json").read_text())
        a['method']=row['method']
        b=json.loads((ROOT/'trials'/f"{row['case']}_{row['method']}_frozen_bias.json").read_text())
        assert a==b
        inherited_checks+=1
    assert inherited_checks==168
    for row in m['videos']:assert fingerprint(ROOT/'videos'/row['file'])==row['sha256']
    for path in (ROOT/'source').glob('*.py'):assert fingerprint(path)==fingerprint(Path(__file__).with_name(path.name))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete',new_episodes=72,paired_diagnostic_replays=14,verification_replays=len(v['exact_replays']),videos=len(m['videos']),retention_gate_passed=s['retention_gate_passed'],unseen_evaluated=False),indent=2))
    path=ROOT/'report.md';path.write_text(path.read_text()+f"\nFinal verification: {len(v['exact_replays'])} bounded episodes exactly replayed; zero-correction full episode exactly matches E118; {v['independently_checked_action_steps']} action steps independently checked against the prescribed bound; {len(m['videos'])} trace-verified videos. All168 inherited trials match their source, including original estimator telemetry. Bounded-run telemetry reports the frozen baseline response prediction error; applied actions and both same-history proposals are separately logged in actions/.\n\nInterpretation: all seven lost-success pairs differ on the first executed approach action and first resulting state, despite identical initial observations and history statistics. Maximum per-joint first-action differences span0.667–7.357 mrad. The bound restores four of those seven successes. Relative to E118, it gains six successes but loses four; all four losses are fitting placements with collisions. Held placements improve12/24 to15/24 without losing an E118 held success. This development result is promising but fails the no-regression criterion, so E118 remains the reference. Next isolate accumulated drift and phase transitions in the four remaining regressions before choosing another intervention.\n")
    (ROOT/'artifact_hashes.json').write_text(json.dumps({str(path.relative_to(ROOT)):fingerprint(path) for path in ROOT.rglob('*') if path.is_file() and path.name!='artifact_hashes.json'},indent=2))
    print((ROOT/'status.json').read_text())


if __name__=='__main__':
    import sys
    {'report':report,'verify':verify,'finish':finish}[sys.argv[1]]()
