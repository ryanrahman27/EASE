"""E122 full outcome report, causal gate audit, and media."""
import json,shutil
from pathlib import Path
import numpy as np
from .selective_history import ROOT,OLD,setup,evaluate,fingerprint

def build():
    s=Path(__file__).with_name('bounded_history_videos.py').read_text().replace('bounded_history_correction import ROOT','selective_history import ROOT').replace('E120','E122').replace('Bounded correction / Retention','Selective correction / Retention')
    needle='    selection=list(dict.fromkeys(selection));manifest='
    extra="    for d in json.loads((ROOT/'diagnosis/summary.json').read_text()):\n        for family in ('unbounded','candidate'):selection.append((d['case'],f\"{family}_{d['seed']}\",'frozen_bias'))\n"
    assert needle in s;s=s.replace(needle,extra+needle).replace('Outcome-selected diagnostic examples.','Includes all six E120 gains and four losses before and after selection. Outcome-selected diagnostic examples.')
    Path(__file__).with_name('selective_history_videos.py').write_text(s)

def report():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==240
    tasks=json.loads((ROOT/'cases.json').read_text());lookup={(r['case'],r['method']):r for r in rows};aggregate=[];changes=[]
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
    diagnostic=json.loads((ROOT/'diagnosis/summary.json').read_text())
    pairs=[dict(case=d['case'],seed=d['seed'],previous_gain=d['gain'],success=lookup[d['case'],f"candidate_{d['seed']}"]['success']) for d in diagnostic]
    retained=sum(d['success'] for d in pairs if d['previous_gain']);recovered=sum(d['success'] for d in pairs if not d['previous_gain'])
    gate=retained==6 and recovered==4 and not any(not r['gain'] for r in changes)
    summary=dict(aggregate=aggregate,paired_changes=changes,targeted_pairs=pairs,gains_retained=retained,regressions_recovered=recovered,retention_gate_passed=gate)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    headline='; '.join(f"{r['method']}: {r['success']}/{r['n']}, {r['collisions']} collision episodes" for r in aggregate if r['split']==r['dof']=='all' and r['method']!='analytical')
    lines=['# E122: Select corrections using preceding measured prediction errors','',headline+'.','',
        'reference = E118 frozen estimator; unbounded = E120 fixed per-step correction; candidate = E122 selective correction. E120 still has a per-step bound; unbounded means no additional selection in these inherited filenames.', '',
        'Keep both estimator weights frozen. Estimate both motion-response matrices before each action, then score them against realized measured joint and task displacement after that action. This is conditional response-model error using realized joint motion, not a command-conditioned forward rollout. Update squared-error EWMA with decay0.9. Use unchanged E120 correction for the first8 transitions, then permit it only if E119 preceding EWMA error is <=0.95 times E118 error; otherwise use the frozen E118 action on the current history. No phase reset. The explicit probe remains fixed but contributes evidence. Current-transition or future errors cannot affect the action that generated them. No contact, outcome, task identity, geometry, analytical fallback or physics rollout enters the gate.', '',
        'The threshold was fixed before inspecting the ten shadow-gate replays. All24 training-body placements and three seeds are evaluated;16 fitting and8 held placements. These are reused development placements, not an untouched test or unseen bodies. No fitting, threshold sweep or checkpoint selection. Lower motion prediction error is a proposed signal, not a proven measure of action value.', '',
        '## Help-versus-hurt diagnostic','',
        'All six E120 gains and four E120 losses relative to E118 are exactly replayed with original actions and a shadow gate. Statistics below use active steps before any prior forbidden collision. The transition causing the first collision is included; subsequent transitions are excluded. Held gains are inspected diagnostically and are not used to fit thresholds.', '',
        '| Case / seed | E120 outcome change | Shadow gate-on fraction | Mean E119/E118 prediction MSE | Selective success |','|---|---|---:|---:|---|']
    for d,p in zip(diagnostic,pairs):lines.append(f"| {d['case']} / {d['seed']} | {'gain' if d['gain'] else 'loss'} | {d['gate_on_fraction']:.3f} | {d['correction_error']/max(d['base_error'],1e-30):.3f} | {p['success']} |")
    lines+=['','## All task outcomes','','| Method | Split | DOF | Success | Collision episodes |','|---|---|---|---:|---:|']
    for r in aggregate:lines.append(f"| {r['method']} | {r['split']} | {r['dof']} | {r['success']}/{r['n']} | {r['collisions']} |")
    lines+=['',f'Retained {retained}/6 gains; recovered {recovered}/4 regressions. Strict retention gate {gate}.','','## Every paired change','','```json',json.dumps(changes,indent=2),'```','',
        'All outcomes, models, simulator trajectories, causal error ledgers and success/failure videos are preserved. E120 is not replaced unless selective correction retains its successes and repairs the regressions without introducing new losses. Shadow gate association and full closed-loop intervention answer different questions; changing actions also changes later observations, evidence and phase timing.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4))
    for i,(method,label) in enumerate((('reference','Frozen E118'),('unbounded','E120 correction'),('candidate','E122 selective'))):
        rr=[next(r for r in aggregate if r['method']==method and r['split']==sp and r['dof']=='all') for sp in ('fitting','held')];x=np.arange(2)+(i-1)*.23;ax.bar(x,[100*r['success']/r['n'] for r in rr],.21,label=label)
        for xx,r in zip(x,rr):ax.text(xx,100*r['success']/r['n']+2,f"{r['success']}/{r['n']}",ha='center')
    ax.set_xticks([0,1],['Fitting placements','Held placements']);ax.set_ylim(0,110);ax.set_ylabel('Task success (%)');ax.legend(ncol=3,loc='upper center');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4));ax.bar(range(10),[d['gate_on_fraction'] for d in diagnostic],color=['#21895c' if d['gain'] else '#bd523a' for d in diagnostic]);ax.set_xticks(range(10),[f"{d['case']}/{d['seed']}\n{'gain' if d['gain'] else 'loss'}" for d in diagnostic]);ax.set_ylim(0,1);ax.set_ylabel('Shadow gate-on fraction');ax.set_title('Recent prediction-error gate on unchanged E120 trajectories');fig.tight_layout();fig.savefig(ROOT/'diagnosis.png',dpi=160);plt.close(fig)
    path=Path(__file__).with_name('TECHNICAL_REPORT.md');s=path.read_text(encoding='utf-8');marker='## E122: selective measured-history correction';assert marker not in s;s=s.replace('Revision 109 · 14 November 2026','Revision 110 · 14 November 2026',1);s+='\n'+marker+'\n\n'+headline+f'. Retained{retained}/6 gains, recovered{recovered}/4 regressions; retention gate{gate}.\n\nTen exact shadow replays and72 causal-gate episodes; no unseen bodies or physics selector. Details: `outputs/ease_selective_history_v1/report.md`.\n';path.write_text(s,encoding='utf-8')
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(headline);print(json.dumps(dict(retained=retained,recovered=recovered,changes=changes,gate=gate),indent=2))

def verify():
    tasks,models,hashes=setup();rows=json.loads((ROOT/'records.json').read_text());checks=[]
    for seed in range(3):
        for outcome in (True,False):
            r=next((r for r in rows if r['method']==f'candidate_{seed}' and r['success']==outcome),None)
            if r is None:continue
            case=r['case'];stem=f'{case}_candidate_{seed}_frozen_bias';old=json.loads((ROOT/'trials'/f'{stem}.json').read_text());states=np.load(ROOT/'states'/f'{stem}.npz')['states'].copy();oldgate=json.loads((ROOT/'actions'/f'{case}_candidate_{seed}_gate.json').read_text())
            new,_,ledger=evaluate(case,tasks[case],f'candidate_{seed}',models[f'reference_{seed}'],models[f'correction_{seed}']);assert new==old and ledger==oldgate;np.testing.assert_array_equal(states,np.load(ROOT/'states'/f'{stem}.npz')['states']);checks.append(stem);print('exact',stem,flush=True)
    steps=0;enabled=0
    for p in (ROOT/'actions').glob('*_candidate_*_frozen_bias.json'):
        actions=json.loads(p.read_text());ledger=json.loads(p.with_name(p.name.replace('_frozen_bias','_gate')).read_text());scores=np.zeros(2);assert len(actions)==len(ledger)
        for i,(a,g) in enumerate(zip(actions,ledger)):
            allow=bool(i<8 or scores[1]<=.95*scores[0]);assert allow==g['allow'];np.testing.assert_array_equal(scores,g['previous_scores'])
            errors=np.array([np.sum((np.array(g[k])-g['dx'])**2) for k in ('base_prediction','correction_prediction')]);np.testing.assert_array_equal(errors,g['errors']);scores=.9*scores+.1*errors
            if not a['active']:continue
            base=np.array(a['base_action']);correction=np.array(a['correction_action']);expected=np.clip(base+np.clip(.25*(correction-base),-.001,.001),-.04,.04) if allow else base
            np.testing.assert_array_equal(expected,a['nominal_action']);assert np.max(np.abs(expected-base))<=.001+1e-12;steps+=1;enabled+=allow
    for p,h in hashes.items():assert fingerprint(p)==h
    (ROOT/'verification.json').write_text(json.dumps(dict(exact_replays=checks,causal_action_steps_checked=steps,enabled_steps=enabled,unchanged_weights=True),indent=2))

def finish():
    p=json.loads((ROOT/'protocol.json').read_text());v=json.loads((ROOT/'verification.json').read_text());s=json.loads((ROOT/'summary.json').read_text());m=json.loads((ROOT/'videos/manifest.json').read_text())
    for path,h in {**p['inputs'],**p['checkpoints']}.items():assert fingerprint(path)==h
    inherited=0
    for r in json.loads((ROOT/'records.json').read_text()):
        if r['method'].startswith('candidate_'):continue
        original=r['method'].replace('unbounded_','candidate_')
        old=json.loads((OLD/'trials'/f"{r['case']}_{original}_frozen_bias.json").read_text());old['method']=r['method']
        assert old==json.loads((ROOT/'trials'/f"{r['case']}_{r['method']}_frozen_bias.json").read_text());inherited+=1
    assert inherited==168
    for r in m['videos']:assert fingerprint(ROOT/'videos'/r['file'])==r['sha256']
    for path in (ROOT/'source').glob('*.py'):assert fingerprint(path)==fingerprint(Path(__file__).with_name(path.name))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete',new_episodes=72,shadow_replays=10,verification_replays=len(v['exact_replays']),videos=len(m['videos']),retention_gate_passed=s['retention_gate_passed'],unseen_evaluated=False),indent=2))
    path=ROOT/'report.md';path.write_text(path.read_text()+f"\nVerified {v['causal_action_steps_checked']} causal gate/action decisions, {len(v['exact_replays'])} exact candidate replays, all168 inherited trial contents, unchanged input/model hashes, and {len(m['videos'])} trace-verified videos.\n\nDecision: keep E120. This gate produced51/72 versus52/72, with21 versus18 collision episodes. It retained3/6 prior gains and recovered1/4 regressions. Across all pairs versus E120 it gained six successes and lost seven. In the shadow diagnosis, the harmful case22/seed0 has lower conditional response error for E119 and95% gate admission, while gain cases admit corrections on only15–54% of clean active steps. The particular recent-error criterion therefore does not reliably identify beneficial corrections. These episode-level associations do not label each individual correction as causal help or harm. Next isolate correction value with matched-state action/phase interventions on fitting cases, preserving held placements for checks, rather than tuning this threshold against known outcomes.\n")
    path=ROOT/'videos/index.html';path.write_text(path.read_text().replace('</h1>','</h1><p>reference = frozen E118; unbounded = E120 per-step correction; candidate = E122 selective correction.</p><p><a href="../diagnosis.png">Help-versus-hurt diagnosis</a></p>',1))
    (ROOT/'artifact_hashes.json').write_text(json.dumps({str(path.relative_to(ROOT)):fingerprint(path) for path in ROOT.rglob('*') if path.is_file() and path.name!='artifact_hashes.json'},indent=2));print((ROOT/'status.json').read_text())

if __name__=='__main__':
    import sys
    {'build':build,'report':report,'verify':verify,'finish':finish}[sys.argv[1]]()
