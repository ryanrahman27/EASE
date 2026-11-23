"""E134 prefix/action audit and outcome media selection."""
import shutil
from pathlib import Path
import numpy as np
from .one_step_response_control import ROOT, PRIOR
from .abstaining_task_value_gate import read,save,fingerprint,COLLECT,DATA,PHASES
from .bounded_history_correction import bounded_action

def run():
    summary=read(ROOT/'summary.json');rows=read(ROOT/'records.json');tasks=read(ROOT/'cases.json');primary=read(PRIOR/'primary.json');expected={(r['case'],r['seed']):r['phase'] for r in primary['episodes']};checks=[]
    for r in rows:
        case,seed=r['case'],r['seed'];stem=f'{case}_candidate_{seed}_frozen_bias';onstem=f'{case}_on_{seed}_frozen_bias'
        trial=read(ROOT/'trials'/f'{stem}.json');on=read(COLLECT/'trials'/f'{onstem}.json');assert trial['calibration']==on['calibration'] and trial['initial_estimator']==on['initial_estimator']
        assert r['chosen']==expected[case,seed]
        logs=read(ROOT/'actions'/f'{stem}.json');states=np.load(ROOT/'states'/f'{stem}.npz')['states'];oldstates=np.load(COLLECT/'states'/f'{onstem}.npz')['states'];first=next((i for i,a in enumerate(logs) if a['phase']==r['chosen']),None)
        stop=len(states) if first is None else first
        np.testing.assert_array_equal(states[:stop],oldstates[:stop]);assert trial['trace'][:stop]==on['trace'][:stop]
        if first is None:
            before=dict(on);before['method']=trial['method'];assert trial==before
        for i,a in enumerate(logs):
            if a['phase']=='probe':continue
            wanted=np.asarray(a['base_action']) if i==first else bounded_action(np.array(a['base_action']),np.array(a['correction_action']))
            np.testing.assert_array_equal(wanted,a['nominal_action'])
        checks.append(dict(case=case,seed=seed,prefix_steps=stop,one_off_step=first))
    for path,h in {**read(ROOT/'protocol.json')['checkpoints'],**read(ROOT/'protocol.json')['inputs'],**read(ROOT/'frozen.json')}.items():assert fingerprint(path)==h
    save(ROOT/'verification.json',dict(exact_prefixes=len(checks),all_interventions_one_step=True,all_remaining_actions_E120=True,choices_match_frozen_OOF_rule=True,prior_inputs_unchanged=True,checks=checks))
    selected=[]
    for dof in (6,7):
        for gain in (True,False):
            changes=[r for r in summary['fitting']['changes'] if r['after']==gain and len(tasks[r['case']]['home'])==dof]
            for c in changes[:2]:selected.append(next(r for r in rows if (r['case'],r['seed'])==(c['case'],c['seed'])))
    for success in (True,False):
        if not any(r['success']==success for r in selected):selected.append(next(r for r in rows if r['success']==success))
    media=[]
    for r in selected:
        media.append(r);case,seed=r['case'],r['seed'];stem=f'{case}_on_{seed}_frozen_bias'
        for folder,exts in (('trials',('json',)),('states',('xml','npz'))):
            for ext in exts:shutil.copy2(COLLECT/folder/f'{stem}.{ext}',ROOT/folder/f'{stem}.{ext}')
        on=read(COLLECT/'trials'/f'{stem}.json');media.append(dict(case=case,method=f'on_{seed}',mode='frozen_bias',success=on['success'],collision=on['collision']))
    save(ROOT/'media_records.json',media)
    s=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','one_step_response_control import ROOT').replace("ROOT/'records.json'","ROOT/'media_records.json'").replace('E128','E134').replace('Task-value ranking diagnosis','One-step response control').replace('Diagnosis only; no new policy.','Fresh one-step intervention rollouts versus E120; family-excluded scalar predictor.')
    Path(__file__).with_name('one_step_response_videos.py').write_text(s)
    f=summary['fitting'];report=f'''# E134: one-step intervention matching the prediction horizon

Same E133 scalar response estimator and fixed10-micrometer margin. At the first accepted clean phase entry, disable the E120 correction for exactly one action, then resume E120. No privileged object/contact state or physics selector. Four fitting-family excluded models;144 complete fresh fitting rollouts.

E120: {f['baseline_success']}/144. Candidate: {f['success']}/144, {f['gains']} rescues and {f['losses']} lost successes. E120 remains immutable. Original held placements have not been evaluated in this experiment.

All144 pre-intervention state/trace prefixes match E120 exactly. Every executed nominal action was checked: exactly the selected action disables correction; every other non-probe action uses unchanged E120 bounds. Choices match the frozen E133 OOF rule. Original inputs/checkpoints remain unchanged.

These are development validation results on four known training bodies, not unseen-body transfer. The fitting-family splits have guided multiple adaptive experiments. A gain here still requires frozen held evaluation and a fresh independent benchmark.

Representative rescues/losses by joint count and success/failure videos are preserved alongside all144 full trials, actions and states.
'''
    (ROOT/'report.md').write_text(report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(['E120','E133 eight-step','E134 one-step'],[97,101,f['success']]);ax.set_ylim(0,144);ax.set_ylabel('Successes /144 fitting episodes');ax.set_title(f"E134: {f['gains']} rescues, {f['losses']} losses")
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');text=p.read_text(encoding='utf-8');marker='## E134: one-step response control';assert marker not in text
    p.write_text(text+'\n'+marker+f"\n\nFresh144 fitting rollouts, frozen E133 family-excluded scalar response and10um threshold; exactly one off action. Candidate{f['success']}/144 vs E12097, {f['gains']} gains/{f['losses']} losses. All prefixes and action restoration verified; held untouched. Report: `outputs/ease_one_step_response_control_v1/report.md`.\n",encoding='utf-8')
    for name in ('one_step_response_control.py','one_step_response_artifacts.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    print('Verified144 full episodes; prepared',len(media),'videos',flush=True)

if __name__=='__main__':run()
