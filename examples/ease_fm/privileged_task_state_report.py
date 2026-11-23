"""E131 explicitly privileged-state report, feature timing audit and provenance."""
import json
import shutil
from pathlib import Path
import numpy as np
from .privileged_task_state import ROOT, DATA, PRIOR, MEASURED, COLLECT, read, save, fingerprint


def quality():
    x=np.load(ROOT/'features.npz')['x'];meta=read(DATA/'gate_rows.json');tasks=read(ROOT/'cases.json')
    rotations=x[:,165:174].reshape(-1,3,3)
    np.testing.assert_allclose(rotations.transpose(0,2,1)@rotations,np.broadcast_to(np.eye(3),rotations.shape),atol=2e-6,rtol=0)
    np.testing.assert_allclose(np.linalg.det(rotations),1,atol=2e-6,rtol=0)
    assert set(np.unique(x[:,177:182])).issubset({0.,1.})
    np.testing.assert_array_equal(x[:,181],x[:,177]*x[:,178])
    cache={};checked=0;hashes={}
    for i,r in enumerate(meta):
        case,seed,phase=r['case'],r['seed'],r['phase'];key=(case,seed)
        if key not in cache:
            bp=COLLECT/'branches'/f'{case}_on_{seed}.json';tp=COLLECT/'trials'/f'{case}_on_{seed}_frozen_bias.json'
            cache[key]=(read(bp),read(tp));hashes[str(bp)]=fingerprint(bp);hashes[str(tp)]=fingerprint(tp)
        branch,trial=cache[key];entry=branch['entries'][phase]
        assert bool(x[i,180])==entry['metadata']['grasped']
        position=x[i,139:142].astype(float)+x[i,162:165]
        np.testing.assert_allclose(x[i,174:177]+position,np.r_[tasks[case]['target'],.215],atol=1e-7,rtol=0)
        if entry['step']>0:
            np.testing.assert_allclose(position,trial['trace'][entry['step']-1]['object'],atol=1e-7,rtol=0);checked+=1
    result=dict(rotation_valid=True,contact_flags_binary=True,bilateral_contact_consistent=True,grasp_flag_preaction=True,object_matches_previous_observation_rows=checked,
        rows=len(x),std_privileged_features=x[:,162:].std(0).tolist(),contact_and_grasp_positive_rows=x[:,177:182].sum(0).tolist())
    save(ROOT/'feature_quality.json',result);save(ROOT/'quality_input_hashes.json',hashes)
    print(json.dumps(result,indent=2))


def report():
    s=read(ROOT/'summary.json');p=read(ROOT/'protocol.json');primary=read(ROOT/'primary_outcomes.json');grids=read(ROOT/'validation_grids.json');quality=read(ROOT/'feature_quality.json')
    y=np.load(ROOT/'features.npz')['y'];heads={}
    for name,path in (('two_head',PRIOR/'oof_predictions.npz'),('measured',MEASURED/'richer_oof.npz'),('privileged',ROOT/'privileged_oof.npz')):
        prob=np.load(path)['probabilities'].mean(0);heads[name]=dict(accuracy=np.mean((prob>.5)==y,axis=0).tolist(),brier=np.mean((prob-y)**2,axis=0).tolist())
    save(ROOT/'head_metrics.json',heads)
    lines=['# E131: PRIVILEGED simulator task-state information upper bound','',
        '**Privileged simulator ground truth. Not a measured-sensor, real-hardware, or deployable-policy result.** This is an upper bound on the tested information access, not a mathematical upper bound on achievable performance. A finite learned model can fail to exploit extra information.', '',
        'The privileged variant improves decisive ranking accuracy modestly but does not improve fixed-rule task success over E120. All zero-loss selections abstain. E120 remains unchanged.', '',
        '## Fixed experiment','']
    for key in ('design','initialization','features','timing','normalization','primary','outcomes','baseline'):lines += [f"**{key}:** {p[key]}",'']
    lines += ['The two-head BCE objective, hidden widths64/32, three seeds,1500 AdamW updates, data weighting, first-phase intervention rule and four ancestral fitting-family folds are unchanged. No original held-placement tuning or evaluation. The input layer grows, so this is a feature-access ablation with corresponding input parameters, not a claim of identical parameter count.', '',
        '## Privileged input definition and causal timing','', '| Columns | Simulator-only input | Source |','|---|---|---|',
        '| 162–164 | Object position minus tool position | Pre-action body/site pose |',
        '| 165–173 | World object rotation matrix | Pre-action body orientation |',
        '| 174–176 | Destination position minus object position | Known destination XY and landing height0.215m, pre-action object |',
        '| 177–179 | Left-pad/object, right-pad/object, table/object contacts | Instantaneous contact pairs at the snapshot |',
        '| 180 | Has grasped before this decision | Cumulative grasp flag from phase-entry metadata |',
        '| 181 | Current bilateral pad/object contact | Both instantaneous finger contact flags |', '',
        'Measured columns0–161 remain exact E130 copies. The extra state is reconstructed from E120-on integration snapshots before the candidate action. It never reads final success, final grasp/lift outcomes, future states, collision/bad flag, privileged Jacobians, or physical parameters into the network. Instantaneous contacts differ from contacts accumulated across a control interval.', '',
        f"Quality checks: all1108 rotations are valid; all contact/grasp values are binary; bilateral contact matches both pad flags; grasp values match pre-action metadata. In{quality['object_matches_previous_observation_rows']} noninitial phase rows, reconstructed object position matches the previous saved observation. Initial entries use the pre-action snapshot directly. Positive counts for left/right/table/grasp-history/bilateral inputs: {quality['contact_and_grasp_positive_rows']}.", '',
        '## Out-of-fold prediction quality','', '| Model | Advantage MSE | Decisive-pair MSE | Correct decisive sign | Beneficial reversed /65 | Harmful reversed /54 |','|---|---:|---:|---:|---:|---:|']
    for name,r in s['metrics'].items():lines.append(f"| {name} | {r['mse']:.4f} | {r['decisive_mse']:.4f} | {r['decisive_sign_accuracy']:.1%} | {r['beneficial_reversed']} | {r['harmful_reversed']} |")
    lines += ['', '| Model | On/off head accuracy | On/off Brier |','|---|---|---|']
    for name,r in heads.items():lines.append(f"| {name} | {r['accuracy'][0]:.4f}/{r['accuracy'][1]:.4f} | {r['brier'][0]:.4f}/{r['brier'][1]:.4f} |")
    lines += ['', 'Decisive sign accuracy rises from52.1% measured to56.3% privileged, while total advantage MSE remains around0.20. With119 decisive and989 neutral, dependent phase-pair examples, this modest sign change does not establish robust policy improvement.', '',
        '## Fixed-rule complete-task outcome reconstruction','',
        'Minimum advantage0.1, maximum ensemble spread2.0; all three predictions must pass. First accepted clean phase gets one up-to8-step correction-off pulse, then E120 resumes. The144 expanded fitting-validation episodes differ from the original72 benchmark. Outcomes come from existing matched full-task continuations, not fresh online-policy rollout or a new held benchmark. Post-collision starts are omitted because success cannot recover.', '',
        '| Method | Success /144 | Gains vs E120 | Losses vs E120 | Switches |','|---|---:|---:|---:|---:|','| E120 | 97 | 0 | 0 | 0 |']
    for name,r in s['primary'].items():lines.append(f"| {name} | {r['success']} | {r['gains']} | {r['losses']} | {r['switches']} |")
    lines += ['', '| Privileged case/seed | Selected phase | E120 success | Privileged success |','|---|---|---|---|']
    for r in primary['privileged']['episodes']:
        if r['success']!=r['baseline_success']:lines.append(f"| {r['case']}/{r['seed']} | {r['phase']} | {r['baseline_success']} | {r['success']} |")
    lines += ['', 'Privileged input rescues50/1 but loses1/2 and7/2. It does not preserve all E120 successes, and it does not retain the measured model\'s22/2 and38/2 rescues. No baseline replacement is warranted.', '',
        '## Fold and seed variability','', '| Model | Fold | MSE | Correct decisive sign |','|---|---:|---:|---:|']
    for r in s['fold_metrics']:lines.append(f"| {r['method']} | {r['fold']} | {r['mse']:.4f} | {r['decisive_sign_accuracy']:.1%} |")
    lines += ['', '| Model | Gate seed | MSE | Correct decisive sign |','|---|---:|---:|---:|']
    for r in s['seed_metrics']:lines.append(f"| {r['method']} | {r['gate_seed']} | {r['mse']:.4f} | {r['decisive_sign_accuracy']:.1%} |")
    lines += ['', '## Same secondary selection grid','', '| Min advantage | Max spread | Original gains/losses/switches | Measured | Privileged |','|---:|---:|---|---|---|']
    for rows in zip(grids['two_head'],grids['measured'],grids['privileged']):
        r=rows[0]['rule'];assert all(q['rule']==r for q in rows)
        cells=[f"{q['gains']}/{q['losses']}/{q['switches']}" for q in rows]
        lines.append(f"| {r['min_advantage']} | {r['max_spread']} | {' | '.join(cells)} |")
    lines += ['', 'All three zero-E120-loss selections choose full abstention. Selected scores use fitting validation for tuning and are not an independent final test. There is no additional full-fitting model, original held-placement retest, or deployment.', '',
        '## Interpretation','',
        'Providing exact object pose and contact/grasp state is insufficient to fix this gate under the tested architecture, objective, data and training budget. The result weakens the hypothesis that missing these inputs alone explains the failures. It does not prove that object state is unimportant or establish a performance ceiling for privileged control.', '',
        'Before further sensing or confidence-rule work, test whether the value learner can generalize a simpler target with denser paired supervision under the same family splits. Keep information access explicit and change one factor at a time. Preserve E120 as the task-success baseline.', '']
    (ROOT/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    names=['two_head','measured','privileged'];labels=['E120','Original','Measured','PRIVILEGED'];vals=[97]+[s['primary'][n]['success'] for n in names]
    axes[0].bar(labels,vals,color=['#777777','#2779b2','#db922d','#9255a0'])
    for i,v in enumerate(vals):axes[0].text(i,v+2,f'{v}/144',ha='center')
    axes[0].set_ylim(0,144);axes[0].set_ylabel('Reconstructed task successes');axes[0].set_title('Fitting-family validation / fixed rule')
    xx=np.arange(3)
    for j,(key,label) in enumerate((('gains','Rescued failures'),('losses','Lost successes'))):
        yy=[s['primary'][n][key] for n in names];axes[1].bar(xx+(j-.5)*.32,yy,.3,label=label)
        for q,v in zip(xx+(j-.5)*.32,yy):axes[1].text(q,v+.08,str(v),ha='center')
    axes[1].set_xticks(xx,labels[1:]);axes[1].set_ylim(0,3.5);axes[1].legend();axes[1].set_title('Paired changes versus E120')
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    path=Path(__file__).with_name('TECHNICAL_REPORT.md');text=path.read_text(encoding='utf-8');marker='## E131: privileged object/contact-state information upper bound';assert marker not in text
    text=text.replace('Revision 118 · 15 November 2026','Revision 119 · 15 November 2026',1)
    text+='\n'+marker+'\n\nExplicit simulator-only inputs:20 object-pose/contact/pre-action grasp features appended to measured162. Fixed two-head BCE and ancestral folds. Privileged decisive sign56.3% vs measured52.1%, but fixed-rule reconstructed success96/144 (1 gain/2 losses), measured97/144 (2/2), E12097/144. All zero-loss selections abstain. Extra information is not a mathematical performance bound. No held-placement retest, deployment or E120 change. Full report: `outputs/ease_privileged_task_state_v1/report.md`.\n'
    path.write_text(text,encoding='utf-8')
    for name in ('privileged_task_state.py','privileged_task_state_features.py','build_privileged_task_state.py','privileged_task_state_report.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    print(json.dumps(heads,indent=2))


def finish():
    v=read(ROOT/'verification.json');assert v['privileged_features_reconstructed'] and v['measured_features_unchanged']
    for path,h in {**read(ROOT/'protocol.json')['inputs'],**read(ROOT/'media_input_hashes.json'),**read(ROOT/'feature_provenance.json'),**read(ROOT/'quality_input_hashes.json')}.items():assert fingerprint(path)==h
    for path,h in read(ROOT/'frozen_training.json').items():assert fingerprint(ROOT/path)==h
    for path in (ROOT/'source').glob('*.py'):assert fingerprint(path)==fingerprint(Path(__file__).with_name(path.name))
    m=read(ROOT/'videos'/'manifest.json')
    for r in m['videos']:assert fingerprint(ROOT/'videos'/r['file'])==r['sha256']
    save(ROOT/'status.json',dict(status='complete',privileged_simulator_inputs=True,deployable_sensing_claim=False,new_fold_models=12,inputs=182,privileged_features=20,new_simulation_rollouts=0,held_placement_evaluated=False,E120_unchanged=True,zero_loss_rules_abstain=True,videos=len(m['videos'])))
    p=ROOT/'report.md';p.write_text(p.read_text()+f"\nVerification: all1108 privileged rows independently reconstructed; measured162 unchanged; valid rotations and binary contact flags; pre-action grasp metadata and previous-observation pose timing checked. Fitting-only normalization, ancestral isolation, model predictions and outcome reconstruction verified. E120 and both control models/data unchanged. Saved{len(m['videos'])} trace-verified videos explicitly labeled PRIVILEGED.\n")
    save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
    print(json.dumps(read(ROOT/'status.json'),indent=2))


if __name__=='__main__':
    import sys
    {'quality':quality,'report':report,'finish':finish}[sys.argv[1]]()
