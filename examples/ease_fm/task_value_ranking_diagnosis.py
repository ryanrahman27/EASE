"""E128: diagnosis only, on excluded fitting families; no gate selection."""
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from .abstaining_task_value_gate import DATA, COLLECT, OLD, SEEDS, Gate, read, save, family, probabilities, fingerprint
from .history_similarity_gate import group

ROOT=Path('outputs/ease_task_value_ranking_diagnosis_v1')
PRIOR=Path('outputs/ease_abstaining_task_value_gate_v1')


def load_fold(fold):
    directory=PRIOR/'models'/f'fold_{fold}'
    n=np.load(directory/'normalization.npz')
    models=[]
    for seed in SEEDS:
        m=Gate();m.load_state_dict(torch.load(directory/f'{seed}.pt',weights_only=True));m.eval();models.append(m)
    return models,n['mean'],n['scale'],n['training_indices']


def analyze():
    torch.set_num_threads(4)
    ROOT.mkdir(exist_ok=False)
    for d in ('source','trials','states','paired'):(ROOT/d).mkdir()
    tasks,meta=read(DATA/'cases.json'),read(DATA/'gate_rows.json')
    f=np.load(DATA/'gate_dataset.npz');x,y=f['x'],f['y']
    o=np.load(PRIOR/'oof_predictions.npz');p,folds=o['probabilities'],o['row_fold']
    assert all(not tasks[r['case']]['validation'] for r in meta)
    inputs=[DATA/'gate_dataset.npz',DATA/'gate_rows.json',DATA/'cases.json',COLLECT/'records.json',PRIOR/'oof_predictions.npz',PRIOR/'validation_grid.json',PRIOR/'frozen_selection.json',OLD/'records.json',OLD/'summary.json']
    inputs += [p for p in (PRIOR/'models').rglob('*') if p.is_file()]
    save(ROOT/'protocol.json',dict(experiment='E128',date='2026-11-15',inputs={str(p):fingerprint(p) for p in inputs},
        scope='Diagnosis only. No new model, confidence rule, policy selection, or held-placement retest. E120 unchanged. Use E126 out-of-fold predictions on E125 fitting-only histories; all ancestral families excluded from their fold training.',
        selection='All first-choice outcome changes of fixed E126 min-advantage0.1/spread2 rule, plus strongest reversed harmful/beneficial rankings, a correct harmful ranking, and strongest correct benefit. Outcome-selected diagnostic examples, not a benchmark.',
        neighbors='Three nearest distinct training families in same phase/DOF/local context using fold-normalized clipped full, history-only, or state/action features. Labels not used to choose neighbors.',
        swaps='Replace one feature block with the closest full-feature training neighbor; neural sensitivity only. Such hybrid features may be off-manifold. Not a physically executed intervention and not causal attribution.',
        simulations='Use already executed paired E120-on and up-to8-step correction-off then E120 continuations. Recheck prefix signatures and all intervention snapshot arrays. Save original trajectories and trace-verified videos. No new rollouts.'))
    delta=y[:,1]-y[:,0];mean=p.mean(0);adv=mean[:,1]-mean[:,0];minimum=(p[:,:,1]-p[:,:,0]).min(0)
    def metrics(ids):
        ids=np.asarray(ids)
        return dict(rows=len(ids),on_accuracy=float(np.mean((mean[ids,0]>.5)==y[ids,0])),off_accuracy=float(np.mean((mean[ids,1]>.5)==y[ids,1])),
            on_brier=float(np.mean((mean[ids,0]-y[ids,0])**2)),off_brier=float(np.mean((mean[ids,1]-y[ids,1])**2)),
            beneficial=int(np.sum(delta[ids]>0)),harmful=int(np.sum(delta[ids]<0)),neutral=int(np.sum(delta[ids]==0)),
            beneficial_rank_reversed=int(np.sum((delta[ids]>0)&(adv[ids]<0))),harmful_rank_reversed=int(np.sum((delta[ids]<0)&(adv[ids]>0))),
            consensus_beneficial=int(np.sum((delta[ids]>0)&(minimum[ids]>.1))),consensus_harmful=int(np.sum((delta[ids]<0)&(minimum[ids]>.1))))
    aggregate=metrics(np.arange(len(x)))
    phase_metrics={phase:metrics([i for i,r in enumerate(meta) if r['phase']==phase]) for phase in sorted({r['phase'] for r in meta})}
    model_audits=[]
    for fold in range(4):
        models,mu,scale,ids=load_fold(fold);val=np.flatnonzero(folds==fold)
        np.testing.assert_array_equal(ids,np.flatnonzero(folds!=fold))
        assert not {family(tasks,meta[i]['case']) for i in ids}&{family(tasks,meta[i]['case']) for i in val}
        np.testing.assert_array_equal(mu,x[ids].mean(0));np.testing.assert_array_equal(scale,np.maximum(x[ids].std(0),1e-6))
        np.testing.assert_array_equal(probabilities(models,mu,scale,x[val]),p[:,val])
        train_p=probabilities(models,mu,scale,x[ids])
        model_audits.append(dict(fold=fold,training_rows=len(ids),validation_rows=len(val),training_head_accuracy=float(np.mean((train_p>.5)==y[None,ids])),validation=metrics(val)))
    grid=read(PRIOR/'validation_grid.json')
    fixed=next(r for r in grid if r['rule']==dict(min_advantage=.1,max_spread=2.))
    index={(r['case'],r['seed'],r['phase']):i for i,r in enumerate(meta)}
    chosen=[];reasons={}
    def add(i,reason):
        i=int(i)
        if i not in chosen:chosen.append(i)
        reasons.setdefault(i,[]).append(reason)
    for r in fixed['episodes']:
        if r['success']!=r['baseline_success']:add(index[r['case'],r['seed'],r['phase']],'first-choice outcome change')
    for mask,values,reason in ((delta<0,adv,'strongest harmful ranking reversal'),(delta>0,-adv,'strongest beneficial ranking reversal'),(delta<0,-adv,'strongest correctly ranked harmful pulse'),(delta>0,adv,'strongest correctly ranked beneficial pulse')):
        ids=np.flatnonzero(mask)
        if len(ids):add(ids[np.argmax(values[ids])],reason)
    featuresets=dict(full=np.arange(139),history=np.arange(33,124),state_action=np.r_[np.arange(16,33),np.arange(124,138)])
    blocks=dict(joint_position=(16,23),bias=(23,30),goal_error=(30,33),history=(33,124),proposed_actions=(124,138))
    details=[];video_selection=[];input_hashes={}
    for i in chosen:
        r=meta[i];case,seed,phase=r['case'],r['seed'],r['phase'];fold=int(folds[i])
        models,mu,scale,train=load_fold(fold)
        eligible=np.array([j for j in train if group(x[j])==group(x[i])],int)
        assert len(eligible)>0
        z=np.clip((x-mu)/scale,-10,10)
        neighbors={}
        for name,dims in featuresets.items():
            dist=np.sqrt(np.mean((z[eligible][:,dims]-z[i,dims])**2,axis=1));seen=set();rows=[]
            for k in np.argsort(dist,kind='stable'):
                j=int(eligible[k]);fam=family(tasks,meta[j]['case'])
                if fam in seen:continue
                seen.add(fam)
                rows.append(dict(row=j,case=meta[j]['case'],seed=meta[j]['seed'],family=fam,distance=float(dist[k]),labels=y[j].tolist(),model_probability=probabilities(models,mu,scale,x[j]).mean(0).tolist()))
                if len(rows)==3:break
            neighbors[name]=rows
        nearest=neighbors['full'][0]['row'];swaps=[]
        baseprob=probabilities(models,mu,scale,x[i]);baseadv=float(np.mean(baseprob[:,1]-baseprob[:,0]))
        for name,(a,b) in blocks.items():
            hybrid=x[i].copy();hybrid[a:b]=x[nearest,a:b]
            h=probabilities(models,mu,scale,hybrid);hadv=float(np.mean(h[:,1]-h[:,0]))
            swaps.append(dict(block=name,mean_advantage=hadv,change=hadv-baseadv,probabilities=h.tolist()))
        on_method,off_method=f'on_{seed}',f'off_{phase}_{seed}'
        onbranch=read(COLLECT/'branches'/f'{case}_{on_method}.json');offbranch=read(COLLECT/'branches'/f'{case}_{off_method}.json');step=offbranch['intervention']
        assert step==onbranch['entries'][phase]['step']
        assert onbranch['signatures'][:step+1]==offbranch['signatures'][:step+1]
        snapshots=[]
        for method in (on_method,off_method):
            path=COLLECT/'snapshots'/f'{case}_{method}_{phase}.npz';snapshots.append(np.load(path));input_hashes[str(path)]=fingerprint(path)
        for key in snapshots[0].files:np.testing.assert_array_equal(snapshots[0][key],snapshots[1][key])
        trajectories=[]
        for method,branch in ((on_method,onbranch),(off_method,offbranch)):
            stem=f'{case}_{method}_frozen_bias';trial=read(COLLECT/'trials'/f'{stem}.json')
            actions=read(COLLECT/'actions'/f'{stem}.json')
            first_collision=next((dict(step=k,phase=t['phase'],bad=t['bad']) for k,t in enumerate(trial['trace']) if t['bad']),None)
            trajectories.append(dict(method=method,success=trial['success'],collision=trial['collision'],grasped=trial['grasped'],lifted=trial['lifted'],final_object=trial['final_object'],first_collision=first_collision,
                pulse_metrics=branch['metrics'][step:step+8],first_command=actions[step]['executed_command']))
            for folder,exts in (('trials',('json',)),('states',('npz','xml'))):
                for ext in exts:
                    src=COLLECT/folder/f'{stem}.{ext}';shutil.copy2(src,ROOT/folder/src.name);input_hashes[str(src)]=fingerprint(src)
            for path in (COLLECT/'actions'/f'{stem}.json',COLLECT/'branches'/f'{case}_{method}.json'):input_hashes[str(path)]=fingerprint(path)
            video_selection.append(dict(case=case,method=method,mode='frozen_bias',success=trial['success'],collision=trial['collision']))
        command_delta=float(np.max(np.abs(np.array(trajectories[1]['first_command'])-trajectories[0]['first_command'])))
        detail=dict(row=i,case=case,seed=seed,phase=phase,fold=fold,dof=group(x[i])[1],reasons=reasons[i],labels=y[i].tolist(),probabilities=p[:,i].tolist(),mean_advantage=float(adv[i]),minimum_advantage=float(minimum[i]),neighbors=neighbors,feature_swaps=swaps,
            snapshot_prefix_exact=True,intervention_step=step,first_command_delta_max=command_delta,entry_metadata=onbranch['entries'][phase]['metadata'],trajectories=trajectories)
        details.append(detail);save(ROOT/'paired'/f'{case}_{seed}_{phase}.json',detail)
        print('diagnosed',case,seed,phase,flush=True)
    unique={(r['case'],r['method']):r for r in video_selection}
    save(ROOT/'records.json',list(unique.values()));save(ROOT/'cases.json',tasks)
    save(ROOT/'diagnostics.json',details)
    save(ROOT/'summary.json',dict(aggregate=aggregate,phase_metrics=phase_metrics,fold_audits=model_audits,selected_pairs=len(details),diagnostic_videos=len(unique),fixed_rule={k:v for k,v in fixed.items() if k!='episodes'},new_rollouts=0,held_training_or_diagnostic_rows=0))
    save(ROOT/'paired_input_hashes.json',input_hashes)
    save(ROOT/'verification.json',dict(oof_predictions_exact=True,ancestral_groups_disjoint=True,normalization_training_only=True,selected_snapshots_and_prefixes_exact=len(details),no_new_rule=True,no_policy_changes=True))
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print(json.dumps(aggregate,indent=2),flush=True)


def report():
    s=read(ROOT/'summary.json');details=read(ROOT/'diagnostics.json');a=s['aggregate']
    lines=['# E128: Task-value ranking errors on excluded fitting families','',
        'Diagnosis only: no policy, estimator, confidence rule or held-placement score changed. E120 stays the baseline. These are out-of-fold predictions on fitting families, not the original held-placement benchmark.', '',
        '## Prediction errors','',f"{a['rows']} clean phase-entry pairs: {a['beneficial']} beneficial off-pulses, {a['harmful']} harmful, {a['neutral']} unchanged. Phase rows share trajectories and are not independent trials.", '',
        f"Mean-ensemble on-head accuracy {a['on_accuracy']:.3f}, off-head accuracy {a['off_accuracy']:.3f}; Brier scores {a['on_brier']:.3f} and{a['off_brier']:.3f}. Rankings reverse on {a['beneficial_rank_reversed']}/{a['beneficial']} beneficial and {a['harmful_rank_reversed']}/{a['harmful']} harmful pairs. Consensus advantage >0.1 accepts {a['consensus_beneficial']} beneficial and {a['consensus_harmful']} harmful phase pairs. These pair counts differ from first-choice complete-episode results.", '',
        'The fixed E126 first-choice rule (minimum advantage0.1, spread2) gains one episode and loses two among144 fitting validation episodes. Its complete-episode choices, rather than arbitrary later phase rows, are included below.', '',
        '| Fold | Training head accuracy (three models) | OOF on accuracy | OOF off accuracy |','|---:|---:|---:|---:|']
    for f in s['fold_audits']:lines.append(f"| {f['fold']} | {f['training_head_accuracy']:.4f} | {f['validation']['on_accuracy']:.4f} | {f['validation']['off_accuracy']:.4f} |")
    lines+=['','## Paired diagnostic cases','','Each off branch starts from an identical integration/history snapshot, disables the bounded residual for up to8 steps, and then resumes E120. Videos show both complete continuations. These are outcome-selected examples.','','| Case/seed | Phase | On/off success | Predicted on/off (mean) | First command difference (rad) | On/off first collision phase |','|---|---|---|---|---:|---|']
    for d in details:
        prob=np.array(d['probabilities']).mean(0);tt=d['trajectories'];coll=[t['first_collision']['phase'] if t['first_collision'] else 'none' for t in tt]
        lines.append(f"| {d['case']}/{d['seed']} | {d['phase']} | {int(d['labels'][0])}/{int(d['labels'][1])} | {prob[0]:.4f}/{prob[1]:.4f} | {d['first_command_delta_max']:.6f} | {coll[0]}/{coll[1]} |")
    lines+=['','## Training neighborhoods and input sensitivity','',
        'Neighbors come only from that model\'s training families, with matching phase, joint count and local-history flag. Distances use fold-normalized, clipped features. Full, history-only, and state/action neighborhoods are saved separately. The three returned neighbors are from distinct ancestral families; labels do not determine neighbor selection. Feature swaps below use the nearest full-feature neighbor and are neural sensitivity probes, not physical counterfactuals or causal explanations.','']
    for d in details:
        lines += [f"### Case{d['case']}, seed{d['seed']}, {d['phase']}",'',f"Reasons: {', '.join(d['reasons'])}. Observed labels {d['labels']}; predicted mean advantage {d['mean_advantage']:.4f}.",'', '| Neighborhood | Neighbor case/seed | Observed on/off | Distance |','|---|---|---|---:|']
        for kind,neighbors in d['neighbors'].items():
            for n in neighbors:lines.append(f"| {kind} | {n['case']}/{n['seed']} | {int(n['labels'][0])}/{int(n['labels'][1])} | {n['distance']:.4f} |")
        lines += ['','| Replaced input block | New predicted advantage | Change |','|---|---:|---:|']
        for r in d['feature_swaps']:lines.append(f"| {r['block']} | {r['mean_advantage']:.4f} | {r['change']:+.4f} |")
    lines+=['','## What the gate observes','',
        'The139 input features contain phase, joint mask, measured q/bias, Cartesian goal error, Gram/cross history statistics, two proposed joint actions, and local-history flag. They do not explicitly include object pose, grasp state, gripper contact, or tool orientation error. Saved simulation traces include some of this information for diagnosis. Its omission is an architectural limitation and a hypothesis for further testing; this experiment does not prove that adding it fixes ranking errors.', '',
        'Do not infer policy improvement from sensitivity probes or selected videos. No new gate was fitted and no confidence threshold was selected. Before another abstention heuristic, test whether the value representation can predict paired outcomes on whole excluded families, including the errors of both heads. E120 remains unchanged.', '']
    (ROOT/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4))
    labels=[f"{d['case']}/{d['seed']}\n{d['phase']}" for d in details];xx=np.arange(len(details))
    ax.bar(xx-.18,[d['mean_advantage'] for d in details],.35,label='Predicted off minus on')
    ax.bar(xx+.18,[d['labels'][1]-d['labels'][0] for d in details],.35,label='Observed success difference')
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(xx,labels);ax.set_ylim(-1.2,1.4);ax.set_ylabel('Off-minus-on advantage');ax.legend(ncol=2);fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    path=Path(__file__).with_name('TECHNICAL_REPORT.md');text=path.read_text(encoding='utf-8');marker='## E128: task-value ranking diagnosis';assert marker not in text
    text=text.replace('Revision 115 · 15 November 2026','Revision 116 · 15 November 2026',1)
    text+=f"\n{marker}\n\nDiagnosis only on{a['rows']} excluded-fitting-family phase pairs: reversed rankings on{a['beneficial_rank_reversed']}/{a['beneficial']} beneficial and{a['harmful_rank_reversed']}/{a['harmful']} harmful pairs. Saved{s['selected_pairs']} exact paired diagnostic cases with training-neighborhood and block-replacement sensitivity analysis. No new model/rule/rollout or held-placement retest. E120 unchanged. Full report: `outputs/ease_task_value_ranking_diagnosis_v1/report.md`.\n"
    path.write_text(text,encoding='utf-8')


def build_videos():
    source=Path(__file__).with_name('history_similarity_videos.py').read_text()
    start=source.index('    for case,method,mode in selection:')
    body=source[start:source.index("\nif __name__=='__main__':")]
    body=body.replace('E127','E128').replace('History similarity / E120 fallback','Task-value ranking diagnosis')
    header=source[:source.index('def run():')].replace('history_similarity_gate import ROOT','task_value_ranking_diagnosis import ROOT').replace('E127','E128')
    init="def run():\n    rows=json.loads((ROOT/'records.json').read_text());out=ROOT/'videos';out.mkdir(exist_ok=False)\n    selection=[(r['case'],r['method'],r['mode']) for r in rows]\n    manifest=dict(selection='Outcome-selected excluded-fitting-family pairs: E120 on versus short correction-off then E120. Diagnosis only; no new policy.',playback='Saved executed states, every second40ms control step at12.5fps (1x).',videos=[])\n"
    Path(__file__).with_name('task_value_ranking_videos.py').write_text(header+init+body+"\nif __name__=='__main__':run()\n")


def deepen():
    f=np.load(DATA/'gate_dataset.npz');y=f['y'];o=np.load(PRIOR/'oof_predictions.npz');folds=o['row_fold'];p=o['probabilities'].mean(0)
    prior_prob=np.zeros_like(y);rows=[]
    for fold in range(4):
        train=folds!=fold;val=~train;freq=y[train].mean(0);prior_prob[val]=freq
        rows.append(dict(fold=fold,training_success_frequency=freq.tolist(),validation_success_frequency=y[val].mean(0).tolist()))
    diagnostic=dict(fold_frequencies=rows,neural_head_accuracy=np.mean((p>.5)==y,axis=0).tolist(),
        training_frequency_head_accuracy=np.mean((prior_prob>.5)==y,axis=0).tolist(),neural_brier=np.mean((p-y)**2,axis=0).tolist(),training_frequency_brier=np.mean((prior_prob-y)**2,axis=0).tolist(),
        note='Constant outcome-frequency reference estimated from each training fold only; diagnostic predictor, not a new policy or confidence rule.')
    details=read(ROOT/'diagnostics.json')
    events=[]
    for d in details:
        for t in d['trajectories']:
            if t['first_collision']:
                events.append(dict(case=d['case'],seed=d['seed'],pulse_phase=d['phase'],method=t['method'],collision_phase=t['first_collision']['phase'],steps_after_pulse_entry=t['first_collision']['step']-d['intervention_step'],contacts=t['first_collision']['bad']))
    diagnostic['collision_delays']=events
    save(ROOT/'deeper_diagnostics.json',diagnostic)
    a=diagnostic['training_frequency_head_accuracy'];b=diagnostic['training_frequency_brier']
    lines=['\n## Class-frequency reference and delayed effects','',
        f"Training-fold outcome-frequency references achieve on/off accuracy {a[0]:.3f}/{a[1]:.3f} and Brier {b[0]:.3f}/{b[1]:.3f}. Neural OOF accuracy is0.603/0.597 with Brier0.347/0.352. These references use no validation labels for estimation and are not proposed control policies.",'',
        '| Fold | Train on/off success frequency | Excluded-family on/off frequency |','|---:|---|---|']
    for r in rows:lines.append(f"| {r['fold']} | {r['training_success_frequency']} | {r['validation_success_frequency']} |")
    lines += ['', '| Case/seed | Pulse phase | Failed branch | First collision phase | Steps after pulse entry | Contact pair |','|---|---|---|---|---:|---|']
    for r in events:lines.append(f"| {r['case']}/{r['seed']} | {r['pulse_phase']} | {r['method']} | {r['collision_phase']} | {r['steps_after_pulse_entry']} | {r['contacts']} |")
    lines += ['', '## Diagnosis and next controlled experiment','',
        '1. There is a large training-to-excluded-family prediction gap, including confidently wrong predictions from both value heads. All training heads fit their labels perfectly; excluded-family prediction quality and ranking collapse. More confidence filtering cannot repair a confidently wrong value representation by itself.',
        '2. Most phase-pair labels (989/1108) have identical on/off outcomes. Only119 directly distinguish the two choices. This is sparse ranking supervision, with dependent repeated phase labels; it is not1108 independent intervention experiments.',
        '3. The selected physical outcomes are delayed collision changes,44 to246 control steps after pulse entry. Nearest training neighborhoods include mixed outcomes. Single-block swaps show sensitivity to history and joint configuration, but do not establish a physical cause.',
        '4. A next controlled fitting-family experiment should compare direct paired-advantage supervision against the two-success-head objective, and separately test additional measured task-state inputs (object-relative pose and grasp/contact state where observable). Keep the same ancestral splits and E120 fallback. No such model or new rule was trained in E128.', '']
    path=ROOT/'report.md';path.write_text(path.read_text()+'\n'.join(lines))
    path=Path(__file__).with_name('TECHNICAL_REPORT.md');text=path.read_text(encoding='utf-8')
    text+=f"\nE128 additional diagnostics: training heads100% accurate; OOF on/off60.3%/59.7%. Fold-training frequency reference{a[0]:.1%}/{a[1]:.1%}. Most labels989/1108 are neutral; selected collisions differ44–246 control steps after pulse entry. Proposed next controlled tests: paired-advantage objective and measured task-state augmentation, not another confidence cutoff. No new policy trained.\n"
    path.write_text(text,encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print(json.dumps(diagnostic,indent=2))


def finish():
    protocol=read(ROOT/'protocol.json');extra=read(ROOT/'paired_input_hashes.json')
    for path,h in {**protocol['inputs'],**extra}.items():assert fingerprint(path)==h
    manifest=read(ROOT/'videos'/'manifest.json')
    for r in manifest['videos']:assert fingerprint(ROOT/'videos'/r['file'])==r['sha256']
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    save(ROOT/'status.json',dict(status='complete',diagnosis_only=True,new_rollouts=0,new_gate_or_rule=False,held_placement_retest=False,baseline_unchanged=True,paired_cases=read(ROOT/'summary.json')['selected_pairs'],videos=len(manifest['videos'])))
    path=ROOT/'report.md';path.write_text(path.read_text()+f"\nVerification: all OOF predictions reproduced, fold ancestry disjoint, fitting-only normalization checked, all selected intervention snapshots/prefixes exact. Source weights, data and E120 records unchanged. Saved{len(manifest['videos'])} trace-verified success/failure videos.\n")
    hashes={str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
    save(ROOT/'artifact_hashes.json',hashes);print(json.dumps(read(ROOT/'status.json'),indent=2))


def verify():
    from .broad_task_value_gate import features
    x=np.load(DATA/'gate_dataset.npz')['x'];y=np.load(DATA/'gate_dataset.npz')['y'];meta=read(DATA/'gate_rows.json')
    lookup={(r['case'],r['seed'],r['off_phase']):r for r in read(COLLECT/'records.json')}
    for i,r in enumerate(meta):np.testing.assert_array_equal(y[i],[lookup[r['case'],r['seed'],p]['success'] for p in (None,r['phase'])])
    checked=0
    for d in read(ROOT/'diagnostics.json'):
        case,seed,phase=d['case'],d['seed'],d['phase'];step=d['intervention_step']
        on=read(COLLECT/'actions'/f'{case}_on_{seed}_frozen_bias.json');a=on[step]
        snap=np.load(COLLECT/'snapshots'/f'{case}_on_{seed}_{phase}.npz')
        feat=features(len(a['q_before']),snap['gram'],snap['cross'],snap['bias'],a['q_before'],a['position_before'],snap['goal'],phase,d['entry_metadata']['local'],a['base_action'],a['correction_action'])
        np.testing.assert_array_equal(feat,x[d['row']])
        for method,offphase in ((f'on_{seed}',None),(f'off_{phase}_{seed}',phase)):
            counts={}
            for a in read(COLLECT/'actions'/f'{case}_{method}_frozen_bias.json'):
                p=a['phase'];k=counts.get(p,0);counts[p]=k+1
                if not a['active']:continue
                base,corr=np.array(a['base_action']),np.array(a['correction_action'])
                expected=base if p==offphase and k<8 else np.clip(base+np.clip(.25*(corr-base),-.001,.001),-.04,.04)
                np.testing.assert_array_equal(expected,a['nominal_action']);checked+=1
    v=read(ROOT/'verification.json');v.update(all1108_labels_verified=True,selected_raw_features_exact=True,selected_action_steps_verified=checked);save(ROOT/'verification.json',v)
    path=ROOT/'report.md';path.write_text(path.read_text()+f'\nAdditional audit: all1108 labels checked against saved collection outcomes; all five selected raw feature vectors exactly reconstructed from intervention snapshots; {checked} on/off action steps match the defined bounded-E120/pulse rule.\n')
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print('Verified',checked,'action steps, all labels and selected raw features')


if __name__=='__main__':
    import sys
    {'analyze':analyze,'report':report,'build':build_videos,'deepen':deepen,'verify':verify,'finish':finish}[sys.argv[1]]()
