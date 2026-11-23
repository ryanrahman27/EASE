"""E178 retrospective prediction/action attribution with explicit causal limits."""
from pathlib import Path
import shutil
import numpy as np
from .matched_unseen_prepare import ROOT,SOURCE,CASES
from .matched_unseen_analyze import METHODS,MODELS
from .abstaining_task_value_gate import read,save,fingerprint

MOTION={'approach','descend','lift','transport','lower','retreat'}

def stats(rows):
    if not rows:return dict(n=0)
    prediction={k:dict(rmse_mm=float(np.sqrt(np.mean([r['prediction_error_mm'][k]**2 for r in rows]))),median_mm=float(np.median([r['prediction_error_mm'][k] for r in rows])),command_rmse_mm=float(np.sqrt(np.mean([r['command_prediction_error_mm'][k]**2 for r in rows]))),jacobian_position_error_mean=float(np.mean([r['jacobian_position_error'][k] for r in rows]))) for k in MODELS}
    progress=np.array([[b['position_progress_mm'] for b in r['branches']] for r in rows]);collisions=np.array([[b['collision'] for b in r['branches']] for r in rows]);delta=progress[:,0]-progress[:,1]
    return dict(n=len(rows),prediction=prediction,action=dict(E120_minus_analytical_progress_mm=float(delta.mean()),E120_progress_better_gt_0_1mm=int(np.sum(delta>.1)),analytical_progress_better_gt_0_1mm=int(np.sum(delta<-.1)),nominal_effect_at_analytical_metric_mm=float(np.mean(progress[:,4]-progress[:,1])),metric_effect_at_E120_nominal_mm=float(np.mean(progress[:,0]-progress[:,4])),bounded_correction_effect_mm=float(np.mean(progress[:,0]-progress[:,2])),alternative_nominal_effect_at_E120_metric_mm=float(np.mean(progress[:,0]-progress[:,5])),alternative_metric_effect_at_analytical_nominal_mm=float(np.mean(progress[:,5]-progress[:,1])),mean_final_action_gap_mrad=float(np.mean([r['final_E120_analytical_gap_mrad'] for r in rows])),mean_bounded_correction_mrad=float(np.mean([r['bounded_correction_mrad'] for r in rows])),counterfactual_collision_steps={m:int(collisions[:,i].sum()) for i,m in enumerate(METHODS)},mean_progress_mm={m:float(progress[:,i].mean()) for i,m in enumerate(METHODS)},mean_local_oracle_action_distance_mrad={m:float(np.mean([r['branches'][i]['oracle_action_distance_mrad'] for r in rows])) for i,m in enumerate(METHODS)}),median_gram_condition=float(np.median([r['gram_condition'] for r in rows])))

def run():
    all_rows=[];case_rows={};verification={};tasks=read(ROOT/'cases.json');chronology=[]
    for source in ('e120_ensemble','analytical'):
        verification[source]=read(ROOT/'analysis'/source/'verification.json');assert verification[source]['histories']==verification[source]['source_predictions_exact']==verification[source]['source_physical_steps_verified']
        for case in CASES:
            rows=read(ROOT/'analysis'/source/f'{case}.json');a=np.load(ROOT/'analysis'/source/f'{case}.npz');logs=read(ROOT/f'replay_{source}/clearance'/f'{case}_{source}_frozen_bias.json');trial=read(ROOT/f'replay_{source}/trials'/f'{case}_{source}_frozen_bias.json');source_index=0 if source=='e120_ensemble' else 1
            assert len(rows)==len(logs)==len(a['steps']);np.testing.assert_array_equal(a['steps'],np.arange(len(rows)))
            original_states=np.load(SOURCE/source/'states'/f'{case}_{source}_frozen_bias.npz')['states'];np.testing.assert_allclose(a['branch_states'][:,source_index],original_states,atol=1e-12,rtol=0)
            for i,r in enumerate(rows):
                errors=1000*np.linalg.norm(a['position_predictions'][i,:,:3]-a['observed_dx'][i,:3],axis=1);np.testing.assert_allclose(errors,[r['prediction_error_mm'][k] for k in MODELS],atol=1e-12,rtol=0)
                assert r['branches'][source_index]['bad']==trial['trace'][i]['bad']
                for j,b in enumerate(r['branches']):np.testing.assert_allclose(b['position_progress_mm'],r['position_error_mm']-1000*a['branch_metrics'][i,j,0],atol=1e-12,rtol=0)
                assert abs((r['branches'][4]['position_progress_mm']-r['branches'][1]['position_progress_mm'])+(r['branches'][0]['position_progress_mm']-r['branches'][4]['position_progress_mm'])-(r['branches'][0]['position_progress_mm']-r['branches'][1]['position_progress_mm']))<1e-12
            all_rows+=rows;case_rows[source,case]=rows
            endpoints=[];cursor=0
            for p in trial['phases']:
                first=cursor;cursor+=p['steps'];last=trial['trace'][cursor-1];ll=logs[cursor-1];endpoints.append(dict(phase=p['phase'],first_step=first,last_step=cursor-1,reached=p['reached'],position_error_mm=1000*last['position_error'],rotation_error=last['rotation_error'],object=last['object'],pads=last['pads'],end_pads=ll['end_pads'],close_good=ll['close_good'],close_ready=ll['close_ready'],aborted=ll['aborted']))
            first_bad=next((dict(step=i,phase=x['phase'],pairs=x['bad']) for i,x in enumerate(trial['trace']) if x['bad']),None)
            chronology.append(dict(case=case,source=source,success=trial['success'],grasped=trial['grasped'],lifted=trial['lifted'],collision=trial['collision'],first_contact=first_bad,phases=endpoints,final_object=trial['final_object'],target=tasks[case]['target']))
    summary={};phase_rows=[]
    for group in ('E120_loss','E120_win_control'):
        summary[group]={}
        for source in ('e120_ensemble','analytical'):
            rr=[r for r in all_rows if r['group']==group and r['source']==source];summary[group][source]=dict(all_states=stats(rr),clean_motion=stats([r for r in rr if r['clean_before'] and r['phase'] in MOTION]),clean_motion_no_pad_contact=stats([r for r in rr if r['clean_before'] and r['phase'] in MOTION and r['pre_pads']==0]))
    per_case=[]
    for case in CASES:
        for source in ('e120_ensemble','analytical'):
            rr=case_rows[source,case];per_case.append(dict(case=case,source=source,clean_motion=stats([r for r in rr if r['clean_before'] and r['phase'] in MOTION])))
            for phase in dict.fromkeys(r['phase'] for r in rr):phase_rows.append(dict(case=case,source=source,phase=phase,stats=stats([r for r in rr if r['phase']==phase and r['clean_before']])))
    divergences=[]
    for case in CASES:
        a=read(ROOT/'replay_e120_ensemble/clearance'/f'{case}_e120_ensemble_frozen_bias.json');b=read(ROOT/'replay_analytical/clearance'/f'{case}_analytical_frozen_bias.json');first=next((i for i in range(min(len(a),len(b))) if np.max(np.abs(np.array(a[i]['final_action'])-b[i]['final_action']))>1e-10),None);prefix=0
        for x,y in zip(a,b):
            if x['phase']!=y['phase'] or not np.array_equal(x['full_qpos_before'],y['full_qpos_before']) or not np.array_equal(x['full_qvel_before'],y['full_qvel_before']):break
            prefix+=1
        selected={}
        for source,logs in (('e120_ensemble',a),('analytical',b)):
            rows=case_rows[source,case];points={0}
            if first is not None and first<len(rows):points.add(first)
            for phase in ('descend','probe','close','lift','transport','lower'):
                ids=[i for i,r in enumerate(rows) if r['phase']==phase]
                if ids:points.update((ids[0],ids[-1]))
            contact=next(x['first_contact'] for x in chronology if x['case']==case and x['source']==source)
            if contact:points.update((max(0,contact['step']-1),contact['step']))
            selected[source]=[rows[i] for i in sorted(points)]
        divergences.append(dict(case=case,first_actual_action_difference_step=first,identical_pre_state_prefix_length=prefix,selected_matched_history_steps=selected))
    save(ROOT/'summary.json',summary);save(ROOT/'per_case.json',per_case);save(ROOT/'per_phase.json',phase_rows);save(ROOT/'chronology.json',chronology);save(ROOT/'divergences.json',divergences)
    save(ROOT/'verification.json',dict(conditions=verification,exact_full_episode_replays=12,matched_histories=len(all_rows),diagnostic_candidates_per_history=len(METHODS),recomputed_errors_and_progress=True,physical_source_branch_match=True,controller_unchanged=True,retrospective=True))
    lines=['# E178: why E120 loses cases0,8,37','','Retrospective matched-history diagnosis of the E177 cohort. Cases9,42,45, where E120 won, are included as controls. All12 original trajectories were exactly replayed with passive full integration-state logging. No weights, controller settings, task budgets or scoring were changed.','',f'{len(all_rows)} pre-action histories from both source policies were analyzed. At each state, both estimators receive identical full-precision history statistics, calibrated bias, current state, goal and controller context. Six action candidates are evaluated for one40ms step in independent restored simulators. The actual source candidate reproduces every saved post-state to1e-12 and every contact union.','', '## Matched prediction and action results','','Primary summary: states clean before the action in approach, descent, lift, transport, lower or retreat. Prediction error is vector RMSE of predicted position change J*dq against the same observed next position change, using measured joint change dq. Command-to-motion error is separate in summary.json. Post-contact states, probe and grasp phases remain in the raw/per-phase results.','','| Group / history source | States | Reference J RMSE (mm) | Correction J RMSE (mm) | Analytical RMSE (mm) | Local true J RMSE (mm) | E120 minus analytical one-step progress (mm) |','|---|---:|---:|---:|---:|---:|---:|']
    for group,ss in summary.items():
        for source,v in ss.items():
            x=v['clean_motion'];p=x['prediction'];lines.append(f"| {group} / {source} | {x['n']} | {p['reference']['rmse_mm']:.4f} | {p['correction']['rmse_mm']:.4f} | {p['analytical']['rmse_mm']:.4f} | {p['oracle']['rmse_mm']:.4f} | {x['action']['E120_minus_analytical_progress_mm']:.4f} |")
    lines+=['','E120 has no single blended Jacobian: its reference Jacobian supplies the main DLS action and projection metric, while the correction Jacobian contributes a bounded action residual. The two neural prediction columns above are component diagnostics, not two separately deployed policies. The local true Jacobian is an offline privileged derivative; finite-step curvature can give it nonzero error.','','## Action attribution','','The full E120/analytical swap changes both the nominal action and the metric used by the unchanged safety projection. Crossed candidates separate these on each fixed state: analytical metric held fixed while changing the nominal target, then E120 nominal held fixed while changing the metric. These two progress differences sum exactly to the total one-step difference; the reverse order is also saved to expose interactions. The no-correction candidate removes only the bounded E120 residual.','','| Case / history | Reference prediction RMSE | Analytical prediction RMSE | Total E120 progress difference | Nominal-target effect | Metric effect | Bounded-correction effect |','|---|---:|---:|---:|---:|---:|---:|']
    for x in per_case:
        v=x['clean_motion'];p=v['prediction'];a=v['action'];lines.append(f"| {x['case']} / {x['source']} | {p['reference']['rmse_mm']:.4f} | {p['analytical']['rmse_mm']:.4f} | {a['E120_minus_analytical_progress_mm']:.4f} | {a['nominal_effect_at_analytical_metric_mm']:.4f} | {a['metric_effect_at_E120_nominal_mm']:.4f} | {a['bounded_correction_effect_mm']:.4f} |")
    lines+=['','All prediction/progress quantities above are mm; progress differences are mean per step, not terminal task effects. Distance to a privileged local-Jacobian action (saved separately) is a diagnostic proxy, not an optimal-action ground truth. The original route, handoff/abort decisions, phase, aperture, clearance gradient and error allowance are held fixed for each source context. A different full policy could have arrived with a different context.','','## Failure chronology','','| Case / source | Success | Grasped | Lifted | Collision | First forbidden contact | First unreached motion phase |','|---|---|---|---|---|---|---|']
    for x in chronology:
        phase=next((p['phase'] for p in x['phases'] if p['phase'] in MOTION and not p['reached']),None);lines.append(f"| {x['case']} / {x['source']} | {x['success']} | {x['grasped']} | {x['lifted']} | {x['collision']} | {x['first_contact']} | {phase} |")
    lines+=['','Full phase endpoints, object positions, pad contacts, guarded-handoff states and first contacts are in chronology.json. divergences.json retains the first actual action divergence and prespecified phase-boundary/contact checkpoints with matched model and action diagnostics. per_phase.json separates early prediction/action errors from later failed-trajectory aftermath.','','## Interpretation limits','','This is retrospective analysis of six outcome-selected cases, not a new held evaluation or a population-level estimator comparison. One-step counterfactuals can identify immediate prediction/action differences, but cannot prove a complete-task rescue or uniquely attribute accumulated failures to one component. Better immediate goal progress may still harm grasp/contact behavior. Collision and object-displacement outcomes are recorded alongside progress. No candidate is deployed, no source outcome is relabeled and no tuning follows this analysis. The full E177 cohort remains42/48 for each estimator, with1 versus0 collisions.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(13,9))
    for row,case in enumerate((0,8,37)):
        rr=case_rows['e120_ensemble',case];time=np.arange(len(rr))*.04
        def smooth(v):return np.convolve(np.asarray(v),np.ones(12)/12,mode='same')
        for model,color in (('reference','#a14c42'),('analytical','#347bb5'),('oracle','#777777')):axes[row,0].plot(time,smooth([r['prediction_error_mm'][model] for r in rr]),label=model,color=color)
        axes[row,0].set_title(f'Case{case}: matched motion prediction');axes[row,0].set_ylabel('Position error (mm),12-step mean');axes[row,0].legend()
        axes[row,1].plot(time,smooth([r['branches'][0]['position_progress_mm']-r['branches'][1]['position_progress_mm'] for r in rr]),color='#29896c');axes[row,1].axhline(0,color='black',lw=.7);axes[row,1].set_title(f'Case{case}: same-state action progress');axes[row,1].set_ylabel('E120 minus analytical (mm/step)')
        contact=next(x['first_contact'] for x in chronology if x['case']==case and x['source']=='e120_ensemble')
        for ax in axes[row]:
            ax.set_xlabel('Task time after calibration (s)')
            if contact:ax.axvline(contact['step']*.04,color='red',ls='--',label='first actual collision')
    fig.suptitle('E178 | E120-source histories | offline matched diagnostics; controller frozen');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    video_rows=[]
    for folder in ('trials','states'):(ROOT/folder).mkdir(exist_ok=True)
    for case in CASES:
        for source in ('e120_ensemble','analytical'):
            method='privileged_'+source;old=f'{case}_{source}_frozen_bias';new=f'{case}_{method}_frozen_bias';r=read(ROOT/f'replay_{source}/trials'/f'{old}.json');r['method']=method;save(ROOT/'trials'/f'{new}.json',r)
            for ext in ('npz','xml'):shutil.copy2(ROOT/f'replay_{source}/states'/f'{old}.{ext}',ROOT/'states'/f'{new}.{ext}')
            video_rows.append(dict(case=case,method=method,mode='frozen_bias',success=r['success'],dof=6))
    save(ROOT/'video_records.json',video_rows)
    template=Path(__file__).with_name('link_clearance_videos.py').read_text().replace('link_clearance_diagnostic import ROOT','matched_unseen_prepare import ROOT').replace('E168','E178').replace('link-clearance comparison','matched-history diagnosis: actual frozen trajectories').replace('PRIVILEGED link/table constraints around E120 versus unchanged E120+release. Fitting cases only; successes, failures and stalls retained.','Actual frozen E177 trajectories: three E120 losses and three wins as controls. Matched one-step branches are offline diagnostics, not full-task successes.').replace('PRIVILEGED geometry constraint diagnostic','ACTUAL frozen-policy trajectory | privileged protections')
    Path(__file__).with_name('matched_unseen_videos.py').write_text(template);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'status.json',dict(status='audited_videos_pending',videos=12));print(summary,flush=True)

def finish():
    videos=read(ROOT/'videos/manifest.json')['videos'];assert len(videos)==12
    for v in videos:assert v['all_saved_states_match_trace'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    verify=read(ROOT/'verification.json');section=f"\n## E178: matched-history diagnosis of E120 losses\n\nRetrospective diagnosis of E177 losses0,8,37 and wins9,42,45 as controls.12 exact full-episode replays; {verify['matched_histories']} matched histories with6 offline one-step diagnostic action candidates each. Source predictions exact and source physical branches verified against recorded states/contacts. No policy tuning or new task-success claims. Report: `outputs/ease_matched_unseen_diagnosis_v1/report.md`; detailed interpretation: `interpretation.md`; videos: `videos/index.html`.\n";review=Path('outputs/ease_iteration_review_v1')
    interpretation=ROOT/'interpretation.md'
    if interpretation.exists():section+='\n'+interpretation.read_text(encoding='utf-8')+'\n'
    for p in (Path(__file__).with_name('TECHNICAL_REPORT.md'),review/'report.md'):
        s=p.read_text(encoding='utf-8')
        if '## E178:' not in s:p.write_text(s+section,encoding='utf-8')
    p=review/'WORKING_STATE.md';s=p.read_text(encoding='utf-8')
    if not s.startswith('# Latest checkpoint: E178'):p.write_text('# Latest checkpoint: E178 complete\n'+section+'\nAll replays, diagnostics, audits and videos complete. Controller unchanged.\n\n---\n'+s,encoding='utf-8')
    studies=read(review/'studies.json')
    if not any(x['experiment']=='E178' for x in studies):studies.append(dict(experiment='E178',folder=str(ROOT),report_exists=True,videos=12));save(review/'studies.json',studies)
    for p,h in {**read(ROOT/'frozen.json'),**read(ROOT/'analysis_frozen.json'),**read(ROOT/'protocol.json')['checkpoints']}.items():assert fingerprint(p)==h
    save(ROOT/'status.json',dict(status='complete',exact_replays=12,matched_histories=verify['matched_histories'],diagnostic_candidates=6,videos=12,controller_changed=False,retrospective=True));save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print('E178 complete',verify['matched_histories'],'histories;12 videos',flush=True)

if __name__=='__main__':
    import sys
    finish() if len(sys.argv)>1 and sys.argv[1]=='finish' else run()
