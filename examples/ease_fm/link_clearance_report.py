"""Audit privileged link constraints, progress, actual contacts and replay artifacts."""
from pathlib import Path
from collections import Counter
import shutil
import numpy as np
import mujoco
from .link_clearance_diagnostic import ROOT,BASE,CONDITIONS,geometry_ids,distances,distance_gradient,MARGIN,ALPHA
from .abstaining_task_value_gate import read,save,fingerprint

def outcome_counts(rows,root):
    contacts=Counter();fail_phases=Counter();link_episodes=0;gripper_episodes=0;link_probe=0
    for r in rows:
        stem=f"{r['case']}_e120_ensemble_frozen_bias";t=read(root/'trials'/f'{stem}.json');m=mujoco.MjModel.from_xml_path(str(root/'states'/f'{stem}.xml'));links,_=geometry_ids(m,6);labels={mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or str(g) for g in links};first=next((x for x in t['trace'] if x['bad']),None)
        if first:contacts[first['phase']]+=1
        link=lambda x:any('table' in pair and bool(set(pair)&labels) for pair in x['bad'])
        link_episodes+=any(link(x) for x in t['trace']);link_probe+=any(x['phase']=='probe' and link(x) for x in t['trace']);gripper_episodes+=any('table' in pair and any('gripper' in name for name in pair) for x in t['trace'] for pair in x['bad'])
        if not r['success'] and not r['collision']:
            for phase in t['phases']:
                if not phase['reached']:fail_phases[phase['phase']]+=1
    return dict(n=len(rows),success=sum(r['success'] for r in rows),collisions=sum(r['collision'] for r in rows),grasped=sum(r['grasped'] for r in rows),lifted=sum(r['lifted'] for r in rows),collision_free_failures=sum(not r['success'] and not r['collision'] for r in rows),link_table_collision_episodes=int(link_episodes),link_table_contacts_during_probe=int(link_probe),gripper_table_collision_episodes=int(gripper_episodes),first_contact_phases=dict(contacts),collision_free_failure_unreached_phases=dict(fail_phases))

def run():
    baseline=[r for r in read(BASE/'records.json') if r['dof']==6];base=outcome_counts(baseline,BASE);lookup={r['case']:r for r in baseline};summary={};checks={};selected={};video_rows=[]
    for condition in CONDITIONS:
        root=ROOT/condition;assert read(root/'verification.json')['episodes']==48;rows=read(root/'records.json');assert len(rows)==48;counts=outcome_counts(rows,root);changes=[];mins=[];qp=0;fallback=0;eligible=0;violations=0;corrected=0;steps=0;probe_violations=0;gradient_checks=0;post_checks=0;hold_comparisons=0;constraint_steps_by_phase=Counter()
        for r in rows:
            stem=f"{r['case']}_e120_ensemble_frozen_bias";logs=read(root/'clearance'/f'{stem}.json');actions=read(root/'actions'/f'{stem}.json');t=read(root/'trials'/f'{stem}.json');b=read(BASE/'trials'/f'{stem}.json');states=np.load(root/'states'/f'{stem}.npz')['states'];assert len(logs)==len(states)==len(t['trace'])==len(actions);assert t['calibration']==b['calibration'] and t['initial_estimator']==b['initial_estimator'];assert t['initial_estimator']['bias']==t['final_estimator']['bias'];m=mujoco.MjModel.from_xml_path(str(root/'states'/f'{stem}.xml'));d=mujoco.MjData(m);scratch=mujoco.MjData(m);links,table=geometry_ids(m,6);gradient_checked=False
            for l,a,state in zip(logs,actions,states):
                steps+=1;np.testing.assert_array_equal(l['q_before'],a['q_before']);np.testing.assert_array_equal(l['executed_command'],a['executed_command']);np.testing.assert_array_equal(np.clip(np.array(l['q_before'])+np.clip(l['final_action'],-.04,.04),-2.5,2.5)-l['q_before'],l['executed_command']);assert l['phase']==a['phase']
                if l['phase'] in ('lower','release'):assert l['goal'][2]==.265
                if l['phase']=='probe':assert not l['eligible'];np.testing.assert_array_equal(l['nominal_action'],l['final_action'])
                d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_forward(m,d);np.testing.assert_allclose(distances(m,d,links,table),l['distance_after'],atol=1e-12,rtol=0);post_checks+=1
                assert np.all(np.array(l['substep_min'])<=np.minimum(l['distance_before'],l['distance_after'])+1e-12)
                badmargin=np.min(l['substep_min'])<MARGIN-1e-6;violations+=badmargin;probe_violations+=badmargin and l['phase']=='probe';eligible+=l['eligible'];qp+=l['solve']['attempted'];fallback+=l['solve']['fallback'];corrected+=np.max(np.abs(np.array(l['final_action'])-l['nominal_action']))>1e-10
                if l['solve']['attempted']:constraint_steps_by_phase[l['phase']]+=1
                if l['eligible']:
                    g=np.array(l['gradient']);rhs=np.array(l['rhs']);u=np.array(l['final_action']);bias=np.array(l['bias']);nominal=np.array(l['nominal_action']);q=np.array(l['q_before']);lo=np.maximum(-.04,-2.5-q);hi=np.minimum(.04,2.5-q);np.testing.assert_allclose(rhs,g@bias-ALPHA*(np.array(l['distance_before'])-MARGIN),atol=1e-14,rtol=0);np.testing.assert_allclose(g@u-rhs,l['solve']['linear_slack'],atol=1e-14,rtol=0)
                    if l['solve']['solver_success']:assert np.min(g@u-rhs)>=-1e-7
                    if l['solve']['attempted']:
                        assert np.all(u>=lo-1e-10) and np.all(u<=hi+1e-10)
                        if l['solve']['fallback']:np.testing.assert_array_equal(u,np.clip(bias,lo,hi))
                        elif np.min(g@bias-rhs)>=0 and np.all(bias>=lo) and np.all(bias<=hi):
                            h=np.array(l['metric']);h/=np.trace(h)/6;cost=lambda x:.5*(x-nominal)@h@(x-nominal);assert cost(u)<=cost(bias)+1e-7;hold_comparisons+=1
                    if not gradient_checked:
                        d.qpos[:6]=q;mujoco.mj_forward(m,d);np.testing.assert_allclose(distances(m,d,links,table),l['distance_before'],atol=1e-12,rtol=0);np.testing.assert_allclose(distance_gradient(m,d,scratch,links,table,6),g,atol=1e-10,rtol=0);gradient_checked=True;gradient_checks+=1
                else:np.testing.assert_array_equal(l['nominal_action'],l['final_action'])
            mins.append(dict(case=r['case'],minimum_m=float(np.min([l['substep_min'] for l in logs])),success=r['success'],collision=r['collision']))
            if r['success']!=lookup[r['case']]['success']:changes.append(dict(case=r['case'],body=r['body'],before=lookup[r['case']]['success'],after=r['success']))
        gains=sum(x['after'] for x in changes);losses=sum(x['before'] for x in changes);early=sum(counts['first_contact_phases'].get(p,0) for p in ('approach','descend'));passed=counts['success']>base['success'] and counts['collisions']<=base['collisions'] and early<sum(base['first_contact_phases'].get(p,0) for p in ('approach','descend')) and losses==0
        summary[condition]=dict(outcomes=counts,gains=gains,losses=losses,changes=changes,positive_fitting_screen=passed,controller_steps=steps,eligible_steps=eligible,qp_steps=qp,changed_action_steps=int(corrected),fallback_steps=fallback,qp_steps_by_phase=dict(constraint_steps_by_phase),actual_substep_below_3mm_steps=int(violations),below_3mm_probe_steps=int(probe_violations),per_episode_minimum_clearance=mins)
        checks[condition]=dict(all_actual_commands_verified=True,release_fix_and_probe_preserved=True,accepted_linear_constraints_verified=True,post_state_geometry_checks=post_checks,independent_gradient_checks=gradient_checks,feasible_hold_cost_comparisons=hold_comparisons,original_calibration_and_frozen_bias=True)
        chosen={13,26,27}
        for success in (True,False):
            options=[r for r in rows if r['success']==success]
            if options:chosen.add(options[0]['case'])
        for gain in (True,False):
            options=[r for r in changes if r['after']==gain]
            if options:chosen.add(options[0]['case'])
        # Explicitly include a collision-free failure if constraints prevent progress.
        options=[r for r in rows if not r['success'] and not r['collision']]
        if options:chosen.add(options[0]['case'])
        selected[condition]=sorted(chosen)
    save(ROOT/'summary.json',dict(baseline=base,conditions=summary,privileged_geometry_diagnostic=True,E120_unchanged=True,release_fix_preserved=True,held_evaluated=False));save(ROOT/'verification.json',dict(conditions=checks,exact_disabled_baseline_replays=4,new_intervention_episodes=96,no_held_execution=True))
    for p,h in read(ROOT/'frozen.json').items():assert fingerprint(p)==h
    for folder in ('trials','states'):(ROOT/folder).mkdir()
    for condition,cases in selected.items():
        for case in cases:
            for source,method in ((ROOT/condition,'privileged_'+condition),(BASE,'baseline_E120_release')):
                if any(v['case']==case and v['method']==method for v in video_rows):continue
                old=f'{case}_e120_ensemble_frozen_bias';new=f'{case}_{method}_frozen_bias';t=read(source/'trials'/f'{old}.json');t['method']=method;save(ROOT/'trials'/f'{new}.json',t)
                for ext in ('xml','npz'):shutil.copy2(source/'states'/f'{old}.{ext}',ROOT/'states'/f'{new}.{ext}')
                video_rows.append(dict(case=case,method=method,mode='frozen_bias',success=t['success'],dof=6))
    save(ROOT/'video_records.json',video_rows)
    lines=['# E168: privileged link-clearance constraints around E120','','**PRIVILEGED GEOMETRY DIAGNOSTIC ONLY.** Exact simulator link/table distances and finite-difference gradients constrain the frozen E120 ensemble command. The estimator remains learned, but the geometry layer is privileged; these results are not history-only adaptation, unseen-body transfer, or a safety guarantee.','', '| Controller | Success /48 | All collision episodes | Link/table collision episodes | Collision-free failures | Rescues / regressions |','|---|---:|---:|---:|---:|---:|',f"| E120 + release reference | {base['success']} | {base['collisions']} | {base['link_table_collision_episodes']} | {base['collision_free_failures']} | — |"]
    for c,v in summary.items():
        x=v['outcomes'];lines.append(f"| {c} | {x['success']} | {x['collisions']} | {x['link_table_collision_episodes']} | {x['collision_free_failures']} | {v['gains']} / {v['losses']} |")
    lines+=['','All48 six-joint fitting placements were executed for each variant (96 new episodes), with four exact disabled-layer baseline replays beforehand. All cases are retained. Calibration, frozen measured bias, original E120 models and bounded residual,265mm release target, task waypoints, phase budgets and strict success/contact scoring remain unchanged. No higher approach or previous descent/centering intervention is included.','', 'The layer enforces linearized constraints G*u ≥ G*bias −0.5*(distance−0.003m), original command bounds and joint-target bounds. Gradients use scratch forward kinematics, never advancing or mutating live state. The objective is either nearest joint command or nearest predicted tool motion using the E120 estimated Jacobian plus0.015² joint regularization. No exact tool Jacobian, future dynamics rollout or object-pose control input is used.','', 'The geometry constraint covers all six arm capsules against the table. Gripper, ground and self-contact remain scored but are not constrained. Calibration and the fixed local probe are intentionally untouched. Constraints therefore may move failure into a probe or another contact type. The assumed displacement is command minus calibrated bias; nonlinear motion and dynamics can violate actual clearance even when the linear QP succeeds. Solver failures use a logged clipped-bias hold, not silent constraint relaxation.','', '| Variant | Grasped / lifted | QP steps / fallbacks | Link/table contact during probe (episodes) | Gripper/table contact (episodes) | Positive fitting screen |','|---|---:|---:|---:|---:|---|']
    # Correct the header delimiter to match its six fields.
    lines[-1]='|---|---:|---:|---:|---:|---|'
    for c,v in summary.items():
        x=v['outcomes'];lines.append(f"| {c} | {x['grasped']} / {x['lifted']} | {v['qp_steps']} / {v['fallback_steps']} | {x['link_table_contacts_during_probe']} | {x['gripper_table_collision_episodes']} | {v['positive_fitting_screen']} |")
    lines+=['', 'Success, collision-free failures and failed phase counts are reported separately so that holding safely without completing the task is not counted as improvement. All paired losses and first-contact phases are retained in summary.json, together with actual sampled substep minimum clearances and constraint activity. Contact-type episode counts can overlap.','', 'Verification covers original calibration/frozen bias, unchanged probe commands and release goals, every final executed command, all accepted linear inequalities, all saved post-state geometry distances, one independently reconstructed distance gradient per eligible episode, and comparison with a feasible hold command’s objective cost. Before intervention outcomes, a nominal-action fast-path joint-bound omission was corrected and archived under pre_intervention_review; numerical checks include known QP solutions, target limits, infeasible fallback and an independent directional geometry derivative. No outcome-driven parameter change occurred.','', 'Success/failure videos are labeled by controller privilege and replay actual saved states. Baseline clips are separately labeled E120+release. Contacts are scored at2ms simulation steps; videos sample80ms and can miss brief contacts. Every rendered state matches saved tool/object traces and decoded frame counts. This diagnostic is fitting-only and does not replace E120 or open a held test.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4));labels=['E120 + release','Joint projection','Task projection'];values=[base]+[summary[c]['outcomes'] for c in CONDITIONS]
    for ax,metric,title in zip(axes,('success','collisions','collision_free_failures'),('Completed tasks /48','Any collision episodes /48','Collision-free failures /48')):
        numbers=[v[metric] for v in values];ax.bar(labels,numbers,color=['#777777','#2379b5','#d18b35']);ax.set_ylim(0,48);ax.set_title(title);ax.tick_params(axis='x',rotation=15)
        for i,x in enumerate(numbers):ax.text(i,x+.5,str(x),ha='center')
    fig.suptitle('E168 | PRIVILEGED link/table constraints | fitting six-joint bodies');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    template=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','link_clearance_diagnostic import ROOT').replace("ROOT/'records.json'","ROOT/'video_records.json'").replace('E128','E168').replace('Task-value ranking diagnosis','PRIVILEGED link-clearance comparison').replace('Outcome-selected excluded-fitting-family pairs: E120 on versus short correction-off then E120. Diagnosis only; no new policy.','PRIVILEGED link/table constraints around E120 versus unchanged E120+release. Fitting cases only; successes, failures and stalls retained.').replace("draw.text((10,42),'Measured-history control | No physics selector or geometry filter',fill='white')","draw.text((10,42),'PRIVILEGED geometry constraint diagnostic' if method.startswith('privileged_') else 'BASELINE E120 + release | No geometry constraint',fill='white')")
    Path(__file__).with_name('link_clearance_videos.py').write_text(template);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');text=p.read_text(encoding='utf-8');assert '## E168:' not in text;p.write_text(text+'\n## E168: privileged link-clearance constraints with frozen E120\n\n'+'; '.join(f"{c}: {v['outcomes']['success']}/48 success, {v['outcomes']['collisions']} collision episodes, {v['gains']} rescues/{v['losses']} regressions" for c,v in summary.items())+'. Baseline37/48 with11 collisions. Exact link geometry is privileged; fitting diagnostic only, release fix and scoring preserved. Report: `outputs/ease_link_clearance_diagnostic_v1/report.md`.\n',encoding='utf-8');save(ROOT/'status.json',dict(status='verified_videos_pending',new_episodes=96,baseline_exact_replays=4,videos_planned=len(video_rows)));print({c:{k:v for k,v in x.items() if k!='per_episode_minimum_clearance'} for c,x in summary.items()},flush=True)

def finish():
    videos=read(ROOT/'videos/manifest.json')['videos'];assert len(videos)==len(read(ROOT/'video_records.json'))
    for v in videos:assert v['all_saved_states_match_trace'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    save(ROOT/'status.json',dict(status='complete',new_episodes=96,baseline_exact_replays=4,videos=len(videos),privileged_geometry=True,E120_unchanged=True,release_fix_preserved=True,held_evaluated=False));save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print('E168 complete with verified success/failure videos',flush=True)

if __name__=='__main__':
    import sys
    {'run':run,'finish':finish}[sys.argv[1]]()
