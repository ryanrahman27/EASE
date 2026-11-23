"""E172 full trajectory audit, paired outcomes, report and video artifacts."""
from pathlib import Path
from collections import Counter
import shutil
import numpy as np
import mujoco
from .descent_stall import ROOT,BASE,REFERENCE,CONDITIONS,prior,geometry_ids,distances,distance_gradient
from .link_clearance_report import outcome_counts
from .abstaining_task_value_gate import read,save,fingerprint

def run():
    refs={'E120_release':(BASE,[r for r in read(BASE/'records.json') if r['dof']==6]),'E170_protected':(REFERENCE,read(REFERENCE/'records.json'))}
    summary={'references':{k:outcome_counts(rows,root) for k,(root,rows) in refs.items()},'conditions':{}};checks={};selected={}
    for c in CONDITIONS:
        root=ROOT/c;rows=read(root/'records.json');assert len(rows)==48 and read(root/'verification.json')['episodes']==48
        paired={}
        for key,(_,rs) in refs.items():
            lookup={r['case']:r for r in rs};assert set(lookup)=={r['case'] for r in rows}
            paired[key]=dict(rescues=[r['case'] for r in rows if r['success'] and not lookup[r['case']]['success']],losses=[r['case'] for r in rows if not r['success'] and lookup[r['case']]['success']])
        post_count=0;gradient_count=0;objective_checks=0;exact_retained=0;changed=Counter();fallbacks=0;first_contacts=[];allowances=[];probe_changes=0
        for r in rows:
            stem=f"{r['case']}_e120_ensemble_frozen_bias";t=read(root/'trials'/f'{stem}.json');ref=read(REFERENCE/'trials'/f'{stem}.json');logs=read(root/'clearance'/f'{stem}.json');states=np.load(root/'states'/f'{stem}.npz')['states'];assert len(states)==len(logs)==len(t['trace'])
            assert t['calibration']==ref['calibration'] and t['initial_estimator']==ref['initial_estimator'];assert t['initial_estimator']['bias']==t['final_estimator']['bias']
            n=next(i for i,l in enumerate(logs) if l['phase']=='descend');assert t['trace'][:n]==ref['trace'][:n];np.testing.assert_array_equal(states[:n],np.load(REFERENCE/'states'/f'{stem}.npz')['states'][:n])
            m=mujoco.MjModel.from_xml_path(str(root/'states'/f'{stem}.xml'));d=mujoco.MjData(m);scratch=mujoco.MjData(m);ids,table=geometry_ids(m,6,True);history=[];checked=False;cost_checked=False;descent_errors=[];recovery=False
            first=next((i for i,x in enumerate(t['trace']) if x['bad']),None)
            if first is not None:first_contacts.append(dict(case=r['case'],step=first,phase=logs[first]['phase'],pairs=t['trace'][first]['bad'],predicted_distance=logs[first]['predicted_distance'],distance_after=logs[first]['distance_after'],allowance=logs[first]['allowance']))
            for l,state in zip(logs,states):
                expected=np.zeros(9)
                if l['phase']=='descend':expected[6:]=.003 if c=='fixed_allowance' else .001+min(.020,max(history[-8:],default=0.))
                np.testing.assert_array_equal(expected,l['allowance']);effective=np.array(l['distance_before'])-expected/prior.ALPHA;np.testing.assert_array_equal(effective,l['effective_distance'])
                if l['phase'] in ('lower','release'):assert l['goal'][2]==.265
                np.testing.assert_array_equal(np.clip(np.array(l['q_before'])+np.clip(l['final_action'],-.04,.04),-2.5,2.5)-l['q_before'],l['executed_command'])
                d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_forward(m,d);np.testing.assert_allclose(distances(m,d,ids,table),l['distance_after'],atol=1e-12,rtol=0);post_count+=1
                if np.max(np.abs(np.array(l['final_action'])-l['nominal_action']))>1e-10:changed[l['phase']]+=1
                if l['phase']=='descend':allowances.append(float(expected[-1]))
                d.qpos[:]=l['full_qpos_before'];d.qvel[:]=l['full_qvel_before'];mujoco.mj_forward(m,d);site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center')
                if l['phase']=='descend':
                    pe=float(np.linalg.norm(np.array(l['goal'])-d.site_xpos[site]));descent_errors.append(pe)
                    if len(descent_errors)>=60 and .025<pe<.075 and descent_errors[-20]-pe<.001 and np.max(np.abs(l['q_before']))>=2.4:recovery=True
                expected_direct=bool(l['phase']=='descend' and recovery)
                assert l['descent_observations']==len(descent_errors)
                assert l['direct_task']==expected_direct
                if expected_direct:
                    d.qpos[:]=l['full_qpos_before'];d.qvel[:]=l['full_qvel_before'];mujoco.mj_forward(m,d)
                    site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');j=np.array(l['estimated_jacobian']);error=np.r_[np.array(l['goal'])-d.site_xpos[site],.15*(np.array([0.,0.,1.])-d.site_xmat[site].reshape(3,3)[:,2])]
                    posture=np.where(np.abs(l['q_before'])>2.35,-.02*np.sign(l['q_before']),0.);np.testing.assert_array_equal(posture,l['posture']);h=j.T@j+.03**2*np.eye(6);target=np.linalg.solve(h,j.T@(.35*error)+.03**2*posture)+l['bias'];np.testing.assert_allclose(target,l['optimization_target'],atol=1e-12,rtol=0);np.testing.assert_allclose(h,l['metric'],atol=1e-12,rtol=0)
                    if not cost_checked and np.max(np.abs(target))>.04 and l['solve']['solver_success']:
                        competitor,ci,_=prior.project(np.clip(target,-.04,.04),np.array(l['bias']),np.array(l['q_before']),effective,np.array(l['gradient']),np.array(l['metric']))
                        if ci['solver_success']:
                            h=np.array(l['metric']);cost=lambda u:.5*(u-target)@h@(u-target)
                            assert cost(np.array(l['final_action']))<=cost(competitor)+1e-7
                            objective_checks+=1;cost_checked=True
                else:np.testing.assert_array_equal(l['optimization_target'],l['nominal_action'])
                if l['eligible']:
                    g=np.array(l['gradient']);bias=np.array(l['bias']);u=np.array(l['final_action']);rhs=g@bias-prior.ALPHA*(effective-prior.MARGIN)
                    np.testing.assert_allclose(rhs,l['rhs'],atol=1e-14,rtol=0)
                    if l['solve']['solver_success']:assert np.min(g@u-rhs)>=-1e-7
                    if l['solve']['fallback']:
                        q=np.array(l['q_before']);np.testing.assert_array_equal(u,np.clip(bias,np.maximum(-.04,-2.5-q),np.minimum(.04,2.5-q)));fallbacks+=1
                    prediction=np.array(l['distance_before'])+g@(u-bias);np.testing.assert_allclose(prediction,l['predicted_distance'],atol=1e-14,rtol=0)
                    if l['phase']=='descend':history.append(float(max(0.,np.max(prediction[6:]-np.array(l['distance_after'])[6:]))))
                    if not checked:
                        d.qpos[:]=l['full_qpos_before'];d.qvel[:]=l['full_qvel_before'];mujoco.mj_forward(m,d);np.testing.assert_allclose(distances(m,d,ids,table),l['distance_before'],atol=1e-12,rtol=0);before=d.qpos.copy();np.testing.assert_allclose(distance_gradient(m,d,scratch,ids,table,6),g,atol=1e-10,rtol=0);np.testing.assert_array_equal(before,d.qpos);checked=True;gradient_count+=1
                else:np.testing.assert_array_equal(l['nominal_action'],l['final_action'])
            if not any(l['direct_task'] for l in logs):
                assert t['trace']==ref['trace'] and t['success']==ref['success'];np.testing.assert_array_equal(states,np.load(REFERENCE/'states'/f'{stem}.npz')['states']);exact_retained+=1
            probe_changes+=r['changed_probe_steps']
        counts=outcome_counts(rows,root);summary['conditions'][c]=dict(outcomes=counts,paired=paired,changed_commands_by_phase=dict(changed),fallbacks=fallbacks,probe_changed_commands=probe_changes,allowance_quantiles_m=np.quantile(allowances,[0,.5,.9,1]).tolist(),first_contacts=first_contacts,strict_screen=counts['success']>44 and counts['collisions']==0 and not paired['E170_protected']['losses'])
        checks[c]=dict(exact_nonactivated_episodes=exact_retained,post_state_checks=post_count,gradient_checks=gradient_count,matched_feasible_objective_checks=objective_checks,direct_task_targets_and_gate_verified=True,causal_allowances_verified=True,actual_commands_verified=True,accepted_constraints_verified=True,pre_descent_exact=True,calibration_and_frozen_bias_preserved=True)
        chosen={26,62,65,67}
        for category in ([r['case'] for r in rows if r['success']],[r['case'] for r in rows if not r['success'] and not r['collision']],paired['E170_protected']['rescues'],paired['E170_protected']['losses']):
            if category:chosen.add(category[0])
        selected[c]=sorted(chosen)
    for p,h in read(ROOT/'frozen.json').items():assert fingerprint(p)==h
    save(ROOT/'summary.json',summary);save(ROOT/'verification.json',dict(conditions=checks,new_interventions=48,exact_replays=4,held_evaluated=False,privileged=True))
    videos=[]
    for folder in ('states','trials'):(ROOT/folder).mkdir(exist_ok=True)
    for c,cases in selected.items():
        for case in cases:
            for source,method in [(ROOT/c,'privileged_'+c),(REFERENCE,'privileged_E170'),(BASE,'baseline_E120_release')]:
                if any(v['case']==case and v['method']==method for v in videos):continue
                old=f'{case}_e120_ensemble_frozen_bias';new=f'{case}_{method}_frozen_bias';t=read(source/'trials'/f'{old}.json');t['method']=method;save(ROOT/'trials'/f'{new}.json',t)
                for ext in ('xml','npz'):shutil.copy2(source/'states'/f'{old}.{ext}',ROOT/'states'/f'{new}.{ext}')
                videos.append(dict(case=case,method=method,mode='frozen_bias',success=t['success'],dof=6))
    save(ROOT/'video_records.json',videos)
    lines=['# E172: selective late-stall descent recovery','','Privileged fitting development following rejected E171 variants. One fixed candidate evaluated on all48 six-joint fitting placements;4 exact E170 replays. E120, measured-error allowance, probe protection, release fix, phase budgets and scoring remain unchanged.','','| Controller | Success /48 | Collision episodes | Collision-free failures | Rescues / losses vs E170 |','|---|---:|---:|---:|---|']
    for k,v in summary['references'].items():lines.append(f"| {k} | {v['success']} | {v['collisions']} | {v['collision_free_failures']} | — |")
    for c,v in summary['conditions'].items():
        x=v['outcomes'];p=v['paired']['E170_protected'];lines.append(f"| {c} | {x['success']} | {x['collisions']} | {x['collision_free_failures']} | {p['rescues']} / {p['losses']} |")
    lines+=['','E171 direct task optimization and its joint-limit gate each completed43/48 and lost a previous success. They are rejected and their complete results/videos remain in ../ease_descent_progress_v1. E172 is an additional adaptive fitting experiment, not independent validation.','', 'Activation uses measured state only: at least60 descent observations,25–75mm position error, less than1mm progress over the previous19 control intervals, and any joint magnitude at least2.4rad. Recovery stays active for the remainder of descent. Earlier broader stall rules also activated on successful case43; their read-only diagnoses are retained. The local late-stall rule activates only on the four reference failures in the fitting diagnosis. No case identifiers enter the controller.','', 'Recovery minimizes the E120-estimated task residual with a posture regularizer: joints with magnitude above2.35rad prefer a0.02rad command toward the center; others prefer zero relative command. Regularization scale0.03 replaces0.015 only during recovery. The same QP bounds and privileged clearance constraints still apply. Outside recovery, the controller is exactly E170. No extra steps, new task Jacobian, oracle object state, goal changes, or model training are used.','', 'Audit independently reconstructs every activation decision, posture vector, objective center/metric, causal clearance allowance, accepted inequality and executed command. Every post-state geometry is checked, along with full-state gradient reconstruction and exact complete state/trace equality for nonactivated episodes. Four disabled-recovery reference replays passed first. Video replays verify saved tool/object traces and decoded frame counts;80ms sampling can miss contacts scored at2ms.','', 'A positive task-success screen requires more than44 successes, zero collisions and no lost E170 success. More local progress without a completed task is reported separately. E120+release remains the original baseline. All results are fitting-only and geometry-privileged; no safety guarantee or unseen-body claim is made.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vals=list(summary['references'].values())+[summary['conditions'][c]['outcomes'] for c in CONDITIONS];labels=['E120','E170',*CONDITIONS];fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key in zip(axes,['success','collisions','collision_free_failures']):
        ns=[v[key] for v in vals];ax.bar(labels,ns);ax.set_title(key);ax.set_ylim(0,48);ax.tick_params(axis='x',rotation=25)
        for i,n in enumerate(ns):ax.text(i,n+.5,str(n),ha='center')
    fig.suptitle('E172 privileged selective descent recovery | fitting only');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    template=Path(__file__).with_name('link_clearance_videos.py').read_text().replace('link_clearance_diagnostic import ROOT','descent_stall import ROOT').replace('E168','E172').replace('link-clearance comparison','selective descent recovery comparison').replace('PRIVILEGED link/table constraints around E120 versus unchanged E120+release. Fitting cases only; successes, failures and stalls retained.','PRIVILEGED selective stall recovery with unchanged E170 protection; matched successes and failures on fitting cases.')
    Path(__file__).with_name('descent_stall_videos.py').write_text(template)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'status.json',dict(status='verified_videos_pending',videos=len(videos)))
    print({c:{k:v for k,v in x.items() if k!='first_contacts'} for c,x in summary['conditions'].items()},flush=True)

def finish():
    videos=read(ROOT/'videos/manifest.json')['videos'];assert len(videos)==len(read(ROOT/'video_records.json'))
    for v in videos:assert v['all_saved_states_match_trace'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    s=read(ROOT/'summary.json');text='; '.join(f"{c}: {v['outcomes']['success']}/48 success, {v['outcomes']['collisions']} collisions, {v['paired']['E170_protected']} versus E170" for c,v in s['conditions'].items())
    section='\n## E172: selective descent stall recovery\n\n'+text+'. E170 protected reference44/48 with0 collisions; E120+release37/48. Privileged fitting diagnostic only,48 new episodes,4 exact replays. E120/probe protection/release preserved. Report: `outputs/ease_descent_stall_v1/report.md`; saved success/failure videos: `videos/index.html`.\n'
    comparisons=read(ROOT/'progress_comparison.json')
    section+='\nAll44 successful reference episodes reproduce exactly. Final descent position errors (mm, reference -> recovery): '+ '; '.join(f"case{r['case']}: {r['conditions']['E170']['terminal_error_mm']:.2f} -> {r['conditions']['E172']['terminal_error_mm']:.2f}" for r in comparisons)+'. No new completed tasks. Orientation tradeoffs and late plateaus remain; joint5 stays near its upper limit. Broader E171 changes were rejected because each lost one success. Next useful diagnosis is collision-free posture/route feasibility, not an assumed benefit from more time.\n'
    review=Path('outputs/ease_iteration_review_v1')
    for p in (Path(__file__).with_name('TECHNICAL_REPORT.md'),review/'report.md'):
        old=p.read_text(encoding='utf-8')
        if '## E172:' not in old:p.write_text(old+section,encoding='utf-8')
    p=review/'WORKING_STATE.md';old=p.read_text(encoding='utf-8')
    if not old.startswith('# Latest checkpoint: E172'):p.write_text('# Latest checkpoint: E172 complete\n'+section+'\nAll runs, audits and videos complete.\n\n---\n'+old,encoding='utf-8')
    studies=read(review/'studies.json')
    if not any(x['experiment']=='E172' for x in studies):studies.append(dict(experiment='E172',folder=str(ROOT),report_exists=True,videos=len(videos)));save(review/'studies.json',studies)
    save(ROOT/'status.json',dict(status='complete',new_episodes=48,exact_replays=4,videos=len(videos),held_evaluated=False,E120_unchanged=True,probe_protection_preserved=True))
    save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print('E172 complete',len(videos),'videos',flush=True)

if __name__=='__main__':
    import sys
    finish() if len(sys.argv)>1 and sys.argv[1]=='finish' else run()
