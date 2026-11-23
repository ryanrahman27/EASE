"""E173 independent route, causal-controller, physical-outcome and video audit."""
from pathlib import Path
import shutil
import numpy as np
import mujoco
from .posture_route import ROOT,BASE,REFERENCE,CONDITIONS,prior,geometry_ids,distances,distance_gradient
from .posture_route_planner import forbidden
from .link_clearance_report import outcome_counts
from .abstaining_task_value_gate import read,save,fingerprint

def run():
    refs={'E120_release':(BASE,[r for r in read(BASE/'records.json') if r['dof']==6]),'E170_protected':(REFERENCE,read(REFERENCE/'records.json'))};summary={'references':{k:outcome_counts(rows,root) for k,(root,rows) in refs.items()},'conditions':{}};checks={};video_rows=[]
    for condition in CONDITIONS:
        root=ROOT/condition;rows=read(root/'records.json');assert len(rows)==48 and read(root/'verification.json')['episodes']==48
        paired={}
        for label,(_,rs) in refs.items():
            lookup={r['case']:r for r in rs};assert set(lookup)=={r['case'] for r in rows};paired[label]=dict(rescues=[r['case'] for r in rows if r['success'] and not lookup[r['case']]['success']],losses=[r['case'] for r in rows if not r['success'] and lookup[r['case']]['success']])
        post_checks=0;gradient_checks=0;path_checks=0;retained=0;fallbacks=0;route_cases=[];extra_steps=0
        for r in rows:
            case=r['case'];stem=f'{case}_e120_ensemble_frozen_bias';t=read(root/'trials'/f'{stem}.json');ref=read(REFERENCE/'trials'/f'{stem}.json');logs=read(root/'clearance'/f'{stem}.json');states=np.load(root/'states'/f'{stem}.npz')['states'];assert len(logs)==len(states)==len(t['trace'])
            assert t['calibration']==ref['calibration'] and t['initial_estimator']==ref['initial_estimator'];assert t['initial_estimator']['bias']==t['final_estimator']['bias']
            m=mujoco.MjModel.from_xml_path(str(root/'states'/f'{stem}.xml'));d=mujoco.MjData(m);scratch=mujoco.MjData(m);ids,table=geometry_ids(m,6,True);site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');es=[];errors=[];attempted=False;route=None;index=0;checked=False;trigger_step=None
            for i,(l,state) in enumerate(zip(logs,states)):
                d.qpos[:]=l['full_qpos_before'];d.qvel[:]=l['full_qvel_before'];mujoco.mj_forward(m,d)
                if l['phase']=='descend':
                    pe=float(np.linalg.norm(np.array(l['goal'])-d.site_xpos[site]));es.append(pe)
                    if not attempted and len(es)>=60 and .025<pe<.075 and es[-20]-pe<.001 and np.max(np.abs(l['q_before']))>=2.4:
                        attempted=True;trigger_step=i;plan=read(root/'planning'/f'{case}.json');np.testing.assert_array_equal(plan['start_qpos'],l['full_qpos_before']);np.testing.assert_array_equal(plan['start_qvel'],l['full_qvel_before']);np.testing.assert_array_equal(plan['goal'],l['goal'])
                        assert t['trace'][:i]==ref['trace'][:i];np.testing.assert_array_equal(states[:i],np.load(REFERENCE/'states'/f'{stem}.npz')['states'][:i])
                        offline=read(ROOT/'plans'/f'{case}.json');assert plan['available']==offline['available'];assert plan['route']==offline['route']
                        if plan['available']:
                            route=np.array(plan['dense_path']);np.testing.assert_array_equal(route[0],l['q_before'])
                            assert np.max(np.abs(np.diff(route,axis=0)))<=.025+1e-12
                            for q in route:
                                scratch.qpos[:]=plan['start_qpos'];scratch.qvel[:]=plan['start_qvel'];scratch.qpos[:6]=q;mujoco.mj_forward(m,scratch);assert np.min(distances(m,scratch,ids,table))>=0 and not forbidden(m,scratch);path_checks+=1
                            assert np.linalg.norm(scratch.site_xpos[site]-l['goal'])<.003;assert np.linalg.norm(scratch.site_xmat[site].reshape(3,3)[:,2]-[0,0,1])<.04;assert np.max(np.abs(route[-1]))<=2.35+1e-12
                active=bool(l['phase']=='descend' and route is not None);assert l['route_active']==active and l['planning_attempted']==attempted and l['descent_observations']==len(es)
                if active:
                    index+=int(np.argmin(np.linalg.norm(route[index:]-l['q_before'],axis=1)));waypoint=route[min(index+3,len(route)-1)];np.testing.assert_array_equal(waypoint,l['waypoint']);target=np.clip(.35*(waypoint-l['q_before'])+np.array(l['bias']),-.04,.04)
                else:target=np.array(l['nominal_action']);assert l['waypoint'] is None
                assert l['route_index']==index;np.testing.assert_array_equal(target,l['optimization_target'])
                allowance=np.zeros(9)
                if l['phase']=='descend':allowance[6:]=.001+min(.020,max(errors[-8:],default=0.))
                np.testing.assert_array_equal(allowance,l['allowance']);effective=np.array(l['distance_before'])-allowance/prior.ALPHA;np.testing.assert_array_equal(effective,l['effective_distance']);assert l['eligible']==bool(active or np.min(effective)<=prior.ACTIVATION)
                np.testing.assert_array_equal(np.clip(np.array(l['q_before'])+np.clip(l['final_action'],-.04,.04),-2.5,2.5)-l['q_before'],l['executed_command'])
                if l['phase'] in ('lower','release'):assert l['goal'][2]==.265
                if l['eligible']:
                    g=np.array(l['gradient']);u=np.array(l['final_action']);bias=np.array(l['bias']);j=np.array(l['estimated_jacobian']);np.testing.assert_allclose(j.T@j+.015**2*np.eye(6),l['metric'],atol=1e-12,rtol=0);rhs=g@bias-prior.ALPHA*(effective-prior.MARGIN);np.testing.assert_allclose(rhs,l['rhs'],atol=1e-14,rtol=0)
                    if l['solve']['solver_success']:assert np.min(g@u-rhs)>=-1e-7
                    if l['solve']['fallback']:
                        q=np.array(l['q_before']);np.testing.assert_array_equal(u,np.clip(bias,np.maximum(-.04,-2.5-q),np.minimum(.04,2.5-q)));fallbacks+=1
                    prediction=np.array(l['distance_before'])+g@(u-bias);np.testing.assert_allclose(prediction,l['predicted_distance'],atol=1e-14,rtol=0)
                    if l['phase']=='descend':errors.append(float(max(0.,np.max(prediction[6:]-np.array(l['distance_after'])[6:]))))
                    if not checked:
                        np.testing.assert_allclose(distances(m,d,ids,table),l['distance_before'],atol=1e-12,rtol=0);live=d.qpos.copy();np.testing.assert_allclose(distance_gradient(m,d,scratch,ids,table,6),g,atol=1e-10,rtol=0);np.testing.assert_array_equal(d.qpos,live);checked=True;gradient_checks+=1
                else:np.testing.assert_array_equal(target,l['final_action'])
                d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_forward(m,d);np.testing.assert_allclose(distances(m,d,ids,table),l['distance_after'],atol=1e-12,rtol=0);post_checks+=1
            if not attempted:
                assert t['trace']==ref['trace'] and t['success']==ref['success'];np.testing.assert_array_equal(states,np.load(REFERENCE/'states'/f'{stem}.npz')['states']);retained+=1
            desc=next(p for p in t['phases'] if p['phase']=='descend');assert desc['steps']==len(es) and len(es)<=(300 if condition=='route_extended' and route is not None else 100);extra_steps+=max(0,len(es)-100)
            if attempted:
                last=next(x for x in reversed(t['trace']) if x['phase']=='descend');route_cases.append(dict(case=case,route_available=route is not None,success=r['success'],collision=r['collision'],trigger_step=trigger_step,descent_steps=len(es),extra_descent_steps=max(0,len(es)-100),last_route_index=index,path_nodes=0 if route is None else len(route),terminal_descent_error_mm=1000*last['position_error'],terminal_rotation_error=last['rotation_error'],failed_phases=[p['phase'] for p in t['phases'] if not p['reached']]))
        counts=outcome_counts(rows,root);summary['conditions'][condition]=dict(outcomes=counts,paired=paired,route_cases=route_cases,extra_descent_steps=extra_steps,extra_simulated_seconds=.04*extra_steps,fallbacks=fallbacks,strict_screen=counts['success']>44 and counts['collisions']==0 and not paired['E170_protected']['losses']);checks[condition]=dict(exact_nonactivated_episodes=retained,post_state_geometry_checks=post_checks,gradient_checks=gradient_checks,independent_static_path_points=path_checks,causal_gate_and_allowance=True,online_plans_equal_static_diagnosis=True,waypoints_and_actual_commands_verified=True,accepted_constraints_verified=True,budgets_verified=True)
        for case in (0,26,62,65,67):
            for source,method in ((root,'privileged_'+condition),(REFERENCE,'privileged_E170'),(BASE,'baseline_E120_release')):
                if any(v['case']==case and v['method']==method for v in video_rows):continue
                for folder in ('trials','states'):(ROOT/folder).mkdir(exist_ok=True)
                old=f'{case}_e120_ensemble_frozen_bias';new=f'{case}_{method}_frozen_bias';t=read(source/'trials'/f'{old}.json');t['method']=method;save(ROOT/'trials'/f'{new}.json',t)
                for ext in ('npz','xml'):shutil.copy2(source/'states'/f'{old}.{ext}',ROOT/'states'/f'{new}.{ext}')
                video_rows.append(dict(case=case,method=method,mode='frozen_bias',success=t['success'],dof=6))
    for p,h in read(ROOT/'frozen.json').items():assert fingerprint(p)==h
    save(ROOT/'summary.json',summary);save(ROOT/'verification.json',dict(conditions=checks,new_episodes=96,exact_replays=4,privileged=True,held_evaluated=False));save(ROOT/'video_records.json',video_rows)
    lines=['# E173: alternative postures and route execution','','Privileged fitting diagnostic. E120, frozen bias, probe protection and release265mm are unchanged. Static planner found alternative grasp postures and sampled collision-free routes for all four failures. Dynamic execution is evaluated separately below.','','| Controller | Success /48 | Collision episodes | Collision-free failures | Rescues / losses vs E170 | Extra descent steps |','|---|---:|---:|---:|---|---:|']
    for label,x in summary['references'].items():lines.append(f"| {label} | {x['success']} | {x['collisions']} | {x['collision_free_failures']} | — | 0 |")
    for c,v in summary['conditions'].items():
        x=v['outcomes'];p=v['paired']['E170_protected'];lines.append(f"| {c} | {x['success']} | {x['collisions']} | {x['collision_free_failures']} | {p['rescues']} / {p['losses']} | {v['extra_descent_steps']} |")
    lines+=['','All96 complete fitting episodes and4 exact protected-reference replays are retained. Neither variant changes the learned model. The online planner is activated only by the previously specified measured late-local-stall gate. It uses exact simulator geometry, object/jaw state and finite kinematics in scratch data,24 IK starts and bounded path search. No physics advances occur during planning; live state is asserted unchanged. All four accepted routes were direct joint-space interpolations to alternative postures, so no RRT route was required.','', 'The route tracker uses nearest remaining dense path node plus3-node lookahead, original joint command clipping, frozen measured bias and the unchanged E170 clearance/error-allowance QP with E120 metric. It takes over descent only; E120 handles all other phases. Probe schedule/aperture, calibration, lower/release goals and strict contact/task scoring remain unchanged. The route is privileged assistance, not learned adaptation.','', 'route_fixed keeps the original100-step descent budget. route_extended allows up to300 steps only following accepted planning, with unchanged reach termination. Extra interactions and simulated time are reported, not hidden as policy improvement. Static paths hold jaw/object state fixed and check discrete samples; they do not certify continuous or dynamic safety. Full episode collisions and task failures remain scored.','', 'The audit reconstructs every gate, plan source state, waypoint, allowance and executed command; independently checks saved path samples, all post-state geometry and representative gradients; verifies accepted constraints and actual descent budgets; and checks complete trace/state equality for nonactivated episodes. Online routes match the pre-execution static diagnosis. Video replays verify saved tool/object states and frame counts;80ms video samples can miss contacts scored at2ms.','', 'See summary.json for all four route outcomes, failure phases and endpoint errors. Publication-readiness assessment, evidence gaps and primary-source related-work links are saved in publication_readiness.md. This study is fitting-only; no unseen-body result or superiority to a flow-matching policy is claimed.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vals=list(summary['references'].values())+[summary['conditions'][c]['outcomes'] for c in CONDITIONS];labels=['E120','E170',*CONDITIONS];fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key in zip(axes,('success','collisions','collision_free_failures')):
        ns=[v[key] for v in vals];ax.bar(labels,ns);ax.set_title(key);ax.set_ylim(0,48);ax.tick_params(axis='x',rotation=25)
        for i,n in enumerate(ns):ax.text(i,n+.5,str(n),ha='center')
    fig.suptitle('E173 privileged route assistance | fitting only | budgets differ');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    template=Path(__file__).with_name('link_clearance_videos.py').read_text().replace('link_clearance_diagnostic import ROOT','posture_route import ROOT').replace('E168','E173').replace('link-clearance comparison','posture/route comparison').replace('PRIVILEGED link/table constraints around E120 versus unchanged E120+release. Fitting cases only; successes, failures and stalls retained.','PRIVILEGED route execution, original vs extended descent budget. Matched E170 and E120+release; fitting-only successes and failures.')
    Path(__file__).with_name('posture_route_videos.py').write_text(template);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'status.json',dict(status='verified_videos_pending',videos=len(video_rows)));print(summary['conditions'],flush=True)

def finish():
    videos=read(ROOT/'videos/manifest.json')['videos'];assert len(videos)==len(read(ROOT/'video_records.json'))
    for v in videos:assert v['all_saved_states_match_trace'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    s=read(ROOT/'summary.json');paragraph='; '.join(f"{c}: {v['outcomes']['success']}/48 success, {v['outcomes']['collisions']} collisions, {v['paired']['E170_protected']} versus E170; {v['extra_descent_steps']} extra descent steps" for c,v in s['conditions'].items());section='\n## E173: privileged alternative-posture route execution\n\n'+paragraph+'. Static alternatives exist for all four failures.96 full fitting episodes and4 exact replays; unchanged E120/probe protection/release.44 nonactivated trajectories checked exactly per variant. Privileged route assistance and extra interaction budget are explicitly labeled. Report: `outputs/ease_posture_route_v1/report.md`; videos: `videos/index.html`; publication assessment: `publication_readiness.md`. No held test.\n'
    section+='\nNeither route variant is promoted. Extended routes reach the descent criterion for cases26 and67, but their objects move about154mm and73mm during descent and are not grasped. Case65 ends12mm from the pose target with77mm object motion; case62 remains114mm from the target. Static pose feasibility therefore does not resolve dynamic route tracking and grasp handoff. The zero-collision44/48 reference remains unchanged. Publication-critical learned-versus-analytical unseen-body evidence is still missing; a prospective known-body placement comparison previously tied41/48.\n'
    review=Path('outputs/ease_iteration_review_v1')
    for p in (Path(__file__).with_name('TECHNICAL_REPORT.md'),review/'report.md'):
        old=p.read_text(encoding='utf-8')
        if '## E173:' not in old:p.write_text(old+section,encoding='utf-8')
    p=review/'WORKING_STATE.md';old=p.read_text(encoding='utf-8')
    if not old.startswith('# Latest checkpoint: E173'):p.write_text('# Latest checkpoint: E173 complete\n'+section+'\nAll runs, audits and videos complete.\n\n---\n'+old,encoding='utf-8')
    studies=read(review/'studies.json')
    if not any(x['experiment']=='E173' for x in studies):studies.append(dict(experiment='E173',folder=str(ROOT),report_exists=True,videos=len(videos)));save(review/'studies.json',studies)
    save(ROOT/'status.json',dict(status='complete',new_episodes=96,exact_replays=4,videos=len(videos),held_evaluated=False,E120_unchanged=True,probe_protection_preserved=True));save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print('E173 complete',len(videos),'verified videos',flush=True)

if __name__=='__main__':
    import sys
    finish() if len(sys.argv)>1 and sys.argv[1]=='finish' else run()
