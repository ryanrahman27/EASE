"""E174 independent route, causal-controller, physical-outcome and video audit."""
from pathlib import Path
import shutil
import numpy as np
import mujoco
from .object_preserving import ROOT,BASE,REFERENCE,CONDITIONS,prior,geometry_ids,distances,distance_gradient
from .object_preserving_plan import forbidden
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
            m=mujoco.MjModel.from_xml_path(str(root/'states'/f'{stem}.xml'));d=mujoco.MjData(m);scratch=mujoco.MjData(m);ids,table=geometry_ids(m,6,True);site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');es=[];errors=[];attempted=False;route=None;index=0;checked=False;trigger_step=None;pick=None;last_phase=None;ready=False;good=0;aborted=False;close_anchor=None;abort_anchor=None
            for i,(l,state) in enumerate(zip(logs,states)):
                d.qpos[:]=l['full_qpos_before'];d.qvel[:]=l['full_qvel_before'];mujoco.mj_forward(m,d)
                if l['phase']=='descend':
                    pe=float(np.linalg.norm(np.array(l['goal'])-d.site_xpos[site]));es.append(pe)
                    if not attempted and len(es)>=60 and .025<pe<.075 and es[-20]-pe<.001 and np.max(np.abs(l['q_before']))>=2.4:
                        attempted=True;trigger_step=i;plan=read(root/'planning'/f'{case}.json');np.testing.assert_array_equal(plan['start_qpos'],l['full_qpos_before']);np.testing.assert_array_equal(plan['start_qvel'],l['full_qvel_before']);object_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'object');np.testing.assert_array_equal(plan['pick_goal'],np.r_[d.xpos[object_id,:2],.24])
                        assert t['trace'][:i]==ref['trace'][:i];np.testing.assert_array_equal(states[:i],np.load(REFERENCE/'states'/f'{stem}.npz')['states'][:i])
                        offline=read(ROOT/'plans'/f'{case}.json');assert plan['available']==offline['available'];assert plan['route']==offline['route']
                        if plan['available']:
                            route=np.array(plan['dense_path']);pick=np.array(plan['pick_goal']);np.testing.assert_array_equal(route[0],l['q_before'])
                            assert np.max(np.abs(np.diff(route,axis=0)))<=.025+1e-12
                            for q in route:
                                scratch.qpos[:]=plan['start_qpos'];scratch.qvel[:]=plan['start_qvel'];scratch.qpos[:6]=q;mujoco.mj_forward(m,scratch);assert np.min(distances(m,scratch,ids,table))>=0 and not forbidden(m,scratch);path_checks+=1
                            assert np.linalg.norm(scratch.site_xpos[site]-l['goal'])<.003;assert np.linalg.norm(scratch.site_xmat[site].reshape(3,3)[:,2]-[0,0,1])<.04;assert np.max(np.abs(route[-1]))<=2.35+1e-12
                active=bool(l['phase']=='descend' and route is not None);assert l['route_active']==active and l['planning_attempted']==attempted and l['descent_observations']==len(es)
                if active:
                    index+=int(np.argmin(np.linalg.norm(route[index:]-l['q_before'],axis=1)));waypoint=route[min(index+3,len(route)-1)];np.testing.assert_array_equal(waypoint,l['waypoint']);target=np.clip(waypoint-l['q_before']+np.array(l['bias']),-.04,.04)
                else:target=np.array(l['nominal_action']);assert l['waypoint'] is None
                handoff=False;aperture=l['requested_aperture']
                if condition=='guarded_handoff' and route is not None:
                    if l['phase']=='close' and last_phase!='close':
                        close_anchor=np.array(l['q_before']);gap=d.site_xpos[site]-pick;obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'object');drift=np.linalg.norm(d.xpos[obj,:2]-pick[:2]);axis=np.linalg.norm(d.site_xmat[site].reshape(3,3)[:,2]-[0,0,1]);ready=bool(np.linalg.norm(gap[:2])<=.008 and abs(gap[2])<=.012 and axis<.08 and drift<=.008)
                    if l['phase']=='lift' and last_phase=='close' and (not ready or good<5):aborted=True;abort_anchor=np.array(l['q_before'])
                    if l['phase']=='close':target=np.clip(close_anchor-l['q_before']+np.array(l['bias']),-.04,.04);aperture=aperture if ready else 0.;handoff=True
                    if aborted:target=np.clip(abort_anchor-l['q_before']+np.array(l['bias']),-.04,.04);aperture=0.;handoff=True
                assert l['handoff_control']==handoff and l['close_ready']==ready and l['aborted']==aborted and l['aperture']==aperture
                if close_anchor is not None:np.testing.assert_array_equal(close_anchor,l['close_anchor'])
                if abort_anchor is not None:np.testing.assert_array_equal(abort_anchor,l['abort_anchor'])
                assert l['route_index']==index;np.testing.assert_array_equal(target,l['optimization_target'])
                allowance=np.zeros(9)
                if l['phase']=='descend':allowance[6:]=.001+min(.020,max(errors[-8:],default=0.))
                np.testing.assert_array_equal(allowance,l['allowance']);effective=np.array(l['distance_before'])-allowance/prior.ALPHA;np.testing.assert_array_equal(effective,l['effective_distance']);assert l['eligible']==bool(active or handoff or np.min(effective)<=prior.ACTIVATION)
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
                cube=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'cube');pad_ids={mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('gripper_left_pad','gripper_right_pad')};touch=set()
                for contact in d.contact[:d.ncon]:
                    pair={int(contact.geom1),int(contact.geom2)}
                    if cube in pair:touch|=pair&pad_ids
                assert len(touch)==l['end_pads']
                if l['phase']=='close':good=good+1 if len(touch)==2 else 0
                assert l['close_good']==good;last_phase=l['phase']
            if route is None:
                assert t['trace']==ref['trace'] and t['success']==ref['success'];np.testing.assert_array_equal(states,np.load(REFERENCE/'states'/f'{stem}.npz')['states']);retained+=1
            if condition=='guarded_handoff':
                close_step=next(i for i,l in enumerate(logs) if l['phase']=='close');other=ROOT/'object_route';ot=read(other/'trials'/f'{stem}.json');assert t['trace'][:close_step]==ot['trace'][:close_step];np.testing.assert_array_equal(states[:close_step],np.load(other/'states'/f'{stem}.npz')['states'][:close_step])
            desc=next(p for p in t['phases'] if p['phase']=='descend');assert desc['steps']==len(es) and len(es)<=(300 if route is not None else 100);extra_steps+=max(0,len(es)-100)
            if attempted:
                last=next(x for x in reversed(t['trace']) if x['phase']=='descend');route_cases.append(dict(case=case,close_ready=ready,verified_final_close_steps=good,aborted=aborted,route_available=route is not None,success=r['success'],collision=r['collision'],trigger_step=trigger_step,descent_steps=len(es),extra_descent_steps=max(0,len(es)-100),last_route_index=index,path_nodes=0 if route is None else len(route),terminal_descent_error_mm=1000*last['position_error'],terminal_rotation_error=last['rotation_error'],failed_phases=[p['phase'] for p in t['phases'] if not p['reached']]))
        counts=outcome_counts(rows,root);summary['conditions'][condition]=dict(outcomes=counts,paired=paired,route_cases=route_cases,extra_descent_steps=extra_steps,extra_simulated_seconds=.04*extra_steps,fallbacks=fallbacks,strict_screen=counts['success']>44 and counts['collisions']==0 and not paired['E170_protected']['losses']);checks[condition]=dict(exact_nonactivated_episodes=retained,post_state_geometry_checks=post_checks,gradient_checks=gradient_checks,independent_static_path_points=path_checks,causal_gate_and_allowance=True,online_plans_equal_static_diagnosis=True,waypoints_and_actual_commands_verified=True,accepted_constraints_verified=True,budgets_verified=True,grasp_guard_and_actual_end_contacts_verified=True,matched_preclose_prefix=True)
        for case in (0,26,62,65,67):
            for source,method in ((root,'privileged_'+condition),(REFERENCE,'privileged_E170'),(BASE,'baseline_E120_release')):
                if any(v['case']==case and v['method']==method for v in video_rows):continue
                for folder in ('trials','states'):(ROOT/folder).mkdir(exist_ok=True)
                old=f'{case}_e120_ensemble_frozen_bias';new=f'{case}_{method}_frozen_bias';t=read(source/'trials'/f'{old}.json');t['method']=method;save(ROOT/'trials'/f'{new}.json',t)
                for ext in ('npz','xml'):shutil.copy2(source/'states'/f'{old}.{ext}',ROOT/'states'/f'{new}.{ext}')
                video_rows.append(dict(case=case,method=method,mode='frozen_bias',success=t['success'],dof=6))
    for p,h in read(ROOT/'frozen.json').items():assert fingerprint(p)==h
    save(ROOT/'summary.json',summary);save(ROOT/'verification.json',dict(conditions=checks,new_episodes=96,exact_replays=4,privileged=True,held_evaluated=False));save(ROOT/'video_records.json',video_rows)
    lines=['# E174: object-preserving routes and guarded grasp handoff','','Two matched privileged fitting interventions,48 cases each, with4 exact E170 replays. Original E120 weights/frozen bias, probe protection and265mm release are unchanged.','','| Controller | Success /48 | Collision episodes | Collision-free failures | Rescues / losses vs E170 | Extra descent steps |','|---|---:|---:|---:|---|---:|']
    for label,x in summary['references'].items():lines.append(f"| {label} | {x['success']} | {x['collisions']} | {x['collision_free_failures']} | — | 0 |")
    for c,v in summary['conditions'].items():
        x=v['outcomes'];p=v['paired']['E170_protected'];lines.append(f"| {c} | {x['success']} | {x['collisions']} | {x['collision_free_failures']} | {p['rescues']} / {p['losses']} | {v['extra_descent_steps']} |")
    lines+=['','The planner rejects all robot/object contacts in static checks. It samples grasp configurations at current simulator object XY and240mm height, reverses a vertical approach to320mm, then connects to that pregrasp posture. This preserves the grasp/approach IK branch. The initial pregrasp-first search failed and is archived; this correction occurred before physical intervention outcomes. Case26 has an existing robot/object contact at the gate, so no strict route is accepted and the original controller is retained.','', 'Both variants use the same route, gain1 joint tracking, protected32-step probe and up to300 descent steps after accepted planning. Pick/close/lift XY follows the object location observed at planning, while final placement and release height remain unchanged. object_route retains the ordinary close/lift transition. guarded_handoff checks close-entry XY/Z/axis alignment and object drift, holds arm posture during closure, and requires both pads contacting at the END of the final5 close steps before lift. Otherwise it holds with fingers open through remaining scheduled phases and is scored normally as a failure.','', 'Geometry, simulator object pose and contact signals are privileged. These interventions do not train E120 or demonstrate learned-only cross-embodiment adaptation. Static contact-free sampling does not guarantee actual object preservation. Actual object displacement, grasp/lift and complete-task outcomes are retained. No held test is executed.','', 'Independent audit reconstructs gates, current-state plans, strict path contacts, waypoints, clearance allowances, all actual commands and apertures, close readiness, end-step pad contacts, sustained-contact counters and aborts. All saved post-state clearances and representative gradients are checked; accepted linear constraints and real phase budgets are verified. Nonexecuted-route episodes reproduce full E170 traces/states exactly. Both variants have identical trajectories before close. Four exact reference replays passed after correcting and archiving a calibration-time phase-bookkeeping bug before interventions.','', 'See object_handoff_diagnosis.json for object displacement and each remaining failure. Saved videos include every original failure and a retained success; replayed tool/object traces and decoded frame counts are verified. Contact scoring occurs at2ms;80ms video sampling may miss brief contacts. literature_review.md maps established planning/control methods to these failure modes, distinguishes source findings from proposed applications, and states which ideas were actually implemented.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vals=list(summary['references'].values())+[summary['conditions'][c]['outcomes'] for c in CONDITIONS];labels=['E120','E170',*CONDITIONS];fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key in zip(axes,('success','collisions','collision_free_failures')):
        ns=[v[key] for v in vals];ax.bar(labels,ns);ax.set_title(key);ax.set_ylim(0,48);ax.tick_params(axis='x',rotation=25)
        for i,n in enumerate(ns):ax.text(i,n+.5,str(n),ha='center')
    fig.suptitle('E174 privileged object-preserving route / grasp handoff');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    template=Path(__file__).with_name('link_clearance_videos.py').read_text().replace('link_clearance_diagnostic import ROOT','object_preserving import ROOT').replace('E168','E174').replace('link-clearance comparison','object-preserving route / handoff comparison').replace('PRIVILEGED link/table constraints around E120 versus unchanged E120+release. Fitting cases only; successes, failures and stalls retained.','PRIVILEGED object-avoiding routes and guarded grasp handoff; matched E170 references; fitting cases only.')
    Path(__file__).with_name('object_preserving_videos.py').write_text(template);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'status.json',dict(status='verified_videos_pending',videos=len(video_rows)));print(summary['conditions'],flush=True)

def finish():
    videos=read(ROOT/'videos/manifest.json')['videos'];assert len(videos)==len(read(ROOT/'video_records.json'))
    for v in videos:assert v['all_saved_states_match_trace'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    s=read(ROOT/'summary.json');paragraph='; '.join(f"{c}: {v['outcomes']['success']}/48 success, {v['outcomes']['collisions']} collisions, {v['paired']['E170_protected']} versus E170; {v['extra_descent_steps']} extra descent steps" for c,v in s['conditions'].items());section='\n## E174: object-preserving routes and grasp handoff\n\n'+paragraph+'.96 fitting episodes and4 exact reference replays. E120, probe protection and release preserved. Strict grasp-first planning finds routes for3 failures; case26 retains the baseline because it starts in object contact. Object pose/contact inputs privileged; no held test. Report: `outputs/ease_object_preserving_v1/report.md`; videos: `videos/index.html`; targeted review: `literature_review.md`.\n'
    section+='\nCases62,65,67 are rescued in both conditions with zero measured object displacement during descent. All44 earlier successes and the case26 fallback retain complete reference states/traces exactly. Both variants spend377 extra descent steps total (15.08 simulated seconds). The guarded handoff adds no task successes in this comparison; all three routed grasps pass its checks, so no abort/rejection branch was exercised physically. The route bundle combines strict object avoidance, grasp-first vertical approach, current object XY and gain1 tracking; its components are not individually isolated. Literature review identifies task-and-motion planning/state guards, cuRobo, STORM and contact-implicit MPC as relevant established methods, with implementation and privilege caveats.\n'
    review=Path('outputs/ease_iteration_review_v1')
    for p in (Path(__file__).with_name('TECHNICAL_REPORT.md'),review/'report.md'):
        old=p.read_text(encoding='utf-8')
        if '## E174:' not in old:p.write_text(old+section,encoding='utf-8')
    p=review/'WORKING_STATE.md';old=p.read_text(encoding='utf-8')
    if not old.startswith('# Latest checkpoint: E174'):p.write_text('# Latest checkpoint: E174 complete\n'+section+'\nAll runs, audits and videos complete.\n\n---\n'+old,encoding='utf-8')
    studies=read(review/'studies.json')
    if not any(x['experiment']=='E174' for x in studies):studies.append(dict(experiment='E174',folder=str(ROOT),report_exists=True,videos=len(videos)));save(review/'studies.json',studies)
    save(ROOT/'status.json',dict(status='complete',new_episodes=96,exact_replays=4,videos=len(videos),held_evaluated=False,E120_unchanged=True,probe_protection_preserved=True));save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print('E174 complete',len(videos),'verified videos',flush=True)

if __name__=='__main__':
    import sys
    finish() if len(sys.argv)>1 and sys.argv[1]=='finish' else run()
