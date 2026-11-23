"""E175 independent route, causal-controller, physical-outcome and video audit."""
from pathlib import Path
import shutil
import numpy as np
import mujoco
from .unseen_estimator import ROOT,BASE,REFERENCE,CONDITIONS,prior,geometry_ids,distances,distance_gradient
from .object_preserving_plan import forbidden
from .early_object_plan import early_gate,observation
from .unseen_estimator_report import outcome_counts
from .arm_pick_place import ROT_SCALE,rotation_error
from .abstaining_task_value_gate import read,save,fingerprint

def comparable(value):
    if isinstance(value,dict):return {k:comparable(v) for k,v in value.items() if k!='planning_seconds'}
    if isinstance(value,list):return [comparable(v) for v in value]
    return value

def run():
    refs={};summary={'references':{},'conditions':{}};checks={}
    for condition in CONDITIONS:
        root=ROOT/condition;rows=read(root/'records.json');assert len(rows)==len(read(ROOT/'cases.json')) and read(root/'verification.json')['episodes']==len(rows)
        paired={}
        for label,(_,rs) in refs.items():
            lookup={r['case']:r for r in rs};assert set(lookup)=={r['case'] for r in rows};paired[label]=dict(rescues=[r['case'] for r in rows if r['success'] and not lookup[r['case']]['success']],losses=[r['case'] for r in rows if not r['success'] and lookup[r['case']]['success']])
        post_checks=0;gradient_checks=0;path_checks=0;retained=0;fallbacks=0;route_cases=[];extra_steps=0
        for r in rows:
            case=r['case'];stem=f'{case}_{condition}_frozen_bias';reference_stem=f'{case}_e120_ensemble_frozen_bias';t=read(root/'trials'/f'{stem}.json');other='analytical' if condition=='e120_ensemble' else 'e120_ensemble';ref=read(ROOT/other/'trials'/f'{case}_{other}_frozen_bias.json');logs=read(root/'clearance'/f'{stem}.json');states=np.load(root/'states'/f'{stem}.npz')['states'];assert len(logs)==len(states)==len(t['trace'])
            assert t['calibration']==ref['calibration'] and t['initial_estimator']==ref['initial_estimator'];assert t['initial_estimator']['bias']==t['final_estimator']['bias']
            m=mujoco.MjModel.from_xml_path(str(root/'states'/f'{stem}.xml'));d=mujoco.MjData(m);scratch=mujoco.MjData(m);ids,table=geometry_ids(m,6,True);site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');object_history=[];trigger_reason=None;es=[];errors=[];attempted=False;route=None;index=0;checked=False;trigger_step=None;pick=None;last_phase=None;ready=False;good=0;aborted=False;close_anchor=None;abort_anchor=None
            for i,(l,state) in enumerate(zip(logs,states)):
                d.qpos[:]=l['full_qpos_before'];d.qvel[:]=l['full_qvel_before'];mujoco.mj_forward(m,d)
                if condition=='analytical':
                    analytical=np.linalg.solve(np.array(l['estimator_gram']),np.array(l['estimator_cross'])).T
                    np.testing.assert_allclose(analytical,l['estimator_jacobian'],atol=1e-14,rtol=0)
                    if l['phase']!='probe':
                        error=np.r_[np.array(l['goal'])-d.site_xpos[site],ROT_SCALE*rotation_error(np.eye(3),d.site_xmat[site].reshape(3,3))]
                        nominal=np.clip(analytical.T@np.linalg.solve(analytical@analytical.T+.015**2*np.eye(6),.35*error)+np.array(l['bias']),-.04,.04)
                        np.testing.assert_allclose(nominal,l['nominal_action'],atol=1e-12,rtol=0)
                if l['eligible']:np.testing.assert_array_equal(l['estimator_jacobian'],l['estimated_jacobian'])
                if l['phase']=='descend':
                    pe=float(np.linalg.norm(np.array(l['goal'])-d.site_xpos[site]));es.append(pe)
                    object_history.append(observation(m,d,l['goal']));assert l['object_guard_observation']==object_history[-1]
                    early=early_gate(object_history);late=len(es)>=60 and .025<pe<.075 and es[-20]-pe<.001 and np.max(np.abs(l['q_before']))>=2.4
                    if not attempted and (early or late):
                        trigger_reason='early_object_tilt' if early else 'late_stall'
                        attempted=True;trigger_step=i;plan=read(root/'planning'/f'{case}.json');np.testing.assert_array_equal(plan['start_qpos'],l['full_qpos_before']);np.testing.assert_array_equal(plan['start_qvel'],l['full_qvel_before']);object_id=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'object');np.testing.assert_array_equal(plan['pick_goal'],np.r_[d.xpos[object_id,:2],.24])
                        if plan['available']:
                            route=np.array(plan['dense_path']);pick=np.array(plan['pick_goal']);np.testing.assert_array_equal(route[0],l['q_before'])
                            assert np.max(np.abs(np.diff(route,axis=0)))<=.025+1e-12
                            for q in route:
                                scratch.qpos[:]=plan['start_qpos'];scratch.qvel[:]=plan['start_qvel'];scratch.qpos[:6]=q;mujoco.mj_forward(m,scratch);assert np.min(distances(m,scratch,ids,table))>=0 and not forbidden(m,scratch);path_checks+=1
                            assert np.linalg.norm(scratch.site_xpos[site]-pick)<.003;assert np.linalg.norm(scratch.site_xmat[site].reshape(3,3)[:,2]-[0,0,1])<.04;assert np.max(np.abs(route[-1]))<=2.35+1e-12
                assert l['trigger_reason']==trigger_reason
                active=bool(l['phase']=='descend' and route is not None);assert l['route_active']==active and l['planning_attempted']==attempted and l['descent_observations']==len(es)
                if active:
                    index+=int(np.argmin(np.linalg.norm(route[index:]-l['q_before'],axis=1)));waypoint=route[min(index+3,len(route)-1)];np.testing.assert_array_equal(waypoint,l['waypoint']);target=np.clip(waypoint-l['q_before']+np.array(l['bias']),-.04,.04)
                else:target=np.array(l['nominal_action']);assert l['waypoint'] is None
                handoff=False;aperture=l['requested_aperture']
                if route is not None:
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
            desc=next(p for p in t['phases'] if p['phase']=='descend');assert desc['steps']==len(es) and len(es)<=(300 if route is not None else 100);extra_steps+=max(0,len(es)-100)
            if attempted:
                last=next(x for x in reversed(t['trace']) if x['phase']=='descend');route_cases.append(dict(case=case,trigger_reason=trigger_reason,close_ready=ready,verified_final_close_steps=good,aborted=aborted,route_available=route is not None,success=r['success'],collision=r['collision'],trigger_step=trigger_step,descent_steps=len(es),extra_descent_steps=max(0,len(es)-100),last_route_index=index,path_nodes=0 if route is None else len(route),terminal_descent_error_mm=1000*last['position_error'],terminal_rotation_error=last['rotation_error'],failed_phases=[p['phase'] for p in t['phases'] if not p['reached']]))
        counts=outcome_counts(rows,root);summary['conditions'][condition]=dict(outcomes=counts,paired=paired,route_cases=route_cases,extra_descent_steps=extra_steps,extra_simulated_seconds=.04*extra_steps,fallbacks=fallbacks);checks[condition]=dict(exact_nonactivated_episodes=retained,post_state_geometry_checks=post_checks,gradient_checks=gradient_checks,independent_static_path_points=path_checks,causal_gate_and_allowance=True,static_plan_feasibility_verified=True,waypoints_and_actual_commands_verified=True,accepted_constraints_verified=True,budgets_verified=True,grasp_guard_and_actual_end_contacts_verified=True,paired_initial_statistics_verified=True)
    for p,h in {**read(ROOT/'frozen.json'),**read(ROOT/'protocol.json')['checkpoints']}.items():assert fingerprint(p)==h
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name)),p.name
    for method in CONDITIONS:assert read(ROOT/f'replay_{method}/verification.json')['exact_replays']==5
    save(ROOT/'summary.json',summary);save(ROOT/'verification.json',dict(conditions=checks,new_episodes=2*len(read(ROOT/'cases.json')),preflight_exact_replays=10,privileged=True,prospective_unseen_bodies=True,reserved_legacy_tests_evaluated=False))
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(summary['conditions'],flush=True)

if __name__=='__main__':run()
