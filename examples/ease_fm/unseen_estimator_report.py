"""E176 paired outcomes, interaction budgets, histories and scientific artifacts."""
from pathlib import Path
from collections import deque
import shutil
import numpy as np
import mujoco
from .unseen_estimator import ROOT,REFERENCE,CONDITIONS,ROT_SCALE
from .abstaining_task_value_gate import read,save,fingerprint
from .arm_pick_place import PickPlace
from .mixed_sim import Arm,AXES
from .local_history import LocalPose
from .pick_place_control import calibrate

def outcome_counts(rows,root):
    return dict(n=len(rows),success=sum(r['success'] for r in rows),collisions=sum(r['collision'] for r in rows),grasped=sum(r['grasped'] for r in rows),lifted=sum(r['lifted'] for r in rows),collision_free_failures=sum(not r['success'] and not r['collision'] for r in rows))

def run():
    summary=read(ROOT/'summary.json');metrics={};case_rows={};selected=set();budgets=dict(approach=80,descend=100,probe=32,close=40,lift=100,transport=100,lower=100,release=40,retreat=80,settle=40);tasks=read(ROOT/'cases.json');calibration_histories={};AXES.update(read(ROOT/'axis_catalog.json'));seen_bodies=set()
    for i,t in enumerate(tasks):
        if t['arm']['seed'] not in seen_bodies:selected.add(i);seen_bodies.add(t['arm']['seed'])
    for method in CONDITIONS:
        root=ROOT/method;rows=read(root/'records.json');details=[];history_checks=0
        for r in rows:
            case=r['case'];stem=f'{case}_{method}_frozen_bias';t=read(root/'trials'/f'{stem}.json');logs=read(root/'clearance'/f'{stem}.json');states=np.load(root/'states'/f'{stem}.npz')['states'];other='analytical' if method=='e120_ensemble' else 'e120_ensemble';ref=read(ROOT/other/'clearance'/f'{case}_{other}_frozen_bias.json')
            np.testing.assert_array_equal(logs[0]['full_qpos_before'],ref[0]['full_qpos_before']);np.testing.assert_array_equal(logs[0]['full_qvel_before'],ref[0]['full_qvel_before']);assert len(t['calibration'])==32 and t['steps']==len(states)+32
            if case not in calibration_histories:
                task=tasks[case];env=PickPlace(Arm(**task['arm']),task['xy'],task['target']);assert env.xml==(root/'states'/f'{stem}.xml').read_text();controller=LocalPose(6,None);records=calibrate(env,task['home'],controller,38000+case);assert records==t['calibration']
                np.testing.assert_array_equal(env.data.qpos,logs[0]['full_qpos_before']);np.testing.assert_array_equal(env.data.qvel,logs[0]['full_qvel_before']);np.testing.assert_array_equal(controller.gram,t['initial_estimator']['gram']);np.testing.assert_array_equal(controller.cross,t['initial_estimator']['cross']);np.testing.assert_array_equal(controller.bias,t['initial_estimator']['bias']);calibration_histories[case]=list(controller.history)
            m=mujoco.MjModel.from_xml_path(str(root/'states'/f'{stem}.xml'));d=mujoco.MjData(m);site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');g=np.array(t['initial_estimator']['gram']);c=np.array(t['initial_estimator']['cross']);history=deque(((dq.copy(),dx.copy()) for dq,dx in calibration_histories[case]),maxlen=32);local=False
            def rebuild():
                gg=1e-6*np.eye(6);cc=np.zeros((6,6))
                for dq,dx in history:gg=.98*gg+np.outer(dq,dq)+2e-8*np.eye(6);cc=.98*cc+np.outer(dq,dx)
                return gg,cc
            for l,state in zip(logs,states):
                if l['phase']=='probe' and not local:
                    assert len(history)==32;local=True;g,c=rebuild()
                assert l['estimator_local']==local;np.testing.assert_allclose(g,l['estimator_gram'],atol=1e-12,rtol=0);np.testing.assert_allclose(c,l['estimator_cross'],atol=1e-12,rtol=0);np.testing.assert_array_equal(l['bias'],t['initial_estimator']['bias']);history_checks+=1
                d.qpos[:]=l['full_qpos_before'];d.qvel[:]=l['full_qvel_before'];mujoco.mj_forward(m,d);p=d.site_xpos[site].copy();axis=d.site_xmat[site].reshape(3,3)[:,2].copy()
                d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_forward(m,d);dq=d.qpos[:6]-l['q_before'];dx=np.r_[d.site_xpos[site]-p,ROT_SCALE*(d.site_xmat[site].reshape(3,3)[:,2]-axis)];history.append((dq.copy(),dx.copy()));g=.98*g+np.outer(dq,dq)+2e-8*np.eye(6);c=.98*c+np.outer(dq,dx)
                if local:g,c=rebuild()
            np.testing.assert_allclose(g,t['final_estimator']['gram'],atol=1e-12,rtol=0);np.testing.assert_allclose(c,t['final_estimator']['cross'],atol=1e-12,rtol=0)
            routed=any(l['route_active'] for l in logs);phase_steps={p['phase']:p['steps'] for p in t['phases']};assert set(phase_steps)==set(budgets)
            for phase,n in phase_steps.items():assert 0<n<=(300 if phase=='descend' and routed else budgets[phase])
            assert phase_steps['probe']==32 and len(states)+32<=(944 if routed else 744)
            trigger=next((l for l in logs if l['planning_attempted']),None);first_bad=next((dict(step=i,phase=x['phase'],contacts=x['bad']) for i,x in enumerate(t['trace']) if x['bad']),None)
            desc=[i for i,l in enumerate(logs) if l['phase']=='descend'];close=next(l for l in reversed(logs) if l['phase']=='close');objects=np.array([t['trace'][i]['object'][:2] for i in desc]);drift=float(np.max(np.linalg.norm(objects-objects[0],axis=1)))
            details.append(dict(case=case,body=r['body'],family=r['family'],success=t['success'],collision=t['collision'],grasped=t['grasped'],lifted=t['lifted'],control_steps=len(states)+32,phase_steps=phase_steps,trigger=None if trigger is None else trigger['trigger_reason'],trigger_descent_observation=None if trigger is None else trigger['descent_observations'],routed=routed,aborted=logs[-1]['aborted'],close_ready=close['close_ready'],final_two_pad_close_steps=close['close_good'],max_cube_descent_displacement_mm=1000*drift,first_forbidden_contact=first_bad,failed_phases=[p['phase'] for p in t['phases'] if not p['reached']]))
        by_family={}
        for family in sorted({r['family'] for r in details}):
            rr=[r for r in details if r['family']==family];by_family[family]=dict(n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr))
        steps=[r['control_steps'] for r in details];metrics[method]=dict(total_steps=sum(steps),median_steps=float(np.median(steps)),max_steps=max(steps),total_simulated_seconds=.04*sum(steps),route_cases=[r['case'] for r in details if r['routed']],early_trigger_cases=[r['case'] for r in details if r['trigger']=='early_object_tilt'],late_trigger_cases=[r['case'] for r in details if r['trigger']=='late_stall'],aborted_cases=[r['case'] for r in details if r['aborted']],by_family=by_family,reconstructed_history_steps=history_checks);case_rows[method]={r['case']:r for r in details}
    pairs=[]
    for case,a in case_rows['e120_ensemble'].items():
        b=case_rows['analytical'][case];pairs.append(dict(case=case,family=a['family'],e120_success=a['success'],analytical_success=b['success'],e120_collision=a['collision'],analytical_collision=b['collision'],e120_steps=a['control_steps'],analytical_steps=b['control_steps']))
        if a['success']!=b['success'] or a['collision']!=b['collision']:selected.add(case)
    for method in CONDITIONS:
        for success in (True,False):
            rr=[r for r in case_rows[method].values() if r['success']==success]
            if rr and not any(r['case'] in selected for r in rr):selected.add(rr[0]['case'])
    paired=dict(both_success=sum(p['e120_success'] and p['analytical_success'] for p in pairs),E120_only=[p['case'] for p in pairs if p['e120_success'] and not p['analytical_success']],analytical_only=[p['case'] for p in pairs if p['analytical_success'] and not p['e120_success']],both_fail=sum(not p['e120_success'] and not p['analytical_success'] for p in pairs))
    save(ROOT/'interaction_budget.json',metrics);save(ROOT/'case_diagnosis.json',case_rows);save(ROOT/'paired.json',dict(summary=paired,cases=pairs));save(ROOT/'history_verification.json',dict(exact_calibration_replays=len(calibration_histories),all_initial_states_and_statistics_verified=True,history_steps={m:metrics[m]['reconstructed_history_steps'] for m in CONDITIONS},probe_history_includes_remaining_calibration_transitions=True,conditional_budgets_verified=True));video_rows=[]
    for folder in ('trials','states'):(ROOT/folder).mkdir(exist_ok=True)
    sources=[(case,ROOT/m,m,'privileged_'+m) for case in sorted(selected) for m in CONDITIONS]
    for case,source,method,label in sources:
        old=f'{case}_{method}_frozen_bias';new=f'{case}_{label}_frozen_bias';t=read(source/'trials'/f'{old}.json');t['method']=label;save(ROOT/'trials'/f'{new}.json',t)
        for ext in ('npz','xml'):shutil.copy2(source/'states'/f'{old}.{ext}',ROOT/'states'/f'{new}.{ext}')
        video_rows.append(dict(case=case,method=label,mode='frozen_bias',success=t['success'],dof=6))
    save(ROOT/'video_records.json',video_rows)
    from .unseen_estimator_writeup import write_report
    write_report(summary,metrics,case_rows,paired,pairs)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.6))
    for ax,key,title in zip(axes,('success','collisions','collision_free_failures'),('Tasks completed','Collision episodes','Collision-free failures')):
        ns=[summary['conditions'][m]['outcomes'][key] for m in CONDITIONS];ax.bar(['E120','Analytical'],ns,color=['#29896c','#3b76ae']);ax.set_title(title);ax.set_ylim(0,52);ax.set_yticks([0,12,24,36,48]);ax.spines[['top','right']].set_visible(False)
        for i,n in enumerate(ns):ax.text(i,n+.8,str(n),ha='center')
    fig.suptitle('E177 | frozen controller | prospective unseen bodies');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    template=Path(__file__).with_name('link_clearance_videos.py').read_text().replace('link_clearance_diagnostic import ROOT','unseen_estimator import ROOT').replace('E168','E177').replace('link-clearance comparison','frozen-controller estimator comparison').replace('PRIVILEGED link/table constraints around E120 versus unchanged E120+release. Fitting cases only; successes, failures and stalls retained.','Prospective new six-joint bodies: E120 versus analytical, identical privileged controller and budget rules. Successes and failures retained; previously used axis families.')
    Path(__file__).with_name('unseen_estimator_videos.py').write_text(template);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'status.json',dict(status='audited_videos_pending',videos=len(video_rows)));print(paired,metrics,flush=True)

if __name__=='__main__':run()
