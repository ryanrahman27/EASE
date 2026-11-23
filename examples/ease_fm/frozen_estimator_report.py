"""E176 paired outcomes, interaction budgets, histories and scientific artifacts."""
from pathlib import Path
from collections import deque
import shutil
import numpy as np
import mujoco
from .frozen_estimator_comparison import ROOT,REFERENCE,CONDITIONS,ROT_SCALE
from .abstaining_task_value_gate import read,save,fingerprint
from .arm_pick_place import PickPlace
from .mixed_sim import Arm,AXES
from .local_history import LocalPose
from .pick_place_control import calibrate

def outcome_counts(rows,root):
    return dict(n=len(rows),success=sum(r['success'] for r in rows),collisions=sum(r['collision'] for r in rows),grasped=sum(r['grasped'] for r in rows),lifted=sum(r['lifted'] for r in rows),collision_free_failures=sum(not r['success'] and not r['collision'] for r in rows))

def run():
    summary=read(ROOT/'summary.json');metrics={};case_rows={};selected={0,26,43,62,65,67};budgets=dict(approach=80,descend=100,probe=32,close=40,lift=100,transport=100,lower=100,release=40,retreat=80,settle=40);tasks=read(ROOT/'cases.json');calibration_histories={};AXES.update(read('outputs/ease_unseen_progress_v1/axis_catalog.json'))
    for method in CONDITIONS:
        root=ROOT/method;rows=read(root/'records.json');details=[];history_checks=0
        for r in rows:
            case=r['case'];stem=f'{case}_{method}_frozen_bias';t=read(root/'trials'/f'{stem}.json');logs=read(root/'clearance'/f'{stem}.json');states=np.load(root/'states'/f'{stem}.npz')['states'];ref=read(REFERENCE/'clearance'/f'{case}_e120_ensemble_frozen_bias.json')
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
    sources.append((26,Path('outputs/ease_object_preserving_v1/guarded_handoff'),'e120_ensemble','privileged_E174_historical'))
    for case,source,method,label in sources:
        old=f'{case}_{method}_frozen_bias';new=f'{case}_{label}_frozen_bias';t=read(source/'trials'/f'{old}.json');t['method']=label;save(ROOT/'trials'/f'{new}.json',t)
        for ext in ('npz','xml'):shutil.copy2(source/'states'/f'{old}.{ext}',ROOT/'states'/f'{new}.{ext}')
        video_rows.append(dict(case=case,method=label,mode='frozen_bias',success=t['success'],dof=6))
    save(ROOT/'video_records.json',video_rows)
    lines=['# E176: E120 versus analytical estimation under the frozen E175 controller','','96 paired fitting episodes,48 per method;5 preflight reference replays. E120 also exactly reproduces all48 complete E175 state sequences and traces. No training, controller tuning or held evaluation.','','| Estimator | Success /48 | Collision episodes | Total control steps | Median steps |','|---|---:|---:|---:|---:|']
    for method in CONDITIONS:
        x=summary['conditions'][method]['outcomes'];v=metrics[method];lines.append(f"| {method} | {x['success']} | {x['collisions']} | {v['total_steps']} | {v['median_steps']:.1f} |")
    lines+=['',f'Paired outcomes: {paired}.','','## What is matched','','Both methods receive the same initial state and32-step calibration, frozen measured bias,32-step local probe, state measurements and privileged geometry/object/contact inputs. Recovery triggers, grasp-first planner, tracking, protections, guarded handoff,265mm release, phase targets, action bounds, early-exit rules and success scoring are unchanged from E175. The motion statistics update with forgetting.98; at the local probe both rebuild from the last32 measured transitions. The independent audit reconstructs these statistics from saved physical states for every executed step.','','E120 is the established three-model reference/correction ensemble with its bounded action residual. Analytical uses the original regularized solve(gram,cross).T, with no neural prediction or residual. This compares the existing estimator recipes; it does not isolate ensembling, training, or the bounded residual separately. The safety-QP metric uses each estimator prediction under the same formula. Online histories and recovery activation can diverge because actions differ.','','## Interaction budget','','Maximum total steps are944 with an accepted recovery route and744 otherwise, including32 calibration steps. Both receive the same conditional allowance: descent100, extended to300 only after a successful plan; all other phase caps match. Each performs one task attempt and32 probe steps. We do not force equal realized steps or pad completed phases. Actual steps above include calibration and all task phases. Planning wall-clock time and computation are not equalized or benchmarked.','','## Paired diagnostic details','','| Case | E120 success | Analytical success | E120 collision | Analytical collision | E120 / analytical steps |','|---|---|---|---|---|---|']
    for p in pairs:
        if p['case'] in selected:lines.append(f"| {p['case']} | {p['e120_success']} | {p['analytical_success']} | {p['e120_collision']} | {p['analytical_collision']} | {p['e120_steps']} / {p['analytical_steps']} |")
    lines+=['','## Failure and recovery details','','| Case / estimator | Trigger | Descent observation | Accepted route | Close ready | Final two-pad close steps | Cube descent displacement (mm) | Abort |','|---|---|---:|---|---|---:|---:|---|']
    for case in sorted(selected):
        for method in CONDITIONS:
            r=case_rows[method][case];lines.append(f"| {case} / {method} | {r['trigger']} | {r['trigger_descent_observation']} | {r['routed']} | {r['close_ready']} | {r['final_two_pad_close_steps']} | {r['max_cube_descent_displacement_mm']:.3f} | {r['aborted']} |")
    lines+=['','Close-ready is only evaluated after an accepted route; false on an ordinary successful episode does not imply a rejected grasp. These are trajectory-level observations, not matched-history causal estimates.','','## Scope and interpretation','','These are48 six-joint fitting placements across two training families (24 each), used during controller development, not48 untouched embodiments or independent random seeds. E175 was developed around E120, so a difference here may reflect compatibility with that frozen controller. A tie establishes no learned task-success advantage. Any gain is a descriptive fitting result under shared privilege and must be tested prospectively. Full state/action/history traces, failures and independent audits are retained. Historical E174 case26 failure video is context only and excluded from E176 metrics.','','The inherited controller applies established grasp-first task-and-motion planning and reactive grasp guards; E176 adds no new literature method or control feature. See the previous experiment’s literature_application.md.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(11,3.6))
    for ax,key,title in zip(axes,('success','collisions','collision_free_failures'),('Tasks completed','Collision episodes','Collision-free failures')):
        ns=[summary['conditions'][m]['outcomes'][key] for m in CONDITIONS];ax.bar(['E120','Analytical'],ns,color=['#29896c','#3b76ae']);ax.set_title(title);ax.set_ylim(0,52);ax.set_yticks([0,12,24,36,48]);ax.spines[['top','right']].set_visible(False)
        for i,n in enumerate(ns):ax.text(i,n+.8,str(n),ha='center')
    fig.suptitle('E176 | frozen controller, matched budgets | privileged fitting cases');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    template=Path(__file__).with_name('link_clearance_videos.py').read_text().replace('link_clearance_diagnostic import ROOT','frozen_estimator_comparison import ROOT').replace('E168','E176').replace('link-clearance comparison','frozen-controller estimator comparison').replace('PRIVILEGED link/table constraints around E120 versus unchanged E120+release. Fitting cases only; successes, failures and stalls retained.','E120 versus analytical estimation with the same privileged E175 controller and budget caps. Fitting cases only. Historical E174 failure is labeled separately.')
    Path(__file__).with_name('frozen_estimator_videos.py').write_text(template);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'status.json',dict(status='audited_videos_pending',videos=len(video_rows)));print(paired,metrics,flush=True)

def finish():
    videos=read(ROOT/'videos/manifest.json')['videos'];assert len(videos)==len(read(ROOT/'video_records.json'))
    for v in videos:assert v['all_saved_states_match_trace'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    verification=read(ROOT/'verification.json');verification['conditions']['analytical']['matched_preintervention_prefix']=False;save(ROOT/'verification.json',verification)
    save(ROOT/'audit_notes.json',dict(plan_equality='All deterministic plan contents compared; only recorded planning_seconds omitted because wall-clock duration is not deterministic.',prefix_equality='Only E120 is asserted equal to E175. Analytical has identical calibration/initial state but different subsequent histories; metadata label corrected to reflect the actual assertions.',controller_changed_after_outcomes=False))
    costs=read(ROOT/'interaction_budget.json');p=ROOT/'report.md';text=p.read_text(encoding='utf-8')
    if '## Family breakdown' not in text:
        lines=['','## Family breakdown','','| Training family | E120 successes | Analytical successes | E120 / analytical collisions |','|---|---:|---:|---|']
        for family,a in costs['e120_ensemble']['by_family'].items():
            b=costs['analytical']['by_family'][family];lines.append(f"| {family} | {a['success']}/{a['n']} | {b['success']}/{b['n']} | {a['collisions']} / {b['collisions']} |")
        lines+=['','E120 uses fewer total control steps in this batch, while analytical has the lower median. Totals include the long aborted/failing trajectories. This supports a fitting-case reliability difference, not a general runtime or sample-efficiency claim. Planner wall-clock timings are excluded from deterministic plan equality. All path values, executed states and E120 traces remain checked exactly.']
        p.write_text(text+'\n'.join(lines)+'\n',encoding='utf-8')
    for name in ('frozen_estimator_audit.py','build_frozen_estimator_audit.py','frozen_estimator_report.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    s=read(ROOT/'summary.json')['conditions'];pair=read(ROOT/'paired.json')['summary'];paragraph=f"E120 {s['e120_ensemble']['outcomes']['success']}/48, analytical {s['analytical']['outcomes']['success']}/48; collisions {s['e120_ensemble']['outcomes']['collisions']} versus {s['analytical']['outcomes']['collisions']}. Paired outcomes: {pair}.96 fitting episodes and5 preflight replays; all48 E120 trajectories exactly reproduce E175. Same frozen controller, protections, probes and conditional step caps; actual interaction costs saved. Privileged fitting comparison, no held evaluation or training."
    section='\n## E176: frozen-controller estimator comparison\n\n'+paragraph+'\n\nReport: `outputs/ease_frozen_estimator_comparison_v1/report.md`; success/failure videos: `videos/index.html`.\n';review=Path('outputs/ease_iteration_review_v1')
    for p in (Path(__file__).with_name('TECHNICAL_REPORT.md'),review/'report.md'):
        t=p.read_text(encoding='utf-8')
        if '## E176:' not in t:p.write_text(t+section,encoding='utf-8')
    p=review/'WORKING_STATE.md';t=p.read_text(encoding='utf-8')
    if not t.startswith('# Latest checkpoint: E176'):p.write_text('# Latest checkpoint: E176 complete\n'+section+'\nAll runs, audits and videos complete.\n\n---\n'+t,encoding='utf-8')
    studies=read(review/'studies.json')
    if not any(x['experiment']=='E176' for x in studies):studies.append(dict(experiment='E176',folder=str(ROOT),report_exists=True,videos=len(videos)));save(review/'studies.json',studies)
    save(ROOT/'status.json',dict(status='complete',new_episodes=96,preflight_exact_replays=5,full_E175_exact_replays=48,videos=len(videos),held_evaluated=False,controller_frozen=True));save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print(paragraph,flush=True)

if __name__=='__main__':
    import sys
    finish() if len(sys.argv)>1 and sys.argv[1]=='finish' else run()
