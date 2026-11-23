"""Full saved-state audit for selective E172 recovery, including exact retention."""
from pathlib import Path

def run():
    s=Path(__file__).with_name('descent_progress_report.py').read_text().replace('descent_progress import','descent_stall import').replace('E171','E172').replace('new_interventions=96','new_interventions=48').replace('new_episodes=96','new_episodes=48').replace('96 intervention','48 intervention').replace('96 new episodes','48 new episodes')
    s=s.replace('post_count=0;gradient_count=0;objective_checks=0;', 'post_count=0;gradient_count=0;objective_checks=0;exact_retained=0;')
    s=s.replace('history=[];checked=False;cost_checked=False','history=[];checked=False;cost_checked=False;descent_errors=[];recovery=False')
    old="expected_direct=bool(l['phase']=='descend' and (c=='direct_task' or np.max(np.abs(l['q_before']))>=2.4))"
    new="""d.qpos[:]=l['full_qpos_before'];d.qvel[:]=l['full_qvel_before'];mujoco.mj_forward(m,d);site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center')
                if l['phase']=='descend':
                    pe=float(np.linalg.norm(np.array(l['goal'])-d.site_xpos[site]));descent_errors.append(pe)
                    if len(descent_errors)>=60 and .025<pe<.075 and descent_errors[-20]-pe<.001 and np.max(np.abs(l['q_before']))>=2.4:recovery=True
                expected_direct=bool(l['phase']=='descend' and recovery)
                assert l['descent_observations']==len(descent_errors)"""
    assert old in s;s=s.replace(old,new)
    old="target=j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*error)+l['bias'];np.testing.assert_allclose(target,l['optimization_target'],atol=1e-12,rtol=0);np.testing.assert_allclose(j.T@j+.015**2*np.eye(6),l['metric'],atol=1e-12,rtol=0)"
    new="posture=np.where(np.abs(l['q_before'])>2.35,-.02*np.sign(l['q_before']),0.);np.testing.assert_array_equal(posture,l['posture']);h=j.T@j+.03**2*np.eye(6);target=np.linalg.solve(h,j.T@(.35*error)+.03**2*posture)+l['bias'];np.testing.assert_allclose(target,l['optimization_target'],atol=1e-12,rtol=0);np.testing.assert_allclose(h,l['metric'],atol=1e-12,rtol=0)"
    assert old in s;s=s.replace(old,new)
    s=s.replace("            probe_changes+=r['changed_probe_steps']", """            if not any(l['direct_task'] for l in logs):
                assert t['trace']==ref['trace'] and t['success']==ref['success'];np.testing.assert_array_equal(states,np.load(REFERENCE/'states'/f'{stem}.npz')['states']);exact_retained+=1
            probe_changes+=r['changed_probe_steps']""")
    s=s.replace('checks[c]=dict(post_state_checks=', 'checks[c]=dict(exact_nonactivated_episodes=exact_retained,post_state_checks=')
    start=s.index("    lines=['# E172:");end=s.index("    (ROOT/'report.md').write_text",start)
    s=s[:start]+'''    lines=['# E172: selective late-stall descent recovery','','Privileged fitting development following rejected E171 variants. One fixed candidate evaluated on all48 six-joint fitting placements;4 exact E170 replays. E120, measured-error allowance, probe protection, release fix, phase budgets and scoring remain unchanged.','','| Controller | Success /48 | Collision episodes | Collision-free failures | Rescues / losses vs E170 |','|---|---:|---:|---:|---|']
    for k,v in summary['references'].items():lines.append(f"| {k} | {v['success']} | {v['collisions']} | {v['collision_free_failures']} | — |")
    for c,v in summary['conditions'].items():
        x=v['outcomes'];p=v['paired']['E170_protected'];lines.append(f"| {c} | {x['success']} | {x['collisions']} | {x['collision_free_failures']} | {p['rescues']} / {p['losses']} |")
    lines+=['','E171 direct task optimization and its joint-limit gate each completed43/48 and lost a previous success. They are rejected and their complete results/videos remain in ../ease_descent_progress_v1. E172 is an additional adaptive fitting experiment, not independent validation.','', 'Activation uses measured state only: at least60 descent observations,25–75mm position error, less than1mm progress over the previous19 control intervals, and any joint magnitude at least2.4rad. Recovery stays active for the remainder of descent. Earlier broader stall rules also activated on successful case43; their read-only diagnoses are retained. The local late-stall rule activates only on the four reference failures in the fitting diagnosis. No case identifiers enter the controller.','', 'Recovery minimizes the E120-estimated task residual with a posture regularizer: joints with magnitude above2.35rad prefer a0.02rad command toward the center; others prefer zero relative command. Regularization scale0.03 replaces0.015 only during recovery. The same QP bounds and privileged clearance constraints still apply. Outside recovery, the controller is exactly E170. No extra steps, new task Jacobian, oracle object state, goal changes, or model training are used.','', 'Audit independently reconstructs every activation decision, posture vector, objective center/metric, causal clearance allowance, accepted inequality and executed command. Every post-state geometry is checked, along with full-state gradient reconstruction and exact complete state/trace equality for nonactivated episodes. Four disabled-recovery reference replays passed first. Video replays verify saved tool/object traces and decoded frame counts;80ms sampling can miss contacts scored at2ms.','', 'A positive task-success screen requires more than44 successes, zero collisions and no lost E170 success. More local progress without a completed task is reported separately. E120+release remains the original baseline. All results are fitting-only and geometry-privileged; no safety guarantee or unseen-body claim is made.']
''' +s[end:]
    s=s.replace('descent_progress_videos.py','descent_stall_videos.py').replace('descent task-progress comparison','selective descent recovery comparison').replace('E172 privileged descent task progress','E172 privileged selective descent recovery')
    s=s.replace('PRIVILEGED direct task objective with E170 protection; matched E170 and E120+release references. Fitting cases only; all outcomes retained.','PRIVILEGED selective stall recovery with unchanged E170 protection; matched successes and failures on fitting cases.')
    s=s.replace('descent task-progress objective','selective descent stall recovery').replace('ease_descent_progress_v1/report.md','ease_descent_stall_v1/report.md')
    Path(__file__).with_name('descent_stall_report.py').write_text(s)

if __name__=='__main__':run()
