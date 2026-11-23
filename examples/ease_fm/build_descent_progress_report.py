"""Adapt the full E170 audit while adding task-QP verification and E171 reporting."""
from pathlib import Path

def run():
    s=Path(__file__).with_name('descent_execution_report.py').read_text()
    s=s.replace('descent_execution import','descent_progress import').replace('E170','E171').replace('E169_protected','E170_protected').replace('privileged_E169','privileged_E170').replace("'E169'","'E170'")
    s=s.replace("counts['success']>41 and counts['collisions']<4","counts['success']>44 and counts['collisions']==0")
    s=s.replace("shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)","shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)")
    anchor="                if l['eligible']:\n                    g="
    insertion="""                expected_direct=bool(l['phase']=='descend' and (c=='direct_task' or np.max(np.abs(l['q_before']))>=2.4))
                assert l['direct_task']==expected_direct
                if expected_direct:
                    d.qpos[:]=l['full_qpos_before'];d.qvel[:]=l['full_qvel_before'];mujoco.mj_forward(m,d)
                    site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');j=np.array(l['estimated_jacobian']);error=np.r_[np.array(l['goal'])-d.site_xpos[site],.15*(np.array([0.,0.,1.])-d.site_xmat[site].reshape(3,3)[:,2])]
                    target=j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*error)+l['bias'];np.testing.assert_allclose(target,l['optimization_target'],atol=1e-12,rtol=0);np.testing.assert_allclose(j.T@j+.015**2*np.eye(6),l['metric'],atol=1e-12,rtol=0)
                else:np.testing.assert_array_equal(l['optimization_target'],l['nominal_action'])
                if l['eligible']:
                    g="""
    assert anchor in s;s=s.replace(anchor,insertion)
    s=s.replace('causal_allowances_verified=True,','direct_task_targets_and_gate_verified=True,causal_allowances_verified=True,')
    start=s.index("    lines=['# E171:");end=s.index("    (ROOT/'report.md').write_text",start)
    s=s[:start]+'''    lines=['# E171: descent progress under active constraints','','Privileged fitting-only control diagnostic. Frozen E120, measured-error descent allowance, probe protection,265mm release fix, original phase budgets and success criteria are retained.96 intervention episodes and4 exact E170 reference replays completed. No held evaluation or retraining.','','| Controller | Success /48 | Collisions | Collision-free failures | Rescues / losses vs E170 |','|---|---:|---:|---:|---|']
    for k,v in summary['references'].items():lines.append(f"| {k} | {v['success']} | {v['collisions']} | {v['collision_free_failures']} | — |")
    for c,v in summary['conditions'].items():
        x=v['outcomes'];p=v['paired']['E170_protected'];lines.append(f"| {c} | {x['success']} | {x['collisions']} | {x['collision_free_failures']} | {p['rescues']} / {p['losses']} |")
    lines+=['','Diagnosis: all four remaining reference failures spend100 descent steps and end with joint5 at its upper bound, active clearance constraints, and little or negative late progress. Case26 also has large orientation error. Full48-case saved-state diagnosis is retained in diagnosis.json.','', 'The original controller clips each damped least-squares joint command before projecting it into constraints. The new objective minimizes the E120-estimated task residual directly under the same command, joint-target and clearance constraints. Its quadratic center is the unclipped damped least-squares action; its metric remains J.T J +0.015 squared I. direct_task applies during every descent step; limit_gated applies only when any measured absolute joint coordinate is at least2.4rad. These settings were frozen before outcomes. No added steps, oracle task Jacobian, goal change, retraining or relaxation of safety/scoring is included.','', 'The E170 causal measured-error allowance and all probe/palm/link protections remain in both variants. E120 weights and calibrated bias are unchanged. Full simulator geometry is still privileged, and the local model can be inaccurate. This is not evidence of learned-only adaptation or independent transfer.','', 'Verification covers four exact reference replays including stalled cases, every saved geometry state and executed command, full-state finite-difference gradients, causal error allowances, accepted constraint inequalities, target/gate reconstruction, frozen calibration and identical pre-descent prefixes. The quadratic task-objective equivalence was checked before interventions. All failures, collisions and solver fallbacks are retained. Videos replay actual saved states with trace/frame-count checks;80ms video sampling can miss contacts scored at2ms.','', 'A positive screen requires more than44 successes, no collisions and no loss of any E170 success. Outcomes are reported against both E170 and original E120+release. No candidate automatically replaces the baseline.']
''' +s[end:]
    s=s.replace('descent_execution_videos.py','descent_progress_videos.py').replace('descent execution-error comparison','descent task-progress comparison').replace('PRIVILEGED descent error allowances with probe protection; matched E169 and E120+release references. Fitting cases only; successes and failures retained.','PRIVILEGED direct task objective with E170 protection; matched E170 and E120+release references. Fitting cases only; all outcomes retained.')
    s=s.replace('E171 privileged descent execution allowances','E171 privileged descent task progress')
    s=s.replace('descent execution-error allowances','descent task-progress objective').replace('versus E169','versus E170').replace('E169 protected reference41/48 with4 collisions','E170 protected reference44/48 with0 collisions').replace('ease_descent_execution_v1/report.md','ease_descent_progress_v1/report.md')
    Path(__file__).with_name('descent_progress_report.py').write_text(s)

if __name__=='__main__':run()
