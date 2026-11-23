"""Pre-intervention numerical checks and archived nominal-bound correction."""
import shutil
from pathlib import Path
import numpy as np
import mujoco
from .link_clearance_diagnostic import ROOT,BASE,project,geometry_ids,distance_gradient,distances
from .abstaining_task_value_gate import read,save,fingerprint

def run():
    assert not any((ROOT/c/'records.json').exists() for c in ('joint_projection','task_projection'))
    source=ROOT/'source/link_clearance_diagnostic.py';current=Path(__file__).with_name('link_clearance_diagnostic.py');before=source.read_text();after=current.read_text();assert before.replace('if np.min(nominal_slack)>=0:return','if np.min(nominal_slack)>=0 and np.all(action>=lower) and np.all(action<=upper):return')==after
    archive=ROOT/'pre_intervention_review';archive.mkdir();shutil.copy2(source,archive/'initial_source.py');shutil.copy2(ROOT/'frozen.json',archive/'initial_frozen.json')
    save(archive/'change.json',dict(reason='Pre-intervention code review: skip optimization only when nominal action satisfies both clearance and actual joint-target bounds. No intervention outcomes existed; disabled-layer exact baseline replays already passed.',old_source=fingerprint(source),new_source=fingerprint(current)))
    shutil.copy2(current,source);frozen=read(ROOT/'frozen.json');frozen[str(source)]=fingerprint(source);save(ROOT/'frozen.json',frozen)
    eye=np.eye(2);zero=np.zeros(2)
    u,info,_=project(zero,zero,zero,np.array([-.017]),np.array([[1.,0.]]),eye);np.testing.assert_allclose(u,[.01,0],atol=1e-8);assert info['solver_success']
    u,info,_=project(np.array([.04,0]),zero,np.array([2.49,0]),np.array([.05]),np.zeros((1,2)),eye);np.testing.assert_allclose(u,[.01,0],atol=1e-8);assert info['solver_success']
    u,info,_=project(zero,zero,zero,np.array([-.001]),np.zeros((1,2)),eye);assert info['fallback'] and not info['solver_success'];np.testing.assert_array_equal(u,zero)
    stem='13_e120_ensemble_frozen_bias';m=mujoco.MjModel.from_xml_path(str(BASE/'states'/f'{stem}.xml'));d=mujoco.MjData(m);scratch=mujoco.MjData(m);s=np.load(BASE/'states'/f'{stem}.npz')['states'][0];d.qpos[:]=s[:m.nq];d.qvel[:]=s[m.nq:];mujoco.mj_forward(m,d);links,table=geometry_ids(m,6);q=d.qpos.copy();vel=d.qvel.copy();g=distance_gradient(m,d,scratch,links,table,6);np.testing.assert_array_equal(d.qpos,q);np.testing.assert_array_equal(d.qvel,vel)
    direction=np.array([.3,-.2,.1,.4,-.5,.2]);eps=1e-6;scratch.qpos[:]=q;scratch.qpos[:6]+=eps*direction;mujoco.mj_forward(m,scratch);plus=distances(m,scratch,links,table);scratch.qpos[:]=q;scratch.qpos[:6]-=eps*direction;mujoco.mj_forward(m,scratch);minus=distances(m,scratch,links,table);error=float(np.max(np.abs((plus-minus)/(2*eps)-g@direction)));assert error<1e-5
    save(ROOT/'preflight_checks.json',dict(before_intervention_outcomes=True,known_QP_solution=True,joint_target_bound_enforced=True,infeasible_case_explicit_hold=True,geometry_directional_derivative_max_error=error,scratch_preserves_live_joint_state=True,exact_disabled_baseline_replays=4));print(read(ROOT/'preflight_checks.json'),flush=True)

if __name__=='__main__':run()
