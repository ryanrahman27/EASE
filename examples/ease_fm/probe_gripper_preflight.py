"""Before-outcome checks for added gripper geometries and arm-joint gradients."""
import numpy as np
import mujoco
from .probe_gripper_clearance import ROOT,REFERENCE,geometry_ids,prior,distances,distance_gradient
from .abstaining_task_value_gate import save

def run():
    checks=[]
    for case,step in ((0,0),(26,100)):
        stem=f'{case}_e120_ensemble_frozen_bias';m=mujoco.MjModel.from_xml_path(str(REFERENCE/'states'/f'{stem}.xml'));states=np.load(REFERENCE/'states'/f'{stem}.npz')['states'];s=states[min(step,len(states)-1)];d=mujoco.MjData(m);scratch=mujoco.MjData(m);d.qpos[:]=s[:m.nq];d.qvel[:]=s[m.nq:];mujoco.mj_forward(m,d);ids,table=geometry_ids(m,6,True);assert len(ids)==9;q=d.qpos.copy();v=d.qvel.copy();g=distance_gradient(m,d,scratch,ids,table,6);np.testing.assert_array_equal(d.qpos,q);np.testing.assert_array_equal(d.qvel,v);direction=np.array([.3,-.2,.1,.4,-.5,.2]);eps=1e-6;scratch.qpos[:]=q;scratch.qpos[:6]+=eps*direction;mujoco.mj_forward(m,scratch);plus=distances(m,scratch,ids,table);scratch.qpos[:]=q;scratch.qpos[:6]-=eps*direction;mujoco.mj_forward(m,scratch);minus=distances(m,scratch,ids,table);error=np.abs((plus-minus)/(2*eps)-g@direction);print(dict(case=case,step=step,distances=distances(m,d,ids,table).tolist(),error=error.tolist(),directional=((plus-minus)/(2*eps)).tolist(),linear=(g@direction).tolist()),flush=True);assert np.max(error)<1e-5
        checks.append(dict(case=case,step=step,geometry_ids=ids,gripper_names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) for i in ids[6:]],directional_derivative_max_error=float(np.max(error)),gripper_derivative_max_error=float(np.max(error[6:])),live_state_unchanged=True))
    assert not any((ROOT/name/'records.json').exists() for name in ('probe_only','gripper_only','both'));save(ROOT/'preflight_checks.json',dict(before_intervention_outcomes=True,checks=checks));print(checks,flush=True)

if __name__=='__main__':run()
