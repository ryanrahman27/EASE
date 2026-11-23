"""Archive the failed native box-distance preflight before any new outcomes."""
from pathlib import Path
import shutil
import numpy as np
import mujoco
from .probe_gripper_clearance import ROOT,REFERENCE,geometry_ids,prior,distances
from .abstaining_task_value_gate import read,save,fingerprint

def run():
    assert all(not (ROOT/c/'records.json').exists() for c in ('probe_only','gripper_only','both'));assert read(ROOT/'replay/verification.json')['exact_replays']==4
    archive=ROOT/'preflight_archive';archive.mkdir(exist_ok=False)
    for name in ('protocol.json','frozen.json'):shutil.copy2(ROOT/name,archive/name)
    shutil.copy2(ROOT/'source/probe_gripper_clearance.py',archive/'initial_controller.py');shutil.copytree(ROOT/'replay',archive/'initial_four_exact_replays')
    m=mujoco.MjModel.from_xml_path(str(REFERENCE/'states/0_e120_ensemble_frozen_bias.xml'));s=np.load(REFERENCE/'states/0_e120_ensemble_frozen_bias.npz')['states'][0];d=mujoco.MjData(m);scratch=mujoco.MjData(m);d.qpos[:]=s[:m.nq];d.qvel[:]=s[m.nq:];mujoco.mj_forward(m,d);ids,table=geometry_ids(m,6,True);g=prior.distance_gradient(m,d,scratch,ids,table,6);direction=np.array([.3,-.2,.1,.4,-.5,.2]);eps=1e-6;scratch.qpos[:]=d.qpos;scratch.qpos[:6]+=eps*direction;mujoco.mj_forward(m,scratch);plus=prior.distances(m,scratch,ids,table);scratch.qpos[:]=d.qpos;scratch.qpos[:6]-=eps*direction;mujoco.mj_forward(m,scratch);minus=prior.distances(m,scratch,ids,table)
    save(archive/'native_distance_failure.json',dict(case=0,step=0,native_distances=prior.distances(m,d,ids,table).tolist(),analytic_plane_distances=distances(m,d,ids,table).tolist(),native_gradient_direction_error=np.abs((plus-minus)/(2*eps)-g@direction).tolist()))
    protocol=read(ROOT/'protocol.json');protocol['gripper']='Privileged conservative infinite-tabletop-plane clearance: minimum world-Z vertex of each palm/pad box minus table top. Exact current box transforms/extents, arm-joint finite differences. This is sufficient vertical clearance, not shortest distance to a bounded table; it can restrict motion outside the tabletop footprint. Jaw commands remain unchanged and gradients hold jaw positions fixed, so jaw motion/dynamics may violate predicted clearance. Object contact scoring unchanged; ground/self-contact not constrained.';protocol['preflight_amendment']='Before any intervention outcome, native box/table distance queries failed directional-derivative checks (including a zero distance at a clearly separated palm). Replaced only new gripper rows with analytic tabletop-plane clearances. Original protocol/source/frozen manifest and four initial exact replays archived. Arm-capsule queries, solver, margin, objectives, coverage flags and task scoring unchanged. Corrected gradients passed; four exact replays repeated before intervention.';save(ROOT/'protocol.json',protocol)
    for name in ('probe_gripper_clearance.py','probe_gripper_preflight.py','probe_gripper_geometry_amendment.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py'),Path(prior.__file__)]});save(ROOT/'replay/records.json',[]);save(ROOT/'replay/verification.json',dict(exact_replays=0,corrected_replay_pending=True));print('Pre-outcome geometry amendment archived; corrected replay required',flush=True)

if __name__=='__main__':run()
