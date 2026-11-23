"""Independent uncached replay and new-state feature smoke check for E138."""
from pathlib import Path
import numpy as np
import mujoco
from .expanded_structured_pairs import ROOT,COLLECT,setup
from . import matched_correction as paired
from . import broad_task_value_gate as broad
from .abstaining_task_value_gate import read,save
from .arm_pick_place import rotation_error
from .compact_response_value import features
from .structured_response import preaction

def run():
    out=ROOT/'uncached_audit';out.mkdir(exist_ok=False)
    for folder in ('trials','states','actions','branches','snapshots'):(out/folder).mkdir()
    tasks,models,hashes=setup();paired.ROOT=out;broad.previous.ROOT=out;broad.previous.core.ROOT=out
    r,_=paired.evaluate(56,tasks[56],0,models['reference_0'],models['correction_0'],None)
    stem='56_on_0_frozen_bias';assert r==read(COLLECT/'trials'/f'{stem}.json');np.testing.assert_array_equal(np.load(out/'states'/f'{stem}.npz')['states'],np.load(COLLECT/'states'/f'{stem}.npz')['states'])
    actions=read(COLLECT/'actions'/f'{stem}.json');b=read(COLLECT/'branches'/'56_on_0.json');s=np.load(COLLECT/'snapshots'/'56_on_0_descend.npz');i=b['entries']['descend']['step'];a=actions[i];n=len(a['q_before']);m=mujoco.MjModel.from_xml_path(str((COLLECT/'states'/f'{stem}.xml').resolve()));d=mujoco.MjData(m);mujoco.mj_setState(m,d,s['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center')
    np.testing.assert_array_equal(d.qpos[:n],a['q_before']);np.testing.assert_allclose(d.site_xpos[site],a['position_before'],atol=1e-12,rtol=0)
    base=broad.features(n,s['gram'],s['cross'],s['bias'],a['q_before'],a['position_before'],s['goal'],'descend',b['entries']['descend']['metadata']['local'],a['base_action'],a['correction_action']);x=np.r_[base,d.site_xpos[site],rotation_error(np.eye(3),d.site_xmat[site].reshape(3,3))].astype(np.float32);f=features(x);assert f.shape==(32,) and np.isfinite(f).all();assert all(np.isfinite(v).all() for v in preaction(x.astype(np.float64)))
    save(ROOT/'smoke_verification.json',dict(uncached_case=56,seed=0,full_trial_exact=True,full_states_exact=True,new_snapshot_features_valid=True,extra_audit_rollouts=1));print('Uncached full rollout exactly matches cached collection; new snapshot features valid',flush=True)

if __name__=='__main__':run()
