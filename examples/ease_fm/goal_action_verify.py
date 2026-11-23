"""Reconstruct original/selected branches from saved integration snapshots."""
import json
import shutil
from pathlib import Path
import numpy as np
import mujoco
from .goal_action_audit import ROOT
from .physics_prediction_audit import forbidden

def run():
    rows=json.loads((ROOT/'records.json').read_text());checks=[]
    for row in rows:
        stem=f"{row['case']}_{row['seed']}_{row['step']}";full=json.loads((ROOT/'states'/f'{stem}.json').read_text());model=mujoco.MjModel.from_xml_path(str(ROOT/'states'/f'{stem}.xml'));data=mujoco.MjData(model)
        gids={name:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name) for name in ('cube','table','gripper_left_pad','gripper_right_pad')};site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_center')
        with np.load(ROOT/'states'/f'{stem}.npz') as f:state={k:f[k].copy() for k in f.files}
        names={'original',*[b['best40'] for b in row['banks'].values() if b['best40'] is not None]}
        for name in sorted(names):
            c=next(c for c in full['candidates'] if c['name']==name);mujoco.mj_resetData(model,data);mujoco.mj_setState(model,data,state['integration_state'],int(state['state_type']))
            n=row['dof'];data.ctrl[:n]=np.clip(data.qpos[:n]+np.clip(c['action'],-.04,.04),-2.5,2.5);data.ctrl[n:]=float(state['aperture']);hits=set()
            for k in range(1,81):
                mujoco.mj_step(model,data);hits.update(forbidden(model,data,gids))
                if k%20==0:mujoco.mj_forward(model,data)
                if k in (20,80):
                    vector=np.r_[data.site_xpos[site],.15*data.site_xmat[site].reshape(3,3)[:,2]]
                    np.testing.assert_array_equal(vector,np.array(c[str(k*2)]['vector']))
                    assert [list(p) for p in sorted(hits)]==c[str(k*2)]['pairs']
            checks.append(dict(state=stem,candidate=name,exact_40_and_160=True))
    (ROOT/'snapshot_replay.json').write_text(json.dumps(checks,indent=2));v=json.loads((ROOT/'verification.json').read_text());v['snapshot_branches_exact']=len(checks);(ROOT/'verification.json').write_text(json.dumps(v,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(len(checks),'snapshot branches exact at40/160ms')

if __name__=='__main__':run()
