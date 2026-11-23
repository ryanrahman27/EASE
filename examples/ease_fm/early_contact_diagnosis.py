"""E163: offline first-contact geometry diagnosis on fitting six-joint bodies."""
from pathlib import Path
import numpy as np
import mujoco
from .abstaining_task_value_gate import read,save
from .release_clearance_control import ROOT as BASE
ROOT=Path('outputs/ease_early_contact_diagnosis_v1')

def run():
    ROOT.mkdir(exist_ok=True);out=[];tasks=read(BASE/'cases.json')
    for r in read(BASE/'records.json'):
        if r['dof']!=6:continue
        stem=f"{r['case']}_e120_ensemble_frozen_bias";t=read(BASE/'trials'/f'{stem}.json');first=next((i for i,x in enumerate(t['trace']) if x['bad']),None)
        if first is None:continue
        model=mujoco.MjModel.from_xml_path(str(BASE/'states'/f'{stem}.xml'));d=mujoco.MjData(model);states=np.load(BASE/'states'/f'{stem}.npz')['states'];snapshots=[]
        for step in sorted(set([0,max(0,first-10),max(0,first-1),first])):
            s=states[step];d.qpos[:]=s[:model.nq];d.qvel[:]=s[model.nq:];mujoco.mj_forward(model,d)
            table=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'table');distances={}
            for k in range(6):
                body=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,f'link{k}');g=int(model.body_geomadr[body]);assert model.geom_bodyid[g]==body
                distances[f'link{k}:geom{g}']=float(mujoco.mj_geomDistance(model,d,g,table,1.,None))
            snapshots.append(dict(step=step,q=d.qpos[:6].tolist(),link_table_distance_m=distances,trace=t['trace'][step]))
        out.append(dict(case=r['case'],body=r['body'],first_contact_step=first,phase=t['trace'][first]['phase'],contacts=t['trace'][first]['bad'],home=tasks[r['case']]['home'],snapshots=snapshots))
    save(ROOT/'diagnosis.json',out);save(ROOT/'protocol.json',dict(scope='Offline simulator geometry at saved fitting states only; no geometry labels enter learned estimator, training or held evaluation. Post-control snapshots can miss transient within-step contacts.',baseline=str(BASE),six_joint_collision_episodes=len(out)))
    import shutil
    shutil.copy2(__file__,ROOT/'source.py')
    print(mujoco.mj_geomDistance.__doc__)
    print([(r['case'],r['body'],r['phase'],r['snapshots'][-2]['link_table_distance_m']) for r in out],flush=True)

if __name__=='__main__':run()
