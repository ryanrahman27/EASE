"""Audit serial-arm task assets and local pose-control rank before policy claims."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np
from .mixed_sim import Arm, MixedReach
from .paired_data import register
from .pick_place_fixture import arm_fixture_xml


def run(data,output):
    manifest=register(data);root=Path(output);root.mkdir(parents=True,exist_ok=False);rows=[]
    shutil.copyfile(Path(__file__),root/'audit_source.py')
    shutil.copyfile(Path(__file__).with_name('pick_place_fixture.py'),root/'fixture_source.py')
    for raw in manifest['arms']:
        arm=Arm(**raw);xml=arm_fixture_xml(arm);(root/f'arm_{arm.seed}.xml').write_text(xml)
        model=mujoco.MjModel.from_xml_string(xml);sim=mujoco.MjData(model)
        homes=next(f['homes'] for f in manifest['families'] if arm.seed in f['bodies'])
        for posture,home in enumerate(homes):
            env=MixedReach(replace(arm,home=tuple(home)),synchronize=True);env.reset(0,[0,0,.9],spread=0)
            jp=np.zeros((3,arm.dof));jr=np.zeros_like(jp);mujoco.mj_jacSite(env.model,env.data,jp,jr,0)
            sim.qpos[:arm.dof]=home;mujoco.mj_forward(model,sim)
            contacts=[]
            for c in sim.contact[:sim.ncon]:
                names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in (c.geom1,c.geom2)]
                contacts.append(names)
            rows.append(dict(body=arm.seed,dof=arm.dof,posture=posture,position_rank=int(np.linalg.matrix_rank(jp,tol=1e-7)),
                pose_rank=int(np.linalg.matrix_rank(np.concatenate([jp,jr]),tol=1e-7)),initial_scene_contacts=contacts))
    (root/'results.json').write_text(json.dumps(rows,indent=2))
    summary=dict(configurations=len(rows),rank6=sum(r['pose_rank']==6 for r in rows),rank5=sum(r['pose_rank']==5 for r in rows),
        rank_below5=sum(r['pose_rank']<5 for r in rows),scope='Local arm rank only, not reachability or grasp feasibility. Exported arm scenes are construction assets, not validated reset/task datasets.',test_evaluated=False)
    (root/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);run(**vars(p.parse_args()))
