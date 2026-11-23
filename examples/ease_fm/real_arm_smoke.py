"""E39 native-model compilation, interface, and gripper smoke audit."""
import argparse
import json
from pathlib import Path
import numpy as np
from .real_arms import RealArm,verify_assets


def run(assets,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);manifest=verify_assets(assets);rows=[]
    for name in ['piper','panda']:
        env=RealArm(assets,name);obs=env.observe();home=obs['q'].copy()
        row=dict(robot=name,arm_joints=env.n,nq=env.model.nq,nu=env.model.nu,home=home.tolist(),tool=obs['position'].tolist(),
            tool_direction=obs['rotation'][:,2].tolist(),joint_limits=env.limits.tolist(),arm_actuators=env.aids.tolist(),
            gripper_range=env.model.actuator_ctrlrange[env.gripper].tolist(),gravity_compensated_bodies=int((env.model.body_gravcomp>0).sum()),
            initial_contacts=int(env.data.ncon),native_timestep=env.model.opt.timestep,control_substeps=env.substeps)
        for openness in [0.,1.]:
            for _ in range(50):obs,_,contacts=env.step(home-obs['q'],openness)
            row['closed_width' if openness==0 else 'open_width']=env.opening()
        assert row['open_width']>row['closed_width']+.03
        rows.append(row);print(json.dumps(row),flush=True)
    (root/'audit.json').write_text(json.dumps(dict(commit=manifest['commit'],asset_files_verified=len(manifest['files']),robots=rows,test_evaluated=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--output',required=True);run(**vars(p.parse_args()))
