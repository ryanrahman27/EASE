"""Render an exact controller replay, preserving its measured physical outcome."""
import argparse
import json
from pathlib import Path
import torch
from .mixed_sim import AXES,Arm
from .arm_pick_place import PickPlace,rollout
from .pick_place_control import PoseController,calibrate
from .motion_transformer import MotionTransformer


def run(cases,checkpoint,case,output):
    torch.set_num_threads(4);source=Path(cases);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    AXES.update(json.loads((source/'axis_catalog.json').read_text()));task=json.loads((source/'cases.json').read_text())[case]
    m=MotionTransformer().cuda();m.load_state_dict(torch.load(Path(checkpoint)/'policy.pt',map_location='cuda',weights_only=True));m.eval()
    env=PickPlace(Arm(**task['arm']),task['xy'],task['target']);controller=PoseController(env.n,m)
    calibrate(env,task['home'],controller,38000+case)
    result=rollout(env,task['home'],controller,render=root,reset=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print({k:v for k,v in result.items() if k not in ('trace','phases')})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['cases','checkpoint','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--case',type=int,default=0);run(**vars(p.parse_args()))
