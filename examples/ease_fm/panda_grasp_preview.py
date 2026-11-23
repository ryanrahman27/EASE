"""Exact rendered replay for the object-height Panda release intervention."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import mujoco
import torch
from .panda_grasp import GraspArm
from .pick_place_control import PoseController,calibrate
from .real_arm_reach import load_models
from .native_reliability import rollout


def run(assets,models,pilot,output,case,method):
    torch.set_num_threads(4);source=Path(pilot);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text());assert protocol['variant']=='supported'
    task=json.loads((Path(protocol['source'])/'cases.json').read_text())[case]
    loaded,_=load_models(models);env=GraspArm(assets,'panda',task['xy'],task['target'],.2,True)
    controller=PoseController(env.n,loaded[method]);assert method.startswith('neural_')
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
        calibrate(env,task['home'],controller,41000+case)
        result=rollout(env,controller,render=root,object_release=True)
    expected=next(r for r in json.loads((source/'records.json').read_text()) if r['method']==method and r['case']==case)
    for key in ['trace','phases','steps','success','final_object']:assert result[key]==expected[key],key
    (root/'replay.json').write_text(json.dumps(dict(exact=True,case=case,method=method,**result),indent=2))
    print(method,case,result['success'],result['steps'],'exact')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','models','pilot','output','method']:p.add_argument('--'+key,required=True)
    p.add_argument('--case',type=int,required=True);run(**vars(p.parse_args()))
