"""Exact rendered replay of an E43/E44 measured-motion controller trial."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import mujoco
import torch
from .panda_motion import MotionDiagnostic
from .panda_grasp import GraspArm
from .real_arm_reach import load_models
from .pick_place_control import calibrate
from .native_reliability import rollout


def run(assets,models,pilot,output,case,method):
    torch.set_num_threads(4);source=Path(pilot);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text());tasks=json.loads((Path(protocol['source'])/'cases.json').read_text());task=tasks[case]
    loaded,_=load_models(models);env=GraspArm(assets,task['robot'],task['xy'],task['target'],.2,True)
    controller=MotionDiagnostic(env.n,loaded[method],protocol['variant'])
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
        calibrate(env,task['home'],controller,41000+case)
        result=rollout(env,controller,object_release=task['robot']=='panda',render=root)
    expected=next(r for r in json.loads((source/'records.json').read_text()) if r['method']==method and r['case']==case)
    for key in ['trace','phases','success','steps','final_object']:assert result[key]==expected[key],key
    (root/'replay.json').write_text(json.dumps(dict(exact=True,case=case,method=method,**result),indent=2))
    print(task['robot'],method,case,result['success'],result['steps'],'exact')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','models','pilot','output','method']:p.add_argument('--'+key,required=True)
    p.add_argument('--case',type=int,required=True);run(**vars(p.parse_args()))
