"""Verify a rendered E46 fresh-placement neural rollout exactly."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import mujoco
import torch
from .retention_fresh import GuardedPose
from .real_arm_reach import load_models
from .real_arm_pick_place import NativePickPlace
from .pick_place_control import calibrate
from .native_reliability import rollout


def run(assets,pilot,output,case,method):
    torch.set_num_threads(4);source=Path(pilot);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    p=json.loads((source/'protocol.json').read_text());task=json.loads((source/'cases.json').read_text())[case]
    group,seed=method.split('_');assert group in ('old','new')
    models,_=load_models(p[group]);env=NativePickPlace(assets,task['robot'],task['xy'],task['target'],.2,True)
    controller=GuardedPose(env.n,models[f'neural_{seed}'])
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
        calibrate(env,task['home'],controller,p['probe_seed']+case)
        result=rollout(env,controller,object_release=task['robot']=='panda',render=root)
    expected=next(r for r in json.loads((source/'records.json').read_text()) if r['method']==method and r['case']==case)
    for key in ['trace','phases','steps','success','final_object']:assert result[key]==expected[key],key
    (root/'replay.json').write_text(json.dumps(dict(exact=True,method=method,case=case,**result),indent=2))
    print(task['robot'],method,case,result['success'],result['steps'],'exact')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','pilot','output','method']:p.add_argument('--'+key,required=True)
    p.add_argument('--case',type=int,required=True);run(**vars(p.parse_args()))
