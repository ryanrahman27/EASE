"""Verify a rendered E41 revision-3 replay against its recorded rollout."""
import argparse
import json
from pathlib import Path
import torch
from unittest.mock import patch
import mujoco
from .real_arm_reach import load_models
from .real_arm_pick_place import NativePickPlace
from .pick_place_control import PoseController,calibrate
from .native_reliability import rollout


def run(assets,models,pilot,output,case,method):
    torch.set_num_threads(4);source=Path(pilot);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((source/'protocol.json').read_text());assert protocol['revision']==3
    task=json.loads((source/'cases.json').read_text())[case];loaded,_=load_models(models)
    assert method.startswith('neural_') or method=='analytical'
    env=NativePickPlace(assets,task['robot'],task['xy'],task['target'],.2,True)
    controller=PoseController(env.n,loaded[method])
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
        calibrate(env,task['home'],controller,41000+case)
        result=rollout(env,controller,render=root)
    expected=next(r for r in json.loads((source/'records.json').read_text()) if r['case']==case and r['method']==method)
    for key in ['trace','phases','success','steps','final_object']:assert result[key]==expected[key],key
    (root/'replay.json').write_text(json.dumps(dict(exact=True,pilot=pilot,case=case,method=method,**result),indent=2))
    print(task['robot'],method,case,'exact',result['success'],result['steps'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','models','pilot','output','method']:p.add_argument('--'+key,required=True)
    p.add_argument('--case',type=int,required=True);run(**vars(p.parse_args()))
