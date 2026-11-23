"""Render and exactly replay one E47 neural trial."""
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
from .local_refresh import refresh


def run(assets,models,pilot,output,case,method,condition):
    torch.set_num_threads(4);source=Path(pilot);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    task=json.loads((source/'cases.json').read_text())[case]
    loaded,hashes=load_models(models)
    assert hashes==json.loads((source/'protocol.json').read_text())['checkpoints']
    assert method.startswith('neural_')
    env=NativePickPlace(assets,task['robot'],task['xy'],task['target'],.2,True)
    controller=GuardedPose(env.n,loaded[method])
    hook=None if condition=='baseline' else lambda e,c:refresh(e,c,condition,47000+case)
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
        calibrate(env,task['home'],controller,41000+case)
        result=rollout(env,controller,object_release=task['robot']=='panda',render=root,pregrasp_hook=hook)
    expected=next(r for r in json.loads((source/'records.json').read_text()) if (r['method'],r['case'],r['condition'])==(method,case,condition))
    for key in ['trace','phases','steps','success','final_object']:assert result[key]==expected[key],key
    (root/'replay.json').write_text(json.dumps(dict(exact=True,method=method,case=case,condition=condition,**result),indent=2))
    print(task['robot'],method,condition,case,result['success'],result['steps'],'exact')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','models','pilot','output','method','condition']:p.add_argument('--'+key,required=True)
    p.add_argument('--case',type=int,required=True);run(**vars(p.parse_args()))
