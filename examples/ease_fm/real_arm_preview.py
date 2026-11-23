"""Render an exact native-arm controller replay against saved records."""
import argparse
import json
from pathlib import Path
from .real_arm_reach import load_models
from .real_arm_pick_place import NativePickPlace
from .pick_place_control import PoseController,calibrate
from .arm_pick_place import rollout


def run(assets,pilot,models,output,case=0,method='neural_0'):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);pilot=Path(pilot)
    protocol=json.loads((pilot/'protocol.json').read_text());task=json.loads((pilot/'cases.json').read_text())[case]
    loaded,_=load_models(models);model=loaded[method]
    env=NativePickPlace(assets,task['robot'],task['xy'],task['target'],protocol.get('base_height',0),protocol.get('compact',False))
    controller=PoseController(env.n,model);calibrate(env,task['home'],controller,40000+case)
    result=rollout(env,task['home'],None if method=='privileged' else controller,render=root,reset=False,sequence=env.sequence())
    expected=next(r for r in json.loads((pilot/'records.json').read_text()) if r['method']==method and r['case']==case)
    assert all(result[k]==expected[k] for k in ['trace','steps','success','final_object'])
    (root/'replay.json').write_text(json.dumps(dict(robot=task['robot'],method=method,case=case,exact=True,**result),indent=2))
    print(task['robot'],method,'exact replay',result['success'],result['steps'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','pilot','models','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--case',type=int,default=0);p.add_argument('--method',default='neural_0');run(**vars(p.parse_args()))
