"""Independent single-robot interface replay on all fixed validation tasks."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from .motion_transformer import MotionController
from .paired_data import register
from .mixed_sim import Arm, MixedReach
from .benchmark import fingerprint


def run(data, checkpoint, output):
    torch.set_num_threads(4)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    manifest=register(data);arms={a['seed']:Arm(**a) for a in manifest['arms']}
    pairs={p['id']:p['bodies'] for p in manifest['pairs']}
    model=MotionController.from_checkpoint(checkpoint,6).model
    original_solve=np.linalg.solve
    def solve(a,b):
        assert a.shape==(3,3),'Joint-model inverse forbidden'
        return original_solve(a,b)
    records=[]
    with patch.object(mujoco,'mj_jacSite',side_effect=AssertionError('Privileged Jacobian forbidden')), \
         patch.object(np.linalg,'solve',side_effect=solve), \
         patch.object(torch.linalg,'solve',side_effect=AssertionError('Neural inverse forbidden')):
        for goal in manifest['goals']['validation']:
            for body in pairs[goal['pair']]:
                arm=replace(arms[body],home=tuple(goal['home']))
                controller=MotionController(model,arm.dof)
                env=MixedReach(arm,synchronize=True);before=env.reset(0,goal['xyz'],spread=0)
                for _ in range(150):
                    after,done,info=env.step(controller.action(before))
                    controller.update(before,after,info['executed_action']);before=after
                    if done:break
                records.append(dict(body_seed=body,goal_id=goal['id'],dof=arm.dof,condition=goal['condition'],steps=env.steps,
                    **{k:info[k] for k in ('success','error','collision')}))
    (root/'records.json').write_text(json.dumps(records,indent=2))
    (root/'audit.json').write_text(json.dumps(dict(manifest_sha256=fingerprint(Path(data)/'manifest.json'),
        checkpoint_sha256=fingerprint(Path(checkpoint)/'policy.pt'),controller_source_sha256=fingerprint(Path(__file__).with_name('motion_transformer.py')),
        records_sha256=fingerprint(root/'records.json'),successes=sum(r['success'] for r in records),trials=len(records),
        joint_inverse_and_privileged_api_blocked=True,calibration_steps=0,test_evaluated=False),indent=2))
    print(sum(r['success'] for r in records),'/',len(records),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('data','checkpoint','output'):parser.add_argument('--'+key,required=True)
    run(**vars(parser.parse_args()))
