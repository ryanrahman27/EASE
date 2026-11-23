"""E32 post-E31 cold-start controls; no calibration and no privileged Jacobian."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from .paired_data import register
from .mixed_sim import Arm, MixedReach
from .kinematic_control import IdentifiedController
from .motion_transformer import MotionTransformer
from .benchmark import fingerprint


def run(data, pilot, output):
    torch.set_num_threads(4)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    manifest=register(data); arms={a['seed']:Arm(**a) for a in manifest['arms']}; pairs={p['id']:p['bodies'] for p in manifest['pairs']}
    meta=json.loads((Path(pilot)/'training.json').read_text())
    assert fingerprint(Path(pilot)/'policy.pt')==meta['policy_sha256']
    model=MotionTransformer().eval()
    model.load_state_dict(torch.load(Path(pilot)/'policy.pt',map_location='cpu',weights_only=True))
    protocol=dict(experiment='E32',manifest_sha256=fingerprint(Path(data)/'manifest.json'),
        model_sha256=meta['policy_sha256'],calibration_steps=0,test_evaluated=False,
        scope='Post-E31 cold-start controls, unchanged fixed validation cases. Analytical zero initialization, seed0 neural first-step bootstrap followed by analytical updates, fixed seed0 neural zero-history Jacobian with only tracking-bias updates, and a fixed +0.01 rad all-joint first action followed by analytical updates. Fixed probe added after the initial three controls; no gain tuning.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    source=root/'source';source.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,source/p.name)
    records=[]
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
        for mode in ('analytical_cold','neural_bootstrap_analytical','neural_constant','fixed_probe_analytical'):
            rows=[]
            for goal in manifest['goals']['validation']:
                for body in pairs[goal['pair']]:
                    arm=replace(arms[body],home=tuple(goal['home'])); n=arm.dof
                    controller=IdentifiedController(np.zeros((32,n,11)),True,update_jacobian=mode!='neural_constant')
                    if mode in ('neural_bootstrap_analytical','neural_constant'):
                        g=torch.zeros(1,7,7);g[0,:n,:n]=1e-6*torch.eye(n)
                        with torch.no_grad():controller.jac=model(g,torch.zeros(1,7,3),torch.arange(7)[None]<n)[0,:n].numpy().T.astype(float)
                    env=MixedReach(arm,synchronize=True); before=env.reset(0,goal['xyz'],spread=0)
                    for step in range(150):
                        action=np.full(n,.01) if mode=='fixed_probe_analytical' and step==0 else controller.action(before)
                        after,done,info=env.step(action)
                        controller.update(before,after,info['executed_action']);before=after
                        if done:break
                    rows.append(dict(mode=mode,body_seed=body,dof=n,goal_id=goal['id'],condition=goal['condition'],steps=env.steps,
                        **{k:info[k] for k in ('success','error','collision')}))
            records.extend(rows);(root/'records.json').write_text(json.dumps(records,indent=2))
            print(mode,sum(r['success'] for r in rows),'/',len(rows),flush=True)
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('data','pilot','output'):parser.add_argument('--'+key,required=True)
    run(**vars(parser.parse_args()))
