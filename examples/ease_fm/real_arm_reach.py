"""E39 frozen-policy native-arm reaching gate, with native grippers and limits."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from .real_arms import RealArm,verify_assets
from .pick_place_control import PoseController
from .motion_transformer import MotionTransformer
from .benchmark import fingerprint
from .arm_pick_place import ROT_SCALE


def load_models(folder):
    models={'privileged':None,'analytical':None};hashes={}
    for seed in range(3):
        path=Path(folder)/f'seed_{seed}'/'policy.pt';sha=fingerprint(path)
        assert sha==json.loads(path.with_name('training.json').read_text())['policy_sha256']
        m=MotionTransformer().cuda();m.load_state_dict(torch.load(path,map_location='cuda',weights_only=True));models[f'neural_{seed}']=m.eval();hashes[str(path)]=sha
    return models,hashes


def probe(env,home,controller,seed):
    obs=env.reset(home);rng=np.random.default_rng(seed);collision=False
    for t in range(32):
        if t%4==0:offset=rng.uniform(-.012,.012,env.n)
        before=obs;obs,command,contacts=env.step(home+offset-obs['q'],1.)
        controller.update(before,obs,command);collision |= bool(contacts)
    return obs,collision


def run(assets,models,output):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    manifest=verify_assets(assets);loaded,hashes=load_models(models)
    (root/'protocol.json').write_text(json.dumps(dict(experiment='E39_reach',asset_commit=manifest['commit'],checkpoints=hashes,
        goals='12 Cartesian offsets per robot: +/- each XYZ axis at .04 and .08 m from native home tool position, maintaining home tool direction. No success filtering.',
        controller='Provided fixed checkpoint weights (hashes recorded above), .35 task gain, .015 damping, .04 rad action clip, native joint/actuator limits. All methods get same 32-step bounded measured probe then up to 200 task steps.',
        success='Five consecutive task steps within .015 m and tool-direction difference .08; no contact during probe/task.',
        native='Preserve native actuators and gravity compensation: PiPER compensated, Panda uncompensated. This is simulated transfer to real robot designs, not hardware validation.',test_evaluated=False),indent=2))
    (root/'source').mkdir()
    for name in ['real_arms.py','real_arm_reach.py','pick_place_control.py']:shutil.copyfile(Path(__file__).with_name(name),root/'source'/name)
    rows=[]
    for name in ['piper','panda']:
        env=RealArm(assets,name);initial=env.reset();home=initial['q'];direction=initial['rotation'][:,2]
        goals=[initial['position']+sign*radius*np.eye(3)[axis] for radius in [.04,.08] for axis in range(3) for sign in [-1,1]]
        for method,model in loaded.items():
            for i,goal in enumerate(goals):
                controller=PoseController(env.n,model)
                guard=nullcontext() if method=='privileged' else patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden'))
                with guard:
                    obs,collision=probe(env,home,controller,39000+i);held=0
                    for step in range(200):
                        if method=='privileged':
                            jac=env.jacobian();bias=(env.data.qfrc_bias[env.vids]-env.data.qfrc_passive[env.vids])/env.model.actuator_gainprm[env.aids,0]
                        else:
                            if model is not None:
                                with patch.object(np.linalg,'solve',side_effect=RuntimeError('Neural inverse forbidden')),patch.object(torch.linalg,'solve',side_effect=RuntimeError('Neural inverse forbidden')):jac,bias=controller.predict()
                            else:jac,bias=controller.predict()
                        error=np.r_[goal-obs['position'],ROT_SCALE*(direction-obs['rotation'][:,2])]
                        action=jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(6),.35*error)+bias
                        before=obs;obs,command,contacts=env.step(action,1.);controller.update(before,obs,command)
                        collision |= bool(contacts);pe=float(np.linalg.norm(goal-obs['position']));de=float(np.linalg.norm(direction-obs['rotation'][:,2]))
                        held=held+1 if pe<.015 and de<.08 and not collision else 0
                        if held>=5:break
                rows.append(dict(robot=name,dof=env.n,method=method,goal_id=i,goal=goal.tolist(),success=held>=5,collision=bool(collision),task_steps=step+1,total_steps=env.steps,error=pe,direction_error=de))
            subset=[r for r in rows if r['robot']==name and r['method']==method]
            print(name,method,sum(r['success'] for r in subset),'/12',flush=True)
            (root/'records.json').write_text(json.dumps(rows,indent=2))
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','models','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))
