"""E50: exact original-controller replays with contact/pose telemetry."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .real_arm_pick_place import NativePickPlace
from .real_arm_reach import load_models
from .real_arms import verify_assets
from .retention_fresh import GuardedPose
from .pick_place_control import calibrate
from .native_reliability import rollout
from .benchmark import fingerprint


def contact_state(env,obs):
    forces=np.zeros(2);counts=np.zeros(2,dtype=int)
    for i,c in enumerate(env.data.contact[:env.data.ncon]):
        pair=[int(c.geom1),int(c.geom2)]
        if env.cube not in pair:continue
        other=pair[1] if pair[0]==env.cube else pair[0]
        body=int(env.model.geom_bodyid[other])
        if body not in env.finger_bodies:continue
        j=env.finger_bodies.index(body);force=np.zeros(6)
        mujoco.mj_contactForce(env.model,env.data,i,force)
        forces[j]+=max(0.,float(force[0]));counts[j]+=1
    relative=obs['rotation'].T@(env.data.xpos[env.obj]-obs['position'])
    object_rotation=env.data.xmat[env.obj].reshape(3,3)
    angle=np.arctan2(obs['rotation'][1,0],obs['rotation'][0,0])-np.arctan2(object_rotation[1,0],object_rotation[0,0])
    yaw=float(abs((angle+np.pi/4)%(np.pi/2)-np.pi/4))
    return dict(opening=env.opening(),normal_forces=forces.tolist(),contact_counts=counts.tolist(),
        object_in_tool=relative.tolist(),yaw_error=yaw,tool_rotation=obs['rotation'].tolist(),
        gripper_force=float(env.data.actuator_force[env.gripper]))


def run(assets,models,reference,output):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    ref=Path(reference);tasks=json.loads((ref/'cases.json').read_text());old=json.loads((ref/'records.json').read_text())
    loaded,hashes=load_models(models);manifest=verify_assets(assets)
    protocol=dict(experiment='E50',reference=reference,reference_sha256=fingerprint(ref/'records.json'),
        cases_sha256=fingerprint(ref/'cases.json'),checkpoints=hashes,asset_commit=manifest['commit'],test_evaluated=False,
        scope='48 exact full original E35 replays: analytical plus three neural seeds, all twelve observed E46 placements. No local refresh. Log current-step cube/finger normal forces, opening, relative object/tool pose, cube-symmetric yaw mismatch, and controller motion/bias/action. Telemetry never enters control. No success-rule, mechanics or weight change.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));shutil.copyfile(ref/'cases.json',root/'cases.json');(root/'source').mkdir()
    for name in ['contact_diagnostic.py','native_reliability.py','real_arm_pick_place.py','real_arms.py','pick_place_control.py','retention_fresh.py','motion_transformer.py']:
        shutil.copyfile(Path(__file__).with_name(name),root/'source'/name)
    rows=[]
    for method,model in loaded.items():
        if method=='privileged':continue
        for case,t in enumerate(tasks):
            env=NativePickPlace(assets,t['robot'],t['xy'],t['target'],.2,True);controller=GuardedPose(env.n,model);telemetry=[]
            def observe(trace,before,after,command,action,motion,bias,jac,error):
                telemetry.append(dict(phase=trace['phase'],**contact_state(env,after),
                    motion_norm=float(np.linalg.norm(motion)),bias_norm=float(np.linalg.norm(bias)),
                    action_norm=float(np.linalg.norm(action)),clipped_joints=int(np.sum(np.abs(motion+bias)>.04))))
            with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
                probe=calibrate(env,t['home'],controller,41000+case)
                result=rollout(env,controller,object_release=t['robot']=='panda',step_observer=observe)
            expected=next(r for r in old if (r['method'],r['case'])==(method.replace('neural_','old_'),case))
            for key in ['trace','phases','steps','success','failure','final_object']:assert result[key]==expected[key]
            assert probe==expected['probe'];assert len(telemetry)==len(result['trace'])
            row=dict(method=method,case=case,robot=t['robot'],probe=probe,telemetry=telemetry,**result)
            rows.append(row);(root/f'{method}_{case}.json').write_text(json.dumps(row))
            print(method,case,result['success'],result['failure'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    assert fingerprint(ref/'records.json')==protocol['reference_sha256']
    (root/'records.json').write_text(json.dumps(rows))
    summary=[]
    for r in rows:
        phase_stats={}
        for phase in ['descend','close','lift','transport']:
            ts=[(t,d) for t,d in zip(r['trace'],r['telemetry']) if t['phase']==phase]
            if not ts:continue
            phase_stats[phase]=dict(steps=len(ts),last=ts[-1][1],
                bilateral_fraction=float(np.mean([t['pads']==2 for t,d in ts])),
                max_position_error=max(t['position_error'] for t,d in ts),
                max_rotation_error=max(t['rotation_error'] for t,d in ts),
                start_relative=ts[0][1]['object_in_tool'],end_relative=ts[-1][1]['object_in_tool'],
                object_height_end=ts[-1][0]['object'][2])
        summary.append(dict(method=r['method'],case=r['case'],robot=r['robot'],success=r['success'],failure=r['failure'],phases=phase_stats))
    (root/'summary.json').write_text(json.dumps(summary,indent=2));(root/'audit.json').write_text(json.dumps(dict(exact_full_replays=48,unchanged_weights=True,telemetry_diagnostic_only=True,test_evaluated=False),indent=2))
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','models','reference','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))
