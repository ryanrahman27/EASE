"""E51: frozen learned angular Jacobians with six independent pose coordinates."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .pose_geometry import rotation_vector,upright_target
from .retention_fresh import GuardedPose
from .real_arm_pick_place import NativePickPlace
from .real_arm_reach import load_models
from .real_arms import verify_assets
from .pick_place_control import calibrate
from .native_reliability import rollout
from .contact_diagnostic import contact_state
from .benchmark import fingerprint


class AngularPose(GuardedPose):
    def update(self,before,after,command):
        dq=after['q']-before['q']
        dx=np.r_[after['position']-before['position'],.15*rotation_vector(after['rotation']@before['rotation'].T)]
        self.gram=.98*self.gram+np.outer(dq,dq)+2e-8*np.eye(self.n)
        self.cross=.98*self.cross+np.outer(dq,dx)
        self.bias=.8*self.bias+.2*(command-dq)


class AngularArm(NativePickPlace):
    def jacobian(self):
        jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jacSite(self.model,self.data,jp,jr,self.site)
        return np.concatenate([jp[:,self.vids],.15*jr[:,self.vids]])


def run(assets,models,cases,output):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    tasks=json.loads(Path(cases).read_text());loaded,hashes=load_models(models);manifest=verify_assets(assets)
    protocol=dict(experiment='E51',cases_sha256=fingerprint(cases),checkpoints=hashes,asset_commit=manifest['commit'],
        conditions=['hold','align'],test_evaluated=False,
        hypothesis='E50 shows corner-oriented Panda grips and subsequent slip. Test full angular motion identification/control with six independent XYZ/rotation-vector coordinates, unlike E42 nine-component axis tracking. Compare upright initial yaw (hold) with nearest world-cardinal cube-face alignment (align). One configuration for both robots and all seeds; original E35 weights frozen.',
        control='Measured angular increment log(R_after R_before.T), scaled .15; same two model calls, shared Gram/cross representation and bias update. Orientation error log(R_target R_current.T). Same gain .35, damping .015, joint clip .04, reference ramp .002. Existing eight-hold .006 m gate now requires full angular error <.08 rad. Native mechanics, phase budgets, object-release rule and final physical success unchanged. No added refresh or target-robot training.',
        fairness='All twelve original E46 homes retained without new IK filtering. Both variants use the same 32 probe actions and seed 41000+case. Targets selected from the common pre-calibration home frame; cubes initially axis-aligned. All failures and total steps retained. Analytical/privileged get the same target. This is a development representation-and-control change, not independent final-test evidence.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));shutil.copyfile(cases,root/'cases.json');(root/'source').mkdir()
    for name in ['angular_control.py','pose_geometry.py','native_reliability.py','contact_diagnostic.py','pick_place_control.py','retention_fresh.py','real_arm_pick_place.py','real_arms.py','motion_transformer.py']:
        shutil.copyfile(Path(__file__).with_name(name),root/'source'/name)
    rows=[]
    for condition in ['hold','align']:
        for method,model in loaded.items():
            for case,t in enumerate(tasks):
                env=AngularArm(assets,t['robot'],t['xy'],t['target'],.2,True)
                controller=AngularPose(env.n,model);target=upright_target(env.reset(t['home'])['rotation'],condition=='align');telemetry=[]
                def observe(trace,before,after,command,action,motion,bias,jac,error):
                    telemetry.append(dict(phase=trace['phase'],**contact_state(env,after)))
                guard=nullcontext() if method=='privileged' else patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden'))
                with guard:
                    probe=calibrate(env,t['home'],controller,41000+case)
                    result=rollout(env,None if method=='privileged' else controller,orientation_target=target,
                        object_release=t['robot']=='panda',step_observer=observe)
                assert result['steps']==32+len(result['trace'])
                row=dict(condition=condition,method=method,case=case,robot=t['robot'],target=target.tolist(),probe=probe,telemetry=telemetry,**result)
                rows.append(row);(root/f'{condition}_{method}_{case}.json').write_text(json.dumps(row))
                print(condition,method,case,result['success'],result['failure'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'records.json').write_text(json.dumps(rows));summary=[]
    for condition in ['hold','align']:
        for robot in ['piper','panda']:
            for method in loaded:
                sub=[r for r in rows if (r['condition'],r['robot'],r['method'])==(condition,robot,method)]
                summary.append(dict(condition=condition,robot=robot,method=method,success=sum(r['success'] for r in sub),n=len(sub),
                    collisions=sum(r['collision'] for r in sub),mean_steps=float(np.mean([r['steps'] for r in sub])),
                    failures=[dict(case=r['case'],failure=r['failure']) for r in sub if not r['success']]))
    for case in range(len(tasks)):
        group=[r for r in rows if r['case']==case]
        assert all(r['probe']==group[0]['probe'] for r in group)
        for condition in ['hold','align']:
            targets=[r['target'] for r in group if r['condition']==condition];assert all(t==targets[0] for t in targets)
    assert len(rows)==120
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    (root/'audit.json').write_text(json.dumps(dict(records=120,matched_calibration=True,matched_targets=True,unchanged_weights=True,step_accounting=True,test_evaluated=False),indent=2))
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','models','cases','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))
