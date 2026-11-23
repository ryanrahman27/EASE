"""E43: measured-motion diagnostics and frozen Transformer inference ablations."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .panda_grasp import GraspArm
from .pick_place_control import PoseController,calibrate
from .native_reliability import rollout
from .real_arm_reach import load_models
from .real_arms import verify_assets
from .benchmark import fingerprint


class MotionDiagnostic(PoseController):
    def __init__(self,n,model=None,variant='baseline'):
        super().__init__(n,model);self.variant=variant;self.predictions=[];self.pending=None;self.smoothed=None

    @torch.no_grad()
    def predict(self):
        if self.variant=='scale' and self.model is not None:
            saved=self.cross.copy();self.cross[:,3:]/=.15
            try:jac,bias=super().predict()
            finally:self.cross=saved
            jac[3:]*=.15
        else:jac,bias=super().predict()
        if self.variant=='smooth' and self.model is not None:
            self.smoothed=jac.copy() if self.smoothed is None else .8*self.smoothed+.2*jac
            jac=self.smoothed.copy()
        # Shadow fit is logged only; it never enters the executed control action.
        shadow=np.linalg.solve(self.gram,self.cross).T
        self.pending=(jac.copy(),shadow,float(np.linalg.cond(self.gram)))
        return jac,bias

    def update(self,before,after,command):
        if self.pending is not None:
            jac,shadow,condition=self.pending;dq=after['q']-before['q']
            dx=np.r_[after['position']-before['position'],.15*(after['rotation'][:,2]-before['rotation'][:,2])]
            self.predictions.append(dict(dq_norm=float(np.linalg.norm(dq)),condition=condition,
                position_motion=float(np.linalg.norm(dx[:3])),direction_motion=float(np.linalg.norm(dx[3:])),
                position_error=float(np.linalg.norm((jac@dq-dx)[:3])),direction_error=float(np.linalg.norm((jac@dq-dx)[3:])),
                shadow_position_error=float(np.linalg.norm((shadow@dq-dx)[:3])),shadow_direction_error=float(np.linalg.norm((shadow@dq-dx)[3:]))))
            self.pending=None
        super().update(before,after,command)


def run(assets,models,cases,reference,output,variant='baseline',robots='panda',experiment=None):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    tasks=json.loads((Path(cases)/'cases.json').read_text());loaded,hashes=load_models(models);manifest=verify_assets(assets)
    protocol=dict(experiment=experiment or ('E44' if variant=='trained' else 'E43'),variant=variant,robots=robots,source=cases,reference=reference,
        cases_sha256=fingerprint(Path(cases)/'cases.json'),commit=manifest['commit'],checkpoints=hashes,test_evaluated=False,
        intervention='Baseline exact E42 object-height release. Scale feeds unscaled tool-direction cross statistics through the same neural estimator and rescales predicted direction Jacobian by .15. Smooth uses .8 previous+.2 new predicted Jacobian. Analytical control is unchanged. Shadow analytical predictions are diagnostic only; no fallback or action blending.',
        scope='Matched observed native cases for selected robots, same calibration and fixed model weights. Logged one-step prediction errors use next measured motion; no simulator Jacobian access.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'source').mkdir()
    for name in ['panda_motion.py','panda_grasp.py','native_reliability.py','pick_place_control.py','real_arms.py','real_arm_pick_place.py','motion_transformer.py']:
        shutil.copyfile(Path(__file__).with_name(name),root/'source'/name)
    rows=[];old=json.loads((Path(reference)/'records.json').read_text())
    for method,model in loaded.items():
        if method=='privileged' or (method=='analytical' and variant!='baseline'):continue
        for i,task in enumerate(tasks):
            if task['robot'] not in robots.split(','):continue
            env=GraspArm(assets,task['robot'],task['xy'],task['target'],.2,True);controller=MotionDiagnostic(env.n,model,variant)
            with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
                probe=calibrate(env,task['home'],controller,41000+i)
                result=rollout(env,controller,object_release=task['robot']=='panda')
            if variant=='baseline':
                expected=next(r for r in old if r['method']==method and r['case']==i)
                for key in ['trace','phases','steps','success','final_object']:assert result[key]==expected[key],key
            assert len(controller.predictions)==len(result['trace'])
            rows.append(dict(method=method,case=i,robot=task['robot'],probe=probe,predictions=controller.predictions,**result))
            (root/'records.json').write_text(json.dumps(rows,indent=2))
            print(variant,method,i,result['success'],result['failure'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','models','cases','reference','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--variant',choices=['baseline','scale','smooth','trained'],default='baseline')
    p.add_argument('--robots',default='panda');p.add_argument('--experiment');run(**vars(p.parse_args()))
