"""E97: matched learned/planner/probe prefix, post-probe estimator substitution."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from . import local_probe as prior
from .local_history import LocalPose
from .probe_estimation import ROOT as AUDIT
from .mixed_sim import AXES
from .benchmark import fingerprint

ROOT=Path('outputs/ease_probe_continuation_v1')
PRIOR=prior.PRIOR
models=prior.models


def continuation_prediction(controller,env,mode,active,original_jac):
    if not active or mode=='learned':return LocalPose.predict(controller)
    if mode=='analytical':return np.linalg.solve(controller.gram,controller.cross).T,controller.bias
    if mode=='exact':
        with patch.object(mujoco,'mj_jacSite',original_jac):return env.jacobian(),controller.bias
    raise ValueError(mode)


def evaluate(case,task,method,model,mode,record_step=None):
    assert mode in ('learned','analytical','exact')
    captured={};active=[];original_jac=mujoco.mj_jacSite
    class BranchArm(prior.ProbeArm):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);active.append(self)
    class BranchPose(LocalPose):
        switched=False
        def predict(self):
            env=active[0]
            if getattr(env,'phase',None)=='close' and not captured:
                captured.update(state=dict(qpos=env.data.qpos.tolist(),qvel=env.data.qvel.tolist(),ctrl=env.data.ctrl.tolist(),
                    time=float(env.data.time),gram=self.gram.tolist(),cross=self.cross.tolist(),bias=self.bias.tolist()),
                    history=[(dq.tolist(),dx.tolist()) for dq,dx in self.history])
                expected=json.loads((AUDIT/'snapshots'/f'{case}_{method}_probe32.json').read_text())
                assert captured['state']==expected['state']
                assert [list(h) for h in captured['history']]==expected['history']
                self.switched=env.route is not None
            return continuation_prediction(self,env,mode,self.switched,original_jac)
    with patch.object(prior,'ProbeArm',BranchArm),patch.object(prior,'LocalPose',BranchPose):
        result=prior.evaluate(case,task,method,model,'probe32',record_step)
    old=json.loads((prior.ROOT/'trials'/f'{case}_{method}_probe32.json').read_text())
    prefix=lambda r:[t for t in r['trace'] if t['phase'] in ('approach','descend','probe')]
    assert prefix(result)==prefix(old) and captured
    if mode=='learned' or not result['route_used']:assert result==old
    result.update(mode=mode,continuation_handoff=captured)
    return result


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    tasks=json.loads((PRIOR/'cases.json').read_text());AXES.update(json.loads((PRIOR/'axis_catalog.json').read_text()));loaded,hashes=models()
    paths=[prior.ROOT/'records.json',prior.ROOT/'protocol.json',AUDIT/'protocol.json',PRIOR/'cases.json',PRIOR/'axis_catalog.json',
        *sorted((prior.ROOT/'trials').glob('*_probe32.json')),*sorted((AUDIT/'snapshots').glob('*_probe32.json')),
        *sorted(Path(__file__).parent.glob('*.py'))]
    inputs={str(p):fingerprint(p) for p in paths}
    spec=dict(experiment='E97',inputs=inputs,checkpoints=hashes,test_evaluated=False,
        design='All 96 E95 checkpoint/task pairs x learned/analytical/exact continuation =288 complete tasks. Identical learned approach, privileged route descent and fixed32 physical probe steps. First actual close-phase predict is the switch, after all32 observed probe transitions update statistics. Only the78 route-assisted pairs switch;18 no-route pairs remain unchanged. All branches match full E96 post-probe physical state, controller statistics and history.',
        estimators='Learned: frozen MotionTransformer with neural solve guard. Analytical: existing np.linalg.solve(gram,cross).T each step. Exact: simulator geometric Jacobian each step, explicitly privileged and accessible only inside post-switch prediction. All modes retain recent32 measured updates, adaptive measured bias, same DLS action computation, geometry filter, contacts, phase budgets and success criteria. No reset or altered probe.',
        verification='Every approach/descent/probe trace equals E95; all96 learned full replays and36 no-route substitutions reproduce E95 exactly. All branch states and histories match E96 and each other. Inputs and checkpoints hash-verified. No weight training or reserved-test evaluation.',
        gate='Analytical continuation must strictly improve learned full-task success, not increase collisions, and retain all learned-successful six-joint cases. Also report exact continuation, every paired gain/loss and retention against E95 no-probe and matched-hold controls. This is a diagnostic adoption gate, not a learned-policy promotion.',
        limits='Repeated development tasks/checkpoints with privileged route planning. Post-switch future histories diverge. The exact branch isolates an estimator/control limitation but does not prove stale history or frozen-network mismatch is the sole cause. Analytical recovery is not a new neural-policy result.',
        videos='Prespecified case0 parent_1 and case2 candidate_0 (both E95 six-joint regressions) and case4 candidate_0 (difficult retained success), all three branches. If needed append first sorted missing success/failure category; append first analytical-versus-learned regression with all branches, disclosing outcome selection.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    rows=[];fallbacks=0
    for case,t in enumerate(tasks):
        for method,model in loaded.items():
            handoff=None
            for mode in ('learned','analytical','exact'):
                r=evaluate(case,t,method,model,mode)
                if handoff is None:handoff=r['continuation_handoff']
                assert handoff==r['continuation_handoff']
                if mode!='learned' and not r['route_used']:fallbacks+=1
                (ROOT/'trials'/f'{case}_{method}_{mode}.json').write_text(json.dumps(r))
                rows.append({k:v for k,v in r.items() if k not in ('trace','handoff','continuation_handoff')})
        print('case',case,'complete;',len(rows),'rollouts',flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2));(ROOT/'audit.json').write_text(json.dumps(dict(exact_replays=96,exact_fallbacks=fallbacks,identical_handoffs=True,unchanged_hashes=True,test_evaluated=False),indent=2));(ROOT/'status.json').write_text(json.dumps(dict(status='complete')))

if __name__=='__main__':run()
