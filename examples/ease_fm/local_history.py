"""E93: fixed 32-transition history after route descent, frozen estimators."""
from collections import deque
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import learned_path_transfer as previous
from .retention_fresh import GuardedPose
from .pick_place_control import ROT_SCALE
from .novelaxis_path_execution import PathArm
from .mixed_sim import AXES
from .benchmark import fingerprint

ROOT=Path('outputs/ease_local_history_v1')
PRIOR=previous.PRIOR
models=previous.models


class LocalPose(GuardedPose):
    def __init__(self,n,model):
        super().__init__(n,model);self.history=deque(maxlen=32);self.local=False
    def rebuild(self):
        gram=1e-6*np.eye(self.n);cross=np.zeros((self.n,6))
        for dq,dx in self.history:
            gram=.98*gram+np.outer(dq,dq)+2e-8*np.eye(self.n)
            cross=.98*cross+np.outer(dq,dx)
        self.gram=gram;self.cross=cross
    def update(self,before,after,command):
        dq=after['q']-before['q'];dx=np.r_[after['position']-before['position'],ROT_SCALE*(after['rotation'][:,2]-before['rotation'][:,2])]
        super().update(before,after,command);self.history.append((dq.copy(),dx.copy()))
        if self.local:self.rebuild()


def evaluate(case,t,method,model,mode,record_step=None):
    handoff={};active=[]
    class LocalArm(PathArm):
        def __setattr__(self,name,value):
            super().__setattr__(name,value)
            if name=='phase' and value=='close':
                c=self.controller
                handoff.update(qpos=self.data.qpos.copy().tolist(),qvel=self.data.qvel.copy().tolist(),gram=c.gram.tolist(),cross=c.cross.tolist(),bias=c.bias.tolist(),history_count=len(c.history))
                if mode=='window32' and self.route is not None:
                    c.local=True;c.rebuild();active.append(True)
                handoff.update(applied_gram=c.gram.tolist(),applied_cross=c.cross.tolist())
    with patch.object(previous,'GuardedPose',LocalPose),patch.object(previous,'PathArm',LocalArm):
        result=previous.evaluate(case,t,method,model,'route300',record_step)
    old=json.loads((previous.ROOT/'trials'/f'{case}_{method}_route300.json').read_text())
    prefix=lambda trace:[r for r in trace if r['phase'] in ('approach','descend')]
    assert prefix(result['trace'])==prefix(old['trace'])
    if mode=='long_history' or not active:assert result==old
    result.update(mode=mode,local_history_applied=bool(active),handoff=handoff)
    return result


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    tasks=json.loads((PRIOR/'cases.json').read_text());AXES.update(json.loads((PRIOR/'axis_catalog.json').read_text()));loaded,hashes=models()
    paths=[previous.ROOT/'protocol.json',previous.ROOT/'records.json',PRIOR/'cases.json',PRIOR/'axis_catalog.json',*sorted((previous.ROOT/'trials').glob('*_route300.json'))]
    inputs={str(p):fingerprint(p) for p in paths}
    spec=dict(experiment='E93',inputs=inputs,checkpoints=hashes,test_evaluated=False,
        design='All96 E91 checkpoint/case pairs, long-history route300 reference and fixed window32 intervention =192 full tasks. No window sweep or outcome tuning. Keep E91 route admission, route300 budget, tracker and frozen weights unchanged. Only after an accepted route descent, reconstruct gram/cross from last32 completed measured transitions, starting from1e-6I/zero and replaying original .98 forgetting and2e-8 diagonal floor. Rebuild after each subsequent observed transition. Keep original measured bias update and all phase rules. Unavailable-route cases retain long history exactly. No reset, extra probe or exact-Jacobian policy inputs.',
        verification='E91 long-history and unavailable-route results exact; identical approach/descent traces and pre-intervention handoff statistics. Task Jacobian access remains blocked by E91 evaluator and neural inverse guard retained. All weights/inputs unchanged.',
        gate='Advance only if pooled success improves without more collisions and every previously successful six-joint hybrid case is retained. Also report original E87 baseline retention and all gains/losses. Development diagnostic with privileged planner, not history-only or independent heldout generalization.',
        videos='Prespecified candidate_0 case0 and case4 in both modes. Append first missing success/failure category if needed. Verify executed replay equality.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('local_history.py','learned_path_transfer.py','novelaxis_path_execution.py','pick_place_control.py','geometry_filter.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    rows=[];fallbacks=0
    for case,t in enumerate(tasks):
        for method,model in loaded.items():
            handoff=None
            for mode in ('long_history','window32'):
                r=evaluate(case,t,method,model,mode)
                h={k:v for k,v in r['handoff'].items() if not k.startswith('applied_')}
                if handoff is None:handoff=h
                assert handoff==h
                if mode=='window32' and not r['local_history_applied']:fallbacks+=1
                (ROOT/'trials'/f'{case}_{method}_{mode}.json').write_text(json.dumps(r))
                rows.append({k:v for k,v in r.items() if k not in ('trace','path_log','descent_state','route_check','handoff')})
        print('case',case,'complete',flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2));(ROOT/'audit.json').write_text(json.dumps(dict(exact_replays=96,exact_fallbacks=fallbacks,identical_handoffs=True,unchanged_hashes=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()
