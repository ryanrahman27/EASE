"""E96: read-only, same-state estimator audit on every E95 rollout."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from . import local_probe as prior
from .local_history import LocalPose
from .arm_pick_place import ROT_SCALE, rotation_error
from .mixed_sim import AXES
from .benchmark import fingerprint

ROOT = Path('outputs/ease_probe_estimation_v1')


def estimate_metrics(jac, exact, history, error):
    dq = np.asarray([h[0] for h in history])
    dx = np.asarray([h[1] for h in history])
    action = np.clip(jac.T @ np.linalg.solve(jac @ jac.T + .015**2*np.eye(6), .35*error), -.04, .04)
    return dict(relative_error=float(np.linalg.norm(jac-exact)/max(np.linalg.norm(exact), 1e-12)),
        position_error=float(np.linalg.norm(jac[:3]-exact[:3])),
        direction_error=float(np.linalg.norm(jac[3:]-exact[3:])),
        history_relative_residual=float(np.linalg.norm(dq @ jac.T-dx)/max(np.linalg.norm(dx), 1e-12)),
        task_linear_residual=float(np.linalg.norm(.35*error-exact @ action)),
        joint_action=action.tolist())


def evaluate(case, task, method, model, mode):
    captured = {}; active = []
    original_jac = mujoco.mj_jacSite
    class AuditArm(prior.ProbeArm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs); active.append(self)
    class AuditPose(LocalPose):
        def predict(self):
            learned, bias = super().predict()
            env = active[0]
            if getattr(env, 'phase', None) == 'close' and not captured:
                state = dict(qpos=env.data.qpos.tolist(), qvel=env.data.qvel.tolist(), ctrl=env.data.ctrl.tolist(),
                    time=float(env.data.time), gram=self.gram.tolist(), cross=self.cross.tolist(), bias=self.bias.tolist())
                analytical = np.linalg.solve(self.gram, self.cross).T
                # This scoped read is never returned to the policy. The outer E95 guard resumes immediately.
                with patch.object(mujoco, 'mj_jacSite', original_jac):
                    exact = env.jacobian()
                obs=env.observe(); goal=np.r_[env.xy, .24]
                error=np.r_[goal-obs['position'], ROT_SCALE*rotation_error(np.eye(3),obs['rotation'])]
                history=[(dq.tolist(),dx.tolist()) for dq,dx in self.history]
                estimates={'learned':learned, 'analytical':analytical, 'exact':exact}
                captured.update(state=state, history=history, error=error.tolist(),
                    matrices={k:v.tolist() for k,v in estimates.items()},
                    metrics={k:estimate_metrics(v,exact,history,error) for k,v in estimates.items()})
                assert state == dict(qpos=env.data.qpos.tolist(), qvel=env.data.qvel.tolist(), ctrl=env.data.ctrl.tolist(),
                    time=float(env.data.time), gram=self.gram.tolist(), cross=self.cross.tolist(), bias=self.bias.tolist())
            return learned, bias
    with patch.object(prior,'ProbeArm',AuditArm), patch.object(prior,'LocalPose',AuditPose):
        result=prior.evaluate(case,task,method,model,mode)
    old=json.loads((prior.ROOT/'trials'/f'{case}_{method}_{mode}.json').read_text())
    assert result==old and captured
    return dict(case=case,method=method,mode=mode,dof=result['dof'],route_used=result['route_used'],
        success=result['success'],collision=result['collision'],exact_replay=True,**captured)


def run():
    torch.set_num_threads(4); ROOT.mkdir(exist_ok=False); (ROOT/'snapshots').mkdir(); (ROOT/'source').mkdir()
    tasks=json.loads((prior.PRIOR/'cases.json').read_text()); AXES.update(json.loads((prior.PRIOR/'axis_catalog.json').read_text()))
    loaded, hashes=prior.models()
    paths=[prior.ROOT/'records.json',prior.ROOT/'protocol.json',prior.PRIOR/'cases.json',prior.PRIOR/'axis_catalog.json',
        *sorted((prior.ROOT/'trials').glob('*.json')), *sorted(Path(__file__).parent.glob('*.py'))]
    inputs={str(p):fingerprint(p) for p in paths}
    spec=dict(experiment='E96',inputs=inputs,checkpoints=hashes,test_evaluated=False,
        design='All 288 E95 rollouts replay unchanged. At the first actual close-phase predict call, after all intervention observations update statistics, compare frozen learned, existing ridge analytical solve, and exact simulator Jacobian on the identical physical state and same measured history. Include all modes/checkpoints/tasks without outcome selection. Primary summaries use 234 route-assisted rollouts; 54 unchanged no-route rollouts are reported separately.',
        metrics='Relative Frobenius error, position and scaled tool-Z errors, in-history relative residual, and exact-Jacobian linearized residual of clipped bias-free DLS action for the current close target. Same damping .015, gain .35, clip .04. This task residual excludes measured bias, dynamics, geometry filtering and contact, and is not a rollout success prediction. Historical residual is in-sample, not held-out prediction.',
        safeguards='Exact Jacobian read only inside diagnostic scope, never fed to controller. Return original learned prediction. Assert state/statistics unchanged and entire E95 result/trace exact. No training, controller intervention, outcome threshold or probe tuning.',
        videos='Reuse all ten E95 success/failure videos via relative links after hash verification; no new behavior is generated. Report them as unchanged evidence.',
        decision='Assess whether analytical estimates benefit from probe data while learned estimates lag. Descriptive paired comparisons only; no causal claim from Frobenius error or correlated repeated tasks.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    rows=[]
    for case,t in enumerate(tasks):
        for method,model in loaded.items():
            for mode in ('no_probe','hold32','probe32'):
                r=evaluate(case,t,method,model,mode)
                (ROOT/'snapshots'/f'{case}_{method}_{mode}.json').write_text(json.dumps(r))
                rows.append({k:v for k,v in r.items() if k not in ('state','history','matrices','error')})
        print('case',case,'verified; snapshots',len(rows),flush=True)
    for p,h in {**inputs,**hashes}.items(): assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'audit.json').write_text(json.dumps(dict(exact_replays=len(rows),unchanged_hashes=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))

if __name__=='__main__':run()
