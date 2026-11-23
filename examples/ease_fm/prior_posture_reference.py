"""E152: matched analytical-prior reference for the fixed E151 posture rule."""
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import calibration_prior_control as prior
from . import home_posture_control as posture
from .arm_pick_place import PickPlace
from .abstaining_task_value_gate import read,save,fingerprint
ROOT=Path('outputs/ease_prior_posture_reference_v1')

def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('source','trials','states','actions','posture'):(ROOT/folder).mkdir()
    shutil.copy2(prior.ROOT/'cases.json',ROOT/'cases.json');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    save(ROOT/'protocol.json',dict(experiment='E152',scope='Additional matched analytical calibration-prior control for E151. Registered after partial E151 outcomes were observed; no E151 coefficient or model was changed. This control isolates learned transition weights from the analytical calibration-prior architecture. Same96 fitting placements; E120 and fresh set unchanged.',rule='Exactly the E151 damped projection N=I-J^T(JJ^T+.015^2I)^-1J times.05(home-q), uniformly scaled to maxabs<=.01rad, added after original non-probe action and then clipped+/-.04. Calibration and probe unchanged. Analytical J uses measured initial calibration prior plus32 recent transitions,.98recency,1e-6ridge.',gate='Even if the E151 primary gate passes, do not attribute a learned task improvement or advance that claim unless learned-posture success also exceeds this matched analytical-prior posture reference. Report paired losses. No parameter sweep.',inputs={str(posture.ROOT/'protocol.json'):fingerprint(posture.ROOT/'protocol.json'),str(Path(posture.__file__)):fingerprint(posture.__file__)}))
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in (ROOT/'protocol.json',ROOT/'cases.json',ROOT/'source'/Path(__file__).name)})

def run():
    episodes=[];core=prior.old.broad.previous.core;original_rollout=core.probe_rollout
    class PostureArm(PickPlace):
        def step(self,action,aperture):
            c=getattr(self,'posture_controller',None)
            if c is not None:
                original=np.asarray(action).copy();delta=np.zeros(self.n);j=None
                if self.phase!='probe':
                    j,_=c.predict();q=self.observe()['q'];projector=np.eye(self.n)-j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),j);delta=projector@(.05*(self.posture_home-q));delta*=min(1.,.01/max(float(np.max(np.abs(delta))),1e-12));action=np.clip(original+delta,-.04,.04);assert np.max(np.abs(action-original))<=.010000000001
                self.posture_logs.append(dict(phase=self.phase,base_action=original.tolist(),posture_delta=delta.tolist(),final_action=np.asarray(action).tolist(),task_space_delta=None if j is None else (j@delta).tolist()))
            return super().step(action,aperture)
    def rollout(env,home,c,**kwargs):
        env.posture_controller=c;env.posture_home=np.asarray(home);env.posture_logs=[];r=original_rollout(env,home,c,**kwargs);episodes.append(env.posture_logs);return r
    # The numerical intervention is exactly the same source block, checked before execution.
    a=Path(posture.__file__).read_text();b=Path(__file__).read_text();start="            c=getattr(self,'posture_controller',None)";end='            return super().step(action,aperture)';assert a[a.index(start):a.index(end)]==b[b.index(start):b.index(end)]
    with patch.object(prior,'ROOT',ROOT),patch.object(core,'PickPlace',PostureArm),patch.object(core,'probe_rollout',rollout):prior.run()
    rows=read(ROOT/'records.json');assert len(rows)==len(episodes)==96
    for r,log in zip(rows,episodes):save(ROOT/'posture'/f"case_{r['case']}.json",log);assert all(np.array_equal(a['base_action'],a['final_action']) for a in log if a['phase']=='probe')
    save(ROOT/'posture_verification.json',dict(exact_E151_intervention_source=True,probe_unchanged=True,all96_logs_saved=True));print('E152 complete',flush=True)

if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()
