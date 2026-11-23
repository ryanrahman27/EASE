"""E149: explicitly privileged exact-kinematics diagnostic for the fixed task controller."""
import shutil
from pathlib import Path
import numpy as np
from . import history_estimator_control as old
from .abstaining_task_value_gate import read,save,fingerprint
from .arm_pick_place import PickPlace,targets
from .local_history import LocalPose
from .local_probe import probe_rollout,probe_offset
from .pick_place_control import calibrate
from .mixed_sim import Arm
ROOT=Path('outputs/ease_exact_map_control_v1')

def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('trials','states','actions','source'):(ROOT/folder).mkdir()
    shutil.copy2(old.ROOT/'cases.json',ROOT/'cases.json');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    save(ROOT/'protocol.json',dict(experiment='E149',scope='PRIVILEGED DIAGNOSTIC ONLY. Exact current MuJoCo tool-position/tool-up Jacobian replaces estimated motion map. Same measured32-step initial calibration, frozen measured bias, close probe, DLS, action limits, waypoints, phase budgets and success/collision criteria. No physics selector, geometry-based route or oracle task decisions. This is a reference for the fixed controller, not a mathematical upper bound on every possible task policy.',motivation='E144 offline exact-geometry audit shows learned maps improve full Jacobian and exact-Jacobian DLS action error as well as measured-next-motion error, yet complete tasks regress. Establish whether accurate kinematics alone improves this controller before spending further training on accuracy.',data='All96 fitting placements only; preserve calibration seeds38000+case. No held/fresh states. All outcomes retained. Exact-Jacobian access is confined to this explicitly privileged evaluator; no learned training data or deployed estimator uses it. E120 unchanged.',inputs={str(old.ROOT/p):fingerprint(old.ROOT/p) for p in ('cases.json','jacobian_diagnosis/summary.json')}))
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in (ROOT/'protocol.json',ROOT/'cases.json',ROOT/'source'/Path(__file__).name)})

def evaluate(case,t):
    active=False;states=[];logs=[]
    class Environment(PickPlace):
        def step(self,action,aperture):
            if getattr(self,'phase',None)=='probe':action=np.clip(self.anchor+probe_offset(self.probe_step,self.n)-self.observe()['q']+c.bias,-.04,.04);self.probe_step+=1
            before=self.observe();r=super().step(action,aperture)
            if active:states.append(np.r_[self.data.qpos.copy(),self.data.qvel.copy()]);logs.append(dict(phase=self.phase,q_before=before['q'].tolist(),nominal_action=action.tolist(),executed_command=r[1].tolist(),aperture=float(aperture),bad=r[2]['bad']))
            return r
    class Controller(LocalPose):
        def predict(self):return env.jacobian(),self.bias
        def update(self,before,after,command):
            bias=self.bias.copy();super().update(before,after,command)
            if active:self.bias=bias
    env=Environment(Arm(**t['arm']),t['xy'],t['target']);c=Controller(env.n,None);state=lambda:dict(gram=c.gram.tolist(),cross=c.cross.tolist(),bias=c.bias.tolist(),history_count=len(c.history));calibration=calibrate(env,t['home'],c,38000+case);initial=state();active=True
    def phases():
        for name,goal,aperture,budget in targets(env.xy,env.target):
            env.phase=name;env.goal=goal
            if name=='close':
                c.local=True;c.rebuild();env.anchor=env.observe()['q'].copy();env.probe_step=0;env.phase='probe';env.goal=env.observe()['position'].copy();yield 'probe',env.goal,0.,32;assert env.probe_step==32;env.phase=name;env.goal=goal
            yield name,goal,aperture,budget
    result=probe_rollout(env,t['home'],c,reset=False,sequence=phases());r=dict(case=case,body=t['arm']['seed'],family=t['arm']['family'],dof=env.n,method='privileged_exact_map',mode='frozen_bias',calibration=calibration,initial_estimator=initial,final_estimator=state(),**result);baseline=read(old.ROOT/'trials'/f'{case}_constant_frozen_bias.json');assert r['calibration']==baseline['calibration'] and initial==baseline['initial_estimator'];assert initial['bias']==r['final_estimator']['bias'];stem=f'{case}_privileged_exact_map_frozen_bias';save(ROOT/'trials'/f'{stem}.json',r);save(ROOT/'actions'/f'{stem}.json',logs);np.savez_compressed(ROOT/'states'/f'{stem}.npz',states=np.array(states));(ROOT/'states'/f'{stem}.xml').write_text(env.xml);assert len(states)==len(r['trace']);return {k:v for k,v in r.items() if k not in ('trace','calibration','initial_estimator','final_estimator')}

def run():
    _,_,hashes=old.broad.setup();path=ROOT/'records.json';rows=read(path) if path.exists() else [];done={r['case'] for r in rows}
    for case,t in enumerate(read(ROOT/'cases.json')):
        if t['validation'] or case in done:continue
        r=evaluate(case,t);rows.append(r);save(path,rows);print('E149',case,r['success'],flush=True)
    assert len(rows)==96
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h
    save(ROOT/'summary.json',{str(n):dict(n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr)) for n in ('all',6,7) if (rr:=[r for r in rows if n=='all' or r['dof']==n])});save(ROOT/'verification.json',dict(episodes=96,identical_calibration=True,identical_frozen_bias=True,source_and_E120_hashes_unchanged=True,privileged_reference_only=True));print(read(ROOT/'summary.json'),flush=True)

if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()
