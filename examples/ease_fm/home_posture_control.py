"""E151: fixed observation-only home-posture objective, matched across estimators."""
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import history_estimator_control as old
from . import learned_calibration_control as learned
from . import exact_map_control as exact
from .arm_pick_place import PickPlace
from .abstaining_task_value_gate import read,save,fingerprint
ROOT=Path('outputs/ease_home_posture_control_v1')
WORKERS=('uniform32','e120_ensemble','learned_prior','privileged_exact_map')

def prepare():
    ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir();shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    save(ROOT/'protocol.json',dict(experiment='E151',motivation='E149 exact map still has35/48 six-joint successes and11 collisions; those collisions include arm links3/4 against the table during approach/descent/lift. Better motion-map accuracy alone does not choose a good posture. Test one fixed observation-only secondary posture objective before further estimator fitting.',change='After each original non-probe task command, compute N=I-J^T(JJ^T+.015^2 I)^-1 J using that controller current map. Add N*.05*(home-q), uniformly scaled so max absolute component<=.01rad, then original +/- .04 action clamp. Home is the original supplied safe starting posture, not an oracle future posture. Damped projection is approximate; log induced task-space motion and clipping. Fixed calibration and local probe actions are untouched. Frozen measured bias and all task waypoints/budgets remain unchanged.',comparisons='All96 fitting placements for uniform32, E120 three-model ensemble, E146 learned-prior three-model ensemble, and explicitly privileged exact map. Each receives identical posture rule; compare matched original no-posture results as well. No hyperparameter sweep, body-specific routing, task-value gate, physics selector, geometry lookup or new learned weights. Exact map remains reference only.',screen='A learned candidate only warrants fresh testing if it exceeds the matched E120 ensemble, is at least the matched uniform32 success count, does not decline on either joint count versus matched E120, has no more collision episodes, and retains fitting gain placements1,13,18. Report every lost success and distinguish controller improvement from learned-estimator improvement. E120 remains unchanged.'))
    for worker in WORKERS:
        root=ROOT/worker;root.mkdir()
        for folder in ('trials','states','actions','histories','source','posture'):(root/folder).mkdir()
        origin=learned.ROOT if worker=='learned_prior' else old.ROOT
        for name in ('cases.json','folds.json'):
            if (origin/name).exists():shutil.copy2(origin/name,root/name)
        if worker=='learned_prior':shutil.copytree(origin/'models',root/'models')
        if worker in ('uniform32','e120_ensemble'):shutil.copy2(old.ROOT/'reference_choices.json',root/'reference_choices.json')
        shutil.copy2(ROOT/'protocol.json',root/'protocol.json')
        for module in (old,learned,exact):shutil.copy2(Path(module.__file__),root/'source'/Path(module.__file__).name)
        save(root/'frozen.json',{str(p):fingerprint(p) for p in [root/'protocol.json',root/'cases.json',*(root/'models').rglob('*'),*(root/'source').glob('*')] if p.is_file()})
    save(ROOT/'frozen.json',{str(ROOT/p):fingerprint(ROOT/p) for p in ('protocol.json','source/home_posture_control.py')})

def run(worker):
    assert worker in WORKERS;root=ROOT/worker;episodes=[]
    class PostureArm(PickPlace):
        def step(self,action,aperture):
            c=getattr(self,'posture_controller',None)
            if c is not None:
                original=np.asarray(action).copy();delta=np.zeros(self.n);j=None
                if self.phase!='probe':
                    j,_=c.predict();q=self.observe()['q'];projector=np.eye(self.n)-j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),j);delta=projector@(.05*(self.posture_home-q));delta*=min(1.,.01/max(float(np.max(np.abs(delta))),1e-12));action=np.clip(original+delta,-.04,.04);assert np.max(np.abs(action-original))<=.010000000001
                self.posture_logs.append(dict(phase=self.phase,base_action=original.tolist(),posture_delta=delta.tolist(),final_action=np.asarray(action).tolist(),task_space_delta=None if j is None else (j@delta).tolist()))
            return super().step(action,aperture)
    def wrap(original):
        def rollout(env,home,c,**kwargs):
            env.posture_controller=c;env.posture_home=np.asarray(home);env.posture_logs=[];r=original(env,home,c,**kwargs);episodes.append(env.posture_logs);save(root/'posture'/f'{len(episodes)-1}.json',env.posture_logs);return r
        return rollout
    core=old.broad.previous.core
    if worker=='privileged_exact_map':
        with patch.object(exact,'ROOT',root),patch.object(exact,'PickPlace',PostureArm),patch.object(exact,'probe_rollout',wrap(exact.probe_rollout)):exact.run()
        path=root/'records.json'
    else:
        with patch.object(core,'PickPlace',PostureArm),patch.object(core,'probe_rollout',wrap(core.probe_rollout)):
            if worker=='learned_prior':
                with patch.object(learned,'ROOT',root):learned.run()
                path=root/'records.json'
            else:
                with patch.object(old,'ROOT',root):old.run(worker)
                path=root/f'records_{worker}.json'
    rows=read(path);assert len(rows)==len(episodes)==96
    for r,log in zip(rows,episodes):
        save(root/'posture'/f"case_{r['case']}.json",log);assert all(np.array_equal(a['base_action'],a['final_action']) for a in log if a['phase']=='probe')
    save(root/'records.json',rows)
    for p,h in read(ROOT/'frozen.json').items():assert fingerprint(p)==h
    save(root/'posture_verification.json',dict(episodes=96,probe_commands_unchanged=True,corrections_bounded=True,no_model_updates=True));print('E151',worker,sum(r['success'] for r in rows),'/96',flush=True)

def summarize():
    rows=sum([read(ROOT/w/'records.json') for w in WORKERS],[]);assert len(rows)==384;stats={};paired={}
    for method in WORKERS:
        rr=[r for r in rows if r['method']==method];stats[method]={str(n):dict(n=len(ss),success=sum(r['success'] for r in ss),collisions=sum(r['collision'] for r in ss)) for n in ('all',6,7) if (ss:=[r for r in rr if n=='all' or r['dof']==n])}
    candidate=[r for r in rows if r['method']=='learned_prior']
    for method in ('uniform32','e120_ensemble','privileged_exact_map'):
        pairs=[dict(case=r['case'],dof=r['dof'],before=next(v['success'] for v in rows if v['case']==r['case'] and v['method']==method),after=r['success']) for r in candidate];paired[method]=dict(gains=sum(r['after'] and not r['before'] for r in pairs),losses=sum(r['before'] and not r['after'] for r in pairs),changes=[r for r in pairs if r['before']!=r['after']])
    a,b=stats['learned_prior'],stats['e120_ensemble'];retained={str(c):next(r['success'] for r in candidate if r['case']==c) for c in (1,13,18)};qualified=a['all']['success']>b['all']['success'] and a['all']['success']>=stats['uniform32']['all']['success'] and all(a[str(n)]['success']>=b[str(n)]['success'] for n in (6,7)) and a['all']['collisions']<=b['all']['collisions'] and all(retained.values());save(ROOT/'records.json',rows);save(ROOT/'summary.json',dict(methods=stats,paired=paired,gain_retention=retained,qualifies_for_fresh_test=qualified,E120_unchanged=True,fresh_evaluated=False));print(read(ROOT/'summary.json'),flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='run':run(sys.argv[2])
    else:{'prepare':prepare,'summarize':summarize}[sys.argv[1]]()
