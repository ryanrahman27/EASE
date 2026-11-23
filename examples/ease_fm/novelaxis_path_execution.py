"""E90: physically track frozen E89 paths, then finish the original task."""
import json
import shutil
from pathlib import Path
import numpy as np
from .novelaxis_paths import ROOT as PATHS, PRIOR
from .novelaxis_oracle import ExactPose, ROOT as ORACLE
from .geometry_filter import GeometryArm
from .arm_pick_place import rollout,targets
from .mixed_sim import AXES,Arm
from .pick_place_control import calibrate
from .benchmark import fingerprint

ROOT=Path('outputs/ease_novelaxis_path_execution_v1')


class PathArm(GeometryArm):
    def step(self,action,aperture):
        if getattr(self,'phase',None)=='descend' and self.route is not None:
            q=self.observe()['q'];waypoint=self.route[self.waypoint]
            if np.max(np.abs(waypoint-q))<.015 and self.waypoint<100:
                self.waypoint+=1;waypoint=self.route[self.waypoint]
            action=np.clip(waypoint-q+self.controller.bias,-.04,.04)
            self.path_log.append(dict(waypoint=self.waypoint,max_error=float(np.max(np.abs(waypoint-q)))))
        return super().step(action,aperture)


def evaluate(case,t,mode,record_step=None):
    p=json.loads((PATHS/'cases'/f'{case}.json').read_text())
    selected=next((a for a in p['attempts'] if a['attempt']==p['selected_attempt']),None)
    usable=selected is not None and selected['contact_free'] and selected['margin_2mm']
    env=PathArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();env.route=None;env.path_log=[];env.waypoint=0
    c=ExactPose(env);probe=calibrate(env,t['home'],c,872000+t['candidate'])
    assert probe==json.loads((PRIOR/f'probe_{case}.json').read_text())
    env.controller=c;env.filter_enabled=True;env.filter_log=[]
    if record_step is not None:
        original=env.step
        def captured(action,aperture):
            result=original(action,aperture);record_step(env);return result
        env.step=captured
    def phases():
        for name,goal,aperture,budget in targets(env.xy,env.target):
            env.phase=name
            if name=='descend':
                np.testing.assert_array_equal(env.data.qpos,np.array(p['state']['qpos']))
                np.testing.assert_array_equal(env.data.qvel,np.array(p['state']['qvel']))
                if usable and mode!='baseline':
                    start=env.observe()['q'];end=np.array(selected['endpoint'])
                    env.route=np.array([(1-a)*start+a*end for a in np.linspace(0,1,101)])
                    if mode=='route300':budget=300
            yield name,goal,aperture,budget
    result=rollout(env,t['home'],c,reset=False,sequence=phases())
    if mode=='baseline' or not usable:
        old=json.loads((ORACLE/'trials'/f'{case}_exact_jacobian.json').read_text());assert all(result[k]==old[k] for k in result)
    return dict(case=case,body=t['body'],family=t['arm']['family'],dof=env.n,mode=mode,route_available=usable,
        route_used=usable and mode!='baseline',last_waypoint=env.waypoint,path_log=env.path_log,
        filter_queries=sum(r['queries'] for r in env.filter_log),**result)


def run():
    ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    tasks=json.loads((PRIOR/'cases.json').read_text());AXES.update(json.loads((PRIOR/'axis_catalog.json').read_text()))
    paths=[PRIOR/'cases.json',PRIOR/'axis_catalog.json',PATHS/'protocol.json',PATHS/'records.json',*sorted((PATHS/'cases').glob('*.json')),*sorted((ORACLE/'trials').glob('*_exact_jacobian.json'))]
    hashes={str(p):fingerprint(p) for p in paths}
    spec=dict(experiment='E90',inputs=hashes,test_evaluated=False,
        design='All16 fixed cases x baseline exact-Jacobian / frozen route100 / frozen route300. Route intervention only where selected E89 route is sampled contact-free with2mm margin; other cases reproduce baseline with no budget extension. Start from original home, calibration and approach, assert exact descent-start qpos/qvel. Track101 joint waypoints, advance one waypoint when maxjoint error<.015rad. Command waypoint-q+measured adaptive bias clipped+/-.04; original geometry filter retained. Only descent action source changes. Original phase completion rule retained. Route300 additionally allows300 descent steps; all other budgets unchanged. Afterwards resume exact-Jacobian control through full original task.',
        limits='Privileged outcome-informed development diagnostic, no learned-policy improvement claim. Route endpoints and selection frozen from E89. No retuning on execution results. Record all failures; separate descent reach from full-task success. Longer-budget result is not budget-matched.',
        videos='Save physically executed full-task demos for cases0 and4 in each mode, label privileged controllers and budget. Verify separate replay equals saved trial. No static animation counted as executed motion.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('novelaxis_path_execution.py','novelaxis_oracle.py','geometry_filter.py','arm_pick_place.py','pick_place_control.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    rows=[]
    for case,t in enumerate(tasks):
        for mode in ('baseline','route100','route300'):
            result=evaluate(case,t,mode)
            (ROOT/'trials'/f'{case}_{mode}.json').write_text(json.dumps(result))
            rows.append({k:v for k,v in result.items() if k not in ('trace','path_log')})
        print(case,[(r['mode'],r['success'],r['collision'],r['last_waypoint']) for r in rows[-3:]],flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2));(ROOT/'audit.json').write_text(json.dumps(dict(exact_baselines=16,unchanged_no_route_replays=6,exact_descent_states=True,input_hashes_unchanged=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()
