"""E89: bounded, privileged static descent-path feasibility diagnostic."""
import json
import shutil
from pathlib import Path
import numpy as np
import mujoco
from .novelaxis_oracle import ExactPose, PRIOR, ROOT as ORACLE
from .geometry_filter import GeometryArm
from .arm_pick_place import rollout, targets, rotation_error, ROT_SCALE
from .mixed_sim import AXES,Arm
from .pick_place_control import calibrate
from .benchmark import fingerprint

ROOT=Path('outputs/ease_novelaxis_paths_v1')


def solve_endpoint(env,goal,start):
    """IK changes arm q only; fingers/object retain recorded descent-start pose."""
    q=np.array(start,dtype=float)
    for _ in range(220):
        env.data.qpos[:env.n]=q;mujoco.mj_forward(env.model,env.data)
        obs=env.observe();error=np.r_[goal-obs['position'],ROT_SCALE*rotation_error(np.eye(3),obs['rotation'])]
        if np.linalg.norm(error[:3])<.001 and np.linalg.norm(error[3:])<.003:
            return q.copy() if not env.contacts()[1] else None
        jac=env.jacobian();delta=jac.T@np.linalg.solve(jac@jac.T+.005**2*np.eye(6),error)
        q=np.clip(q+np.clip(delta,-.15,.15),-2.4,2.4)
    return None


def sample_path(env,start,end):
    samples=[]
    for alpha in np.linspace(0,1,101):
        q=(1-alpha)*start+alpha*end;env.data.qpos[:env.n]=q;mujoco.mj_forward(env.model,env.data)
        bad=env.contacts()[1]
        clearance=env.clearance(q,env.data.qpos[env.n:env.n+2])
        samples.append(dict(alpha=float(alpha),clearance=clearance,bad=bad))
    return dict(samples=samples,minimum_clearance=min(s['clearance'] for s in samples),
        contact_free=all(not s['bad'] for s in samples),margin_2mm=all(s['clearance']>=.002-1e-6 for s in samples))


def run():
    ROOT.mkdir(exist_ok=False);(ROOT/'cases').mkdir();(ROOT/'source').mkdir()
    inputs={str(p):fingerprint(p) for p in (PRIOR/'cases.json',PRIOR/'axis_catalog.json',ORACLE/'records.json')}
    tasks=json.loads((PRIOR/'cases.json').read_text());AXES.update(json.loads((PRIOR/'axis_catalog.json').read_text()))
    spec=dict(experiment='E89',inputs=inputs,test_evaluated=False,
        design='All16 E87 cases. Replay E88 exact-Jacobian controller to completion and save full state immediately before descent. Verify full replay against saved E88 trace. On a separate model preserve object/finger pose and solve descent endpoint from20 starts: recorded arm posture then19 fixed uniform[-1.8,1.8] random starts, seed89000+case. Each solve uses220 iterations, unchanged IK damping .005, stepclip .15, jointlimits +/-2.4, XYZ1mm/tool-axis scaled.003 tolerances. Accept contact-free endpoints only; no early stopping across starts.',
        paths='For every accepted endpoint evaluate101 equally spaced joint-interpolation samples from recorded descent start. Report physical forbidden-contact freedom and separate2mm expanded-geometry clearance. No physics integration, object motion, velocity limits, grasping or whole-task execution. Choose highest minimum clearance endpoint; ties shortest joint displacement then earliest attempt. Failure to find a route is not proof of infeasibility; finite samples do not certify continuous safety.',
        videos='Illustrative static-kinematic path animations selected deterministically: first contact-free route and first blocked route among difficult-family cases, with all-case fallback if absent. Explicit labels distinguish kinematic samples from executed motion. Save source states, endpoint vectors, checks and video hashes.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('novelaxis_paths.py','novelaxis_oracle.py','geometry_filter.py','arm_pick_place.py','pick_place_control.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    rows=[]
    for case,t in enumerate(tasks):
        ref=ORACLE/'trials'/f'{case}_exact_jacobian.json';inputs[str(ref)]=fingerprint(ref)
        env=GeometryArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();c=ExactPose(env)
        probe=calibrate(env,t['home'],c,872000+t['candidate']);assert probe==json.loads((PRIOR/f'probe_{case}.json').read_text())
        env.controller=c;env.filter_enabled=True;env.filter_log=[];saved={}
        def phases():
            for item in targets(env.xy,env.target):
                if item[0]=='descend':
                    saved.update(qpos=env.data.qpos.copy().tolist(),qvel=env.data.qvel.copy().tolist(),goal=item[1].tolist())
                yield item
        result=rollout(env,t['home'],c,reset=False,sequence=phases());old=json.loads(ref.read_text());assert all(result[k]==old[k] for k in result)
        search=GeometryArm(Arm(**t['arm']),t['xy'],t['target']);search.prepare_geometry()
        start=np.array(saved['qpos'][:env.n]);rng=np.random.default_rng(89000+case);attempts=[]
        for attempt in range(20):
            search.data.qpos[:]=saved['qpos'];search.data.qvel[:]=0;mujoco.mj_forward(search.model,search.data)
            q=solve_endpoint(search,np.array(saved['goal']),start if attempt==0 else rng.uniform(-1.8,1.8,env.n))
            record=dict(attempt=attempt,endpoint_found=q is not None)
            if q is not None:record.update(endpoint=q.tolist(),distance=float(np.linalg.norm(q-start)),**sample_path(search,start,q))
            attempts.append(record)
        valid=[r for r in attempts if r['endpoint_found']]
        best=max(valid,key=lambda r:(r['minimum_clearance'],-r['distance'],-r['attempt'])) if valid else None
        row=dict(case=case,body=t['body'],family=t['arm']['family'],dof=env.n,oracle_success=result['success'],oracle_collision=result['collision'],
            endpoints=len(valid),contact_free_paths=sum(r['contact_free'] for r in valid),margin_paths=sum(r['margin_2mm'] for r in valid),selected_attempt=None if best is None else best['attempt'])
        (ROOT/'cases'/f'{case}.json').write_text(json.dumps(dict(**row,state=saved,attempts=attempts),indent=2));rows.append(row)
        print(json.dumps(row),flush=True)
    for p,h in inputs.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2));(ROOT/'audit.json').write_text(json.dumps(dict(exact_replays=16,inputs=inputs,hashes_unchanged=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()
