"""E174 privileged finite-kinematics posture/route feasibility, no physics advance."""
from pathlib import Path
import time
import numpy as np
import mujoco
from scipy.optimize import least_squares
from .descent_stall import geometry_ids,distances
from .abstaining_task_value_gate import read,save,fingerprint
ROOT=Path('outputs/ease_object_preserving_v1')
REFERENCE=Path('outputs/ease_descent_execution_v1/history_allowance')

def forbidden(model,data):
    invalid=[]
    for c in data.contact[:data.ncon]:
        names={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or str(int(g)) for g in (c.geom1,c.geom2)}
        if names=={'cube','table'}:continue
        invalid.append(sorted(names))
    return invalid

def plan_goal(model,qpos,qvel,goal,fixed_endpoint=None):
    started=time.monotonic();data=mujoco.MjData(model);data.qpos[:]=qpos;data.qvel[:]=qvel;site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');ids,table=geometry_ids(model,6,True);start=np.array(qpos[:6]);rng=np.random.default_rng(174000);queries=0
    def pose(q):
        nonlocal queries
        data.qpos[:6]=q;mujoco.mj_forward(model,data);queries+=1
        return np.r_[data.site_xpos[site]-goal,.15*(data.site_xmat[site].reshape(3,3)[:,2]-[0,0,1])]
    def valid(q):
        pose(q);return bool(np.min(distances(model,data,ids,table))>=0 and not forbidden(model,data))
    attempts=[];ends=[] if fixed_endpoint is None else [np.array(fixed_endpoint).copy()]
    for k in range(24 if fixed_endpoint is None else 0):
        initial=np.clip(start,-2.35,2.35) if k==0 else rng.uniform(-2.2,2.2,6)
        result=least_squares(pose,initial,bounds=(-2.35,2.35),max_nfev=220,ftol=1e-9,xtol=1e-9,gtol=1e-9)
        error=pose(result.x);clear=distances(model,data,ids,table);bad=forbidden(model,data);accepted=bool(np.linalg.norm(error[:3])<.003 and np.linalg.norm(error[3:])/.15<.04 and np.min(clear)>=.003 and not bad)
        attempts.append(dict(attempt=k,position_error=float(np.linalg.norm(error[:3])),rotation_error=float(np.linalg.norm(error[3:])/.15),clearances=clear.tolist(),bad=bad,q=result.x.tolist(),accepted=accepted))
        if accepted and all(np.linalg.norm(result.x-q)>.1 for q in ends):ends.append(result.x.copy())
    ends.sort(key=lambda q:float(np.linalg.norm(q-start)));route=None;kind=None
    def edge(a,b):
        n=max(1,int(np.ceil(np.max(np.abs(b-a))/.025)))
        return all(valid(a+(b-a)*(i/n)) for i in range(n+1))
    if valid(start):
        for end in ends:
            if edge(start,end):route=[start,end];kind='straight_joint_path';break
        if route is None and ends:
            trees=[([start],[-1]),([q.copy() for q in ends],[-1]*len(ends))]
            def extend(tree,target):
                qs,parents=tree;near=int(np.argmin([np.linalg.norm(q-target) for q in qs]));delta=target-qs[near];norm=np.linalg.norm(delta)
                q=target.copy() if norm<=.15 else qs[near]+.15*delta/norm
                if not edge(qs[near],q):return None,False
                qs.append(q);parents.append(near);return len(qs)-1,norm<=.15
            def chain(tree,index):
                qs,parents=tree;result=[]
                while index!=-1:result.append(qs[index]);index=parents[index]
                return result[::-1]
            for iteration in range(2000):
                a=iteration%2;b=1-a;target=trees[b][0][rng.integers(len(trees[b][0]))] if rng.random()<.2 else rng.uniform(-2.4,2.4,6)
                ia,_=extend(trees[a],target)
                if ia is None:continue
                for _ in range(70):
                    ib,reached=extend(trees[b],trees[a][0][ia])
                    if ib is None:break
                    if reached:
                        first=chain(trees[a],ia);second=chain(trees[b],ib)
                        route=first+second[-2::-1] if a==0 else second+first[-2::-1];kind='rrt_connect';break
                if route is not None:break
    if route is not None:
        # Deterministic farthest-visible shortcut, then retain full dense samples.
        short=[route[0]];i=0
        while i<len(route)-1:
            j=len(route)-1
            while j>i+1 and not edge(route[i],route[j]):j-=1
            short.append(route[j]);i=j
        route=short;dense=[]
        for a,b in zip(route[:-1],route[1:]):
            n=max(1,int(np.ceil(np.max(np.abs(b-a))/.025)));dense.extend((a+(b-a)*t/n).tolist() for t in range(n))
        dense.append(route[-1].tolist());assert all(valid(np.array(q)) for q in dense)
        route=[q.tolist() for q in route]
    else:dense=[]
    return dict(available=route is not None,route=route,dense_path=dense,kind=kind,endpoint_attempts=attempts,distinct_accepted_endpoints=len(ends),geometry_queries=queries,planning_seconds=time.monotonic()-started,start_qpos=np.asarray(qpos).tolist(),start_qvel=np.asarray(qvel).tolist(),goal=np.asarray(goal).tolist(),privileged_static_only=True)

def plan(model,qpos,qvel,unused_goal):
    data=mujoco.MjData(model);data.qpos[:]=qpos;data.qvel[:]=qvel;mujoco.mj_forward(model,data)
    obj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'object');xy=data.xpos[obj,:2].copy();goal=np.r_[xy,.24];pre=np.r_[xy,.32]
    result=plan_goal(model,qpos,qvel,pre);result['pregrasp_goal']=pre.tolist();result['pick_goal']=goal.tolist();result['object_xy_at_plan']=xy.tolist();result['vertical_checks']=[]
    if not result['available']:return result
    q=np.array(result['dense_path'][-1]);ids,table=geometry_ids(model,6,True);site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');vertical=[]
    for z in np.linspace(.32,.24,17)[1:]:
        target=np.r_[xy,z]
        def error(candidate):
            data.qpos[:6]=candidate;mujoco.mj_forward(model,data)
            return np.r_[data.site_xpos[site]-target,.15*(data.site_xmat[site].reshape(3,3)[:,2]-[0,0,1])]
        solved=least_squares(error,q,bounds=(-2.35,2.35),max_nfev=220,ftol=1e-9,xtol=1e-9,gtol=1e-9);e=error(solved.x);good=bool(np.linalg.norm(e[:3])<.003 and np.linalg.norm(e[3:])/.15<.04)
        n=max(1,int(np.ceil(np.max(np.abs(solved.x-q))/.015)));samples=[q+(solved.x-q)*i/n for i in range(1,n+1)]
        for sample in samples:
            error(sample)
            good=good and bool(np.min(distances(model,data,ids,table))>=0 and not forbidden(model,data))
        result['vertical_checks'].append(dict(target=target.tolist(),accepted=good,endpoint=solved.x.tolist()))
        if not good:
            result['available']=False;result['failure']='vertical_approach_not_found';result['dense_path']=[];return result
        vertical.extend(x.tolist() for x in samples);q=solved.x.copy()
    result['pregrasp_nodes']=len(result['dense_path']);result['dense_path']+=vertical;result['goal']=goal.tolist();result['kind']='object_clear_pregrasp_then_vertical';result['route']=[result['dense_path'][0],result['dense_path'][result['pregrasp_nodes']-1],result['dense_path'][-1]]
    return result

def run():
    ROOT.mkdir(exist_ok=True);(ROOT/'plans').mkdir(exist_ok=True);summary=[]
    save(ROOT/'planning_protocol.json',dict(experiment='E174',privileged=True,plan='Current simulator object XY; pregrasp height320mm, final240mm. Strictly reject ALL robot/object contacts while planning; cube/table allowed.24 fixed-seed IK starts; original bounded RRT fallback. From pregrasp,16 Cartesian vertical waypoints at5mm spacing, warm-start IK; dense joint checks<=15mrad. Original E170 link/palm/table geometry protection remains. No live physics advance. A geometric path does not guarantee dynamic preservation.',inputs={str(REFERENCE/'records.json'):fingerprint(REFERENCE/'records.json')}))
    for row in read('outputs/ease_descent_stall_v1/diagnosis_60_local.json')['reference_cases']:
        if row['trigger'] is None:continue
        case=row['case'];stem=f'{case}_e120_ensemble_frozen_bias';l=read(REFERENCE/'clearance'/f'{stem}.json')[row['trigger']['step']];m=mujoco.MjModel.from_xml_path(str(REFERENCE/'states'/f'{stem}.xml'));p=ROOT/'plans'/f'{case}.json'
        if not p.exists():save(p,plan(m,l['full_qpos_before'],l['full_qvel_before'],l['goal']))
        r=read(p);summary.append(dict(case=case,available=r['available'],kind=r['kind'],nodes=len(r['dense_path']),failure=r.get('failure'),pregrasp=r['pregrasp_goal'],pick_goal=r['pick_goal']));save(ROOT/'planning_summary.json',summary);print(summary[-1],flush=True)

if __name__=='__main__':run()
