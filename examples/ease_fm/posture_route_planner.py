"""E173 privileged finite-kinematics posture/route feasibility, no physics advance."""
from pathlib import Path
import time
import numpy as np
import mujoco
from scipy.optimize import least_squares
from .descent_stall import geometry_ids,distances
from .abstaining_task_value_gate import read,save,fingerprint
ROOT=Path('outputs/ease_posture_route_v1')
REFERENCE=Path('outputs/ease_descent_execution_v1/history_allowance')

def forbidden(model,data):
    invalid=[]
    for c in data.contact[:data.ncon]:
        names={mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or str(int(g)) for g in (c.geom1,c.geom2)}
        if 'cube' in names and ('table' in names or 'gripper_left_pad' in names or 'gripper_right_pad' in names):continue
        invalid.append(sorted(names))
    return invalid

def plan(model,qpos,qvel,goal):
    started=time.monotonic();data=mujoco.MjData(model);data.qpos[:]=qpos;data.qvel[:]=qvel;site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');ids,table=geometry_ids(model,6,True);start=np.array(qpos[:6]);rng=np.random.default_rng(173000);queries=0
    def pose(q):
        nonlocal queries
        data.qpos[:6]=q;mujoco.mj_forward(model,data);queries+=1
        return np.r_[data.site_xpos[site]-goal,.15*(data.site_xmat[site].reshape(3,3)[:,2]-[0,0,1])]
    def valid(q):
        pose(q);return bool(np.min(distances(model,data,ids,table))>=0 and not forbidden(model,data))
    attempts=[];ends=[]
    for k in range(24):
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

def run():
    ROOT.mkdir(exist_ok=True);(ROOT/'plans').mkdir(exist_ok=True)
    protocol=dict(experiment='E173',stage='Static privileged feasibility before execution',planning='At the measured E172 late-local-stall gate, use a separate scratch state.24 fixed-seed bounded IK starts,220 evaluations each, joint endpoints+/-2.35rad, position<3mm, axis error<.04, link/palm/pad table clearance>=3mm and original forbidden contacts absent. Search straight joint routes then2000-iteration bidirectional RRT, step.15rad, discrete edge spacing at most.025rad. Shortcut deterministically. Start/path requires nonpenetration and original forbidden contacts absent. Static object/jaw state; no dynamics, no grasp success claim, no continuous safety certificate. No candidate outcome tuning. Failure to find a path is not infeasibility proof.',inputs={str(REFERENCE/'records.json'):fingerprint(REFERENCE/'records.json')})
    if not (ROOT/'planning_protocol.json').exists():save(ROOT/'planning_protocol.json',protocol)
    gate=read('outputs/ease_descent_stall_v1/diagnosis_60_local.json')['reference_cases'];summary=[]
    for row in gate:
        if row['trigger'] is None:continue
        case=row['case'];stem=f'{case}_e120_ensemble_frozen_bias';path=ROOT/'plans'/f'{case}.json'
        if not path.exists():
            logs=read(REFERENCE/'clearance'/f'{stem}.json');l=logs[row['trigger']['step']];m=mujoco.MjModel.from_xml_path(str(REFERENCE/'states'/f'{stem}.xml'));result=plan(m,l['full_qpos_before'],l['full_qvel_before'],np.array(l['goal']));result.update(case=case,trigger=row['trigger']);save(path,result)
        result=read(path);summary.append({k:result[k] for k in ('case','available','kind','distinct_accepted_endpoints','geometry_queries','planning_seconds')});save(ROOT/'planning_summary.json',summary);print(summary[-1],flush=True)

if __name__=='__main__':run()
