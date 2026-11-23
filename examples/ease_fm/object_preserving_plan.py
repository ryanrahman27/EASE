"""Grasp-first approach feasibility: choose grasp posture, reverse approach, connect."""
import numpy as np
import mujoco
from scipy.optimize import least_squares
from .object_preserving_planner import ROOT,REFERENCE,plan_goal,forbidden,geometry_ids,distances
from .abstaining_task_value_gate import read,save,fingerprint

def plan(model,qpos,qvel,unused_goal):
    data=mujoco.MjData(model);data.qpos[:]=qpos;data.qvel[:]=qvel;mujoco.mj_forward(model,data);obj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'object');site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');xy=data.xpos[obj,:2].copy();goal=np.r_[xy,.24];pre=np.r_[xy,.32];ids,table=geometry_ids(model,6,True)
    base=dict(available=False,route=None,dense_path=[],kind=None,start_qpos=np.asarray(qpos).tolist(),start_qvel=np.asarray(qvel).tolist(),goal=goal.tolist(),pick_goal=goal.tolist(),pregrasp_goal=pre.tolist(),object_xy_at_plan=xy.tolist(),vertical_attempts=[],privileged_static_only=True)
    initial_bad=forbidden(model,data)
    if initial_bad:
        base.update(failure='initial_robot_object_contact',initial_contacts=initial_bad);return base
    endpoints=plan_goal(model,qpos,qvel,goal);candidates=[x for x in endpoints['endpoint_attempts'] if x['accepted']];candidates.sort(key=lambda x:np.linalg.norm(np.array(x['q'])-qpos[:6]));base['endpoint_attempts']=endpoints['endpoint_attempts'];used=[]
    for candidate in candidates:
        grasp=np.array(candidate['q'])
        if any(np.linalg.norm(grasp-q)<.1 for q in used):continue
        used.append(grasp);q=grasp.copy();upward=[q.tolist()];good=True
        for z in np.linspace(.24,.32,17)[1:]:
            target=np.r_[xy,z]
            def error(qnew):
                data.qpos[:6]=qnew;mujoco.mj_forward(model,data);return np.r_[data.site_xpos[site]-target,.15*(data.site_xmat[site].reshape(3,3)[:,2]-[0,0,1])]
            solved=least_squares(error,q,bounds=(-2.35,2.35),max_nfev=220,ftol=1e-9,xtol=1e-9,gtol=1e-9);e=error(solved.x);good=bool(np.linalg.norm(e[:3])<.003 and np.linalg.norm(e[3:])/.15<.04)
            n=max(1,int(np.ceil(np.max(np.abs(solved.x-q))/.015)));samples=[q+(solved.x-q)*i/n for i in range(1,n+1)]
            for sample in samples:
                error(sample);good=good and bool(np.min(distances(model,data,ids,table))>=0 and not forbidden(model,data))
            if not good:break
            upward.extend(x.tolist() for x in samples);q=solved.x.copy()
        base['vertical_attempts'].append(dict(grasp=grasp.tolist(),vertical_feasible=good))
        if not good:continue
        connection=plan_goal(model,qpos,qvel,pre,fixed_endpoint=q)
        if not connection['available']:continue
        base.update(available=True,kind='strict_object_clear_grasp_first',dense_path=connection['dense_path']+upward[-2::-1],pregrasp_nodes=len(connection['dense_path']),route=[np.asarray(qpos[:6]).tolist(),q.tolist(),grasp.tolist()],connection=connection)
        return base
    base['failure']='no_connected_vertical_grasp';return base

def run():
    save(ROOT/'planning_protocol.json',dict(experiment='E174',stage='Before physical interventions',privileged=True,amendment='First pregrasp-first search failed all four static checks and is archived. Corrected search chooses collision-free grasp IK first, constructs a reverse vertical approach on that same IK branch, then connects to its pregrasp posture. No physical intervention outcomes used.',planning='24 fixed-seed grasp IK starts at current simulator object XY,z240mm, endpoint limits+/-2.35rad. Reverse16 Cartesian steps to320mm; <=15mrad dense approach sampling. Connect current posture to the selected pregrasp endpoint using original bounded geometric search. All robot/object contact forbidden in static planning; cube/table allowed. Existing initial object contact causes explicit no-route fallback, not silent exemption. Geometry/dynamics privilege and finite-sample limitations retained.',inputs={str(REFERENCE/'records.json'):fingerprint(REFERENCE/'records.json')}))
    rows=[]
    for row in read('outputs/ease_descent_stall_v1/diagnosis_60_local.json')['reference_cases']:
        if row['trigger'] is None:continue
        case=row['case'];stem=f'{case}_e120_ensemble_frozen_bias';l=read(REFERENCE/'clearance'/f'{stem}.json')[row['trigger']['step']];m=mujoco.MjModel.from_xml_path(str(REFERENCE/'states'/f'{stem}.xml'));r=plan(m,l['full_qpos_before'],l['full_qvel_before'],l['goal']);save(ROOT/'plans'/f'{case}.json',r);rows.append(dict(case=case,available=r['available'],nodes=len(r['dense_path']),failure=r.get('failure')));save(ROOT/'planning_summary.json',rows);print(rows[-1],flush=True)

if __name__=='__main__':run()
