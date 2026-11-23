"""Build a separate strict-object planner; preserve the E173 baseline source."""
from pathlib import Path

def run():
    s=Path(__file__).with_name('posture_route_planner.py').read_text();s=s[:s.index('\ndef run():')]
    s=s.replace('E173','E174').replace("ROOT=Path('outputs/ease_posture_route_v1')","ROOT=Path('outputs/ease_object_preserving_v1')").replace("if 'cube' in names and ('table' in names or 'gripper_left_pad' in names or 'gripper_right_pad' in names):continue","if names=={'cube','table'}:continue").replace('def plan(model,qpos,qvel,goal):','def plan_goal(model,qpos,qvel,goal):').replace('173000','174000')
    s+='''
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
'''
    Path(__file__).with_name('object_preserving_planner.py').write_text(s)

if __name__=='__main__':run()
