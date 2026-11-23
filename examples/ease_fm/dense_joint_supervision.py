"""E148: measured directional probe labels at saved fitting states, training only."""
import shutil
from pathlib import Path
import numpy as np
import mujoco
from . import history_estimator_control as old
from .abstaining_task_value_gate import read,save,fingerprint,PHASES
from .arm_pick_place import ROT_SCALE,targets
ROOT=Path('outputs/ease_dense_joint_supervision_v1')

def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('source','branches'):(ROOT/folder).mkdir()
    shutil.copy2(old.ROOT/'cases.json',ROOT/'cases.json');shutil.copy2(old.ROOT/'folds.json',ROOT/'folds.json');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    save(ROOT/'protocol.json',dict(experiment='E148',scope='Training-only measured directional supervision, no new task controller or held/fresh outcome. Same96 E144 constant-controller fitting episodes and original four family splits. At most one state per task phase, the first clean pre-action state with a saved predecessor (step>=1); fixed local-probe phase excluded. No selection by later success, motion error or model disagreement.',probes='From each reconstructed state execute one40ms simulation step for each active joint with command frozen measured calibration bias plus or minus.01rad on that joint, clipped to +/- .04 and existing joint target limits. Other joints receive calibrated holding bias. Current phase gripper command fixed. Every branch starts from identical saved qpos/qvel and reconstructed preceding ctrl, then mj_forward and zero qacc_warmstart. Saved trajectories do not contain complete integrator state: these are explicitly reconstructed counterfactuals, not claims of exact original-episode continuation. Both signs and all joints retained, including contact branches.',labels='Actual observed joint displacement and tool position/up-vector displacement are labels. No exact Jacobian call, numerical Jacobian label, simulator parameter or future query enters network history features. The supervised query is the measured joint increment after the branch. Target is measured tool displacement. History and initial measured calibration prior are held fixed across all branches. Padded14-query slots with explicit active-query mask.',reason='One-next-motion supervision constrains only the executed direction, whereas inverse control queries the whole map. Joint-balanced branch measurements constrain additional directions while preserving deployment observations. Simulation branch actions are data collection only, never a deployment physics selector.',inputs={str(old.ROOT/p):fingerprint(old.ROOT/p) for p in ('records_constant.json','cases.json','folds.json')}))

def collect():
    from unittest.mock import patch
    tasks=read(ROOT/'cases.json');folds=read(ROOT/'folds.json');rows=[];arrays=[];hashes={};steps=0
    for case,t in enumerate(tasks):
        if t['validation']:continue
        stem=f'{case}_constant_frozen_bias';source=old.ROOT;paths=[source/'histories'/f'{stem}.npz',source/'states'/f'{stem}.npz',source/'states'/f'{stem}.xml',source/'actions'/f'{stem}.json',source/'trials'/f'{stem}.json']
        for p in paths:hashes[str(p)]=fingerprint(p)
        h=np.load(paths[0]);states=np.load(paths[1])['states'];m=mujoco.MjModel.from_xml_path(str(paths[2]));actions=read(paths[3]);trial=read(paths[4]);n=len(t['home']);bias=np.array(trial['initial_estimator']['bias']);prior=np.zeros((7,6));prior[:n]=np.linalg.solve(trial['initial_estimator']['gram'],trial['initial_estimator']['cross']);site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');apertures={name:ap for name,_,ap,_ in targets(t['xy'],t['target'])}
        for phase in range(len(PHASES)):
            candidates=np.flatnonzero(h['clean']&(h['phase']==phase)&(np.arange(len(states))>=1))
            if not len(candidates):continue
            i=int(candidates[0]);state=states[i-1];d=mujoco.MjData(m);d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];d.ctrl[:n]=np.clip(np.array(actions[i-1]['q_before'])+actions[i-1]['executed_command'],-2.5,2.5);d.ctrl[n:]=actions[i-1]['aperture'];mujoco.mj_forward(m,d);d.qacc_warmstart[:]=0;before_q=d.qpos[:n].copy();before_pos=d.site_xpos[site].copy();before_up=d.site_xmat[site].reshape(3,3)[:,2].copy();np.testing.assert_allclose(before_q,actions[i]['q_before'],atol=1e-12,rtol=0);base_ctrl=d.ctrl.copy();query=np.zeros((14,7),np.float32);target=np.zeros((14,6),np.float32);mask=np.zeros(14,bool);branch_states=[];contact_count=[]
            for joint in range(n):
                for sign in (-1,1):
                    slot=2*joint+(sign==1);mujoco.mj_resetData(m,d);d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];d.ctrl[:]=base_ctrl;mujoco.mj_forward(m,d);d.qacc_warmstart[:]=0;command=bias.copy();command[joint]+=sign*.01;command=np.clip(command,-.04,.04);d.ctrl[:n]=np.clip(before_q+command,-2.5,2.5);d.ctrl[n:]=apertures[PHASES[phase]];contacts=0
                    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Exact Jacobian forbidden during measured probe collection')):
                        for _ in range(20):mujoco.mj_step(m,d);contacts+=d.ncon
                    mujoco.mj_forward(m,d);query[slot,:n]=d.qpos[:n]-before_q;target[slot]=np.r_[d.site_xpos[site]-before_pos,ROT_SCALE*(d.site_xmat[site].reshape(3,3)[:,2]-before_up)];mask[slot]=True;branch_states.append(np.r_[d.qpos.copy(),d.qvel.copy()]);contact_count.append(contacts);steps+=1
            assert np.isfinite(query).all() and np.isfinite(target).all();np.savez_compressed(ROOT/'branches'/f'{case}_{PHASES[phase]}.npz',before_state=state,previous_ctrl=base_ctrl,after_states=np.array(branch_states),contact_substep_counts=contact_count,query=query,target=target,query_mask=mask);rows.append(dict(case=case,phase=PHASES[phase],step=i,fold=folds[str(case)],dof=n,queries=int(mask.sum())));arrays.append(dict(dq=h['dq_history'][i],dx=h['dx_history'][i],prior=prior,mask=np.arange(7)<n,query=query,target=target,query_mask=mask,case=case,fold=folds[str(case)],phase=phase))
        print('E148 collected',case,'contexts',len(rows),'branches',steps,flush=True)
    np.savez_compressed(ROOT/'dataset.npz',**{k:np.stack([r[k] for r in arrays]) for k in arrays[0]});save(ROOT/'rows.json',rows);save(ROOT/'input_hashes.json',hashes);save(ROOT/'summary.json',dict(contexts=len(rows),branches=steps,placements=len(set(r['case'] for r in rows)),full_task_episodes=0,training_only=True,fresh_evaluated=False));save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'dataset.npz',ROOT/'rows.json',ROOT/'cases.json',ROOT/'folds.json',ROOT/'source'/Path(__file__).name]});print(read(ROOT/'summary.json'),flush=True)

if __name__=='__main__':
    import sys
    {'prepare':prepare,'collect':collect}[sys.argv[1]]()
