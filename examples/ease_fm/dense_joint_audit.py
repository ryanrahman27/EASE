"""Verify measured branch labels and evaluate fixed family-excluded estimators."""
import shutil
import numpy as np
import torch
import mujoco
from .dense_joint_supervision import ROOT
from . import history_estimator_control as control
from . import calibration_prior_learning as learning
from .abstaining_task_value_gate import read,save,fingerprint,PHASES
from .arm_pick_place import ROT_SCALE,targets

def run():
    torch.set_num_threads(4);archive=np.load(ROOT/'dataset.npz');d={k:archive[k] for k in archive.files};rows=read(ROOT/'rows.json');tasks=read(ROOT/'cases.json');models={};oldmodels=control.load_neural();choices=read(control.ROOT/'reference_choices.json')
    for split in range(4):
        models[split]=[]
        for seed in learning.SEEDS:
            m=learning.PriorWeights().cuda().eval();m.load_state_dict(torch.load(learning.ROOT/'models'/f'fold_{split}'/f'{seed}.pt',weights_only=True));models[split].append(m)
    reconstructed=0;replayed=set();cached={}
    for idx,row in enumerate(rows):
        case=row['case'];n=row['dof'];phase=row['phase'];assert not tasks[case]['validation'];stem=f'{case}_constant_frozen_bias'
        if case not in cached:
            model=mujoco.MjModel.from_xml_path(str(control.ROOT/'states'/f'{stem}.xml'));history=np.load(control.ROOT/'histories'/f'{stem}.npz');cached[case]=(model,{k:history[k] for k in history.files},read(control.ROOT/'trials'/f'{stem}.json'))
        model,h,trial=cached[case];b=np.load(ROOT/'branches'/f'{case}_{phase}.npz');md=mujoco.MjData(model);before=b['before_state'];md.qpos[:]=before[:model.nq];md.qvel[:]=before[model.nq:];mujoco.mj_forward(model,md);site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');q0=md.qpos[:n].copy();p0=md.site_xpos[site].copy();z0=md.site_xmat[site].reshape(3,3)[:,2].copy();assert h['clean'][row['step']];candidates=np.flatnonzero(h['clean']&(h['phase']==PHASES.index(phase))&(np.arange(len(h['clean']))>=1));assert row['step']==candidates[0];np.testing.assert_array_equal(d['dq'][idx],h['dq_history'][row['step']]);np.testing.assert_array_equal(d['dx'][idx],h['dx_history'][row['step']]);key=(row['fold'],n);replay=key not in replayed
        for slot,after in enumerate(b['after_states']):
            md.qpos[:]=after[:model.nq];md.qvel[:]=after[model.nq:];mujoco.mj_forward(model,md);q=np.zeros(7,np.float32);q[:n]=md.qpos[:n]-q0;x=np.r_[md.site_xpos[site]-p0,ROT_SCALE*(md.site_xmat[site].reshape(3,3)[:,2]-z0)].astype(np.float32);np.testing.assert_array_equal(q,d['query'][idx,slot]);np.testing.assert_array_equal(x,d['target'][idx,slot]);assert d['query_mask'][idx,slot];reconstructed+=1
            if replay:
                mujoco.mj_resetData(model,md);md.qpos[:]=before[:model.nq];md.qvel[:]=before[model.nq:];md.ctrl[:]=b['previous_ctrl'];mujoco.mj_forward(model,md);md.qacc_warmstart[:]=0;command=np.array(trial['initial_estimator']['bias']);command[slot//2]+=(1 if slot%2 else -1)*.01;md.ctrl[:n]=np.clip(q0+np.clip(command,-.04,.04),-2.5,2.5);md.ctrl[n:]=next(ap for name,_,ap,_ in targets(tasks[case]['xy'],tasks[case]['target']) if name==phase)
                for _ in range(20):mujoco.mj_step(model,md)
                mujoco.mj_forward(model,md);np.testing.assert_array_equal(np.r_[md.qpos.copy(),md.qvel.copy()],after)
        replayed.add(key)
    v={k:torch.tensor(d[k],device='cuda') for k in ('dq','dx','mask','prior','query')};pred={k:np.zeros_like(d['target'],dtype=float) for k in ('uniform','constant','prior','E143','E146')}
    with torch.no_grad():
        f=learning.features(v['dq'],v['dx'],v['mask'],v['prior'])
        for split in range(4):
            ids=np.flatnonzero(d['fold']==split);q,x,mask,prior=[v[k][ids] for k in ('dq','dx','mask','prior')];js={'E146':torch.stack([m(q,x,mask,prior,f[ids])[0] for m in models[split]]).mean(0),'E143':torch.stack([m(q,x,mask,f[ids,:,:8])[0] for m in oldmodels[split]]).mean(0)};qq,xx=q.double(),x.double();ch=choices[split]
            for name,tau,ridge in [('uniform',.98,1e-6),('constant',ch['forgetting'],ch['ridge']),('prior',.98,1e-6)]:
                w=tau**torch.arange(31,-1,-1,device='cuda',dtype=torch.float64);g=qq.transpose(1,2)@(qq*w[None,:,None])+ridge*torch.eye(7,device='cuda',dtype=torch.float64)[None];cross=qq.transpose(1,2)@(xx*w[None,:,None]);js[name]=torch.linalg.solve(g,cross+ridge*prior if name=='prior' else cross)
            for name,j in js.items():pred[name][ids]=(v['query'][ids].double()@j).cpu().numpy()
    metrics={}
    for name,p in pred.items():
        metrics[name]={}
        for n in ('all',6,7):
            keep=d['query_mask']&((d['mask'].sum(1)==n)[:,None] if n!='all' else True);e=(p-d['target'])[keep];metrics[name][str(n)]=dict(position_rmse_mm=float(np.sqrt(np.mean(e[:,:3]**2))*1000),rotation_rmse_mm=float(np.sqrt(np.mean(e[:,3:]**2))*1000),queries=int(keep.sum()))
    np.savez_compressed(ROOT/'fixed_predictions.npz',**pred);save(ROOT/'prediction_summary.json',metrics)
    for p,h in {**read(ROOT/'frozen.json'),**read(ROOT/'input_hashes.json')}.items():assert fingerprint(p)==h
    save(ROOT/'verification.json',dict(all_labels_reconstructed_from_saved_states=reconstructed,independent_exact_branch_replays=len(replayed),all_fold_joint_count_cells_replayed=True,clean_phase_entry_selection=True,history_alignment=True,family_excluded_fixed_models=True,input_hashes_unchanged=True));shutil.copy2(__file__,ROOT/'source'/__file__.split('/')[-1]);print(metrics,flush=True)

if __name__=='__main__':run()
