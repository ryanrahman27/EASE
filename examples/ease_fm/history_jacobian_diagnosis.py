"""Privileged OFFLINE geometry audit; never supplies a rollout action."""
import shutil
from pathlib import Path
import numpy as np
import torch
import mujoco
from . import history_estimator_control as control
from .abstaining_task_value_gate import read,save,fingerprint,PHASES
from .learned_history_weights import token_features
from .arm_pick_place import ROT_SCALE,direction_jacobian,targets
ROOT=control.ROOT/'jacobian_diagnosis'

def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);nets=control.load_neural();folds=read(control.ROOT/'folds.json');choices=read(control.ROOT/'reference_choices.json');tasks=read(control.ROOT/'cases.json');rows=[]
    for worker in ('constant','learned'):
        data=np.load(control.ROOT/('diagnosis_'+worker)/'dataset.npz');controller='constant' if worker=='constant' else 'learned_ensemble'
        for case in np.unique(data['case']):
            ids=np.flatnonzero((data['case']==case)&(data['step']>0));q=torch.tensor(data['dq'][ids],device='cuda');x=torch.tensor(data['dx'][ids],device='cuda');mask=torch.tensor(data['mask'][ids],device='cuda');f=folds[str(case)];ch=choices[f];prior=data['prior'][ids];n=len(tasks[case]['home']);js={}
            with torch.no_grad():
                features=token_features(q,x,mask);js['learned']=torch.stack([m(q,x,mask,features)[0] for m in nets[f]]).mean(0).cpu().numpy();qq,xx=q.double(),x.double()
                for name,tau,ridge in [('constant',ch['forgetting'],ch['ridge']),('uniform',.98,1e-6),('prior',.98,1e-6)]:
                    w=tau**torch.arange(31,-1,-1,device='cuda',dtype=torch.float64);g=qq.transpose(1,2)@(qq*w[None,:,None])+ridge*torch.eye(7,device='cuda',dtype=torch.float64)[None];cross=qq.transpose(1,2)@(xx*w[None,:,None]);cross=cross+ridge*torch.tensor(prior,device='cuda') if name=='prior' else cross;js[name]=torch.linalg.solve(g,cross).cpu().numpy()
            stem=f'{case}_{controller}_frozen_bias';model=mujoco.MjModel.from_xml_path(str(control.ROOT/'states'/f'{stem}.xml'));d=mujoco.MjData(model);states=np.load(control.ROOT/'states'/f'{stem}.npz')['states'];site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');goals={name:goal for name,goal,_,_ in targets(tasks[case]['xy'],tasks[case]['target'])};trial=read(control.ROOT/'trials'/f'{stem}.json');bias=np.array(trial['initial_estimator']['bias'])
            for ix,row in enumerate(ids):
                step=int(data['step'][row]);phase=(*PHASES,'probe')[data['phase'][row]]
                if phase=='probe':continue
                state=states[step-1];d.qpos[:]=state[:model.nq];d.qvel[:]=state[model.nq:];mujoco.mj_forward(model,d);p=np.zeros((3,model.nv));r=np.zeros_like(p);mujoco.mj_jacSite(model,d,p,r,site);rot=d.site_xmat[site].reshape(3,3);truth=np.r_[p[:,:n],ROT_SCALE*direction_jacobian(rot,r[:,:n])];error=np.r_[goals[phase]-d.site_xpos[site],ROT_SCALE*(np.array([0,0,1])-rot[:,2])];action=lambda j:np.clip(j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*error)+bias,-.04,.04);oracle=action(truth);qq=data['dq'][row,:,:n].astype(float);eigs,vecs=np.linalg.eigh(qq.T@qq);weak=vecs[:,eigs<max(eigs[-1]*.01,1e-10)];metrics={}
                for name,arr in js.items():
                    j=arr[ix,:n].T;diff=j-truth;a=action(j);metrics[name]=dict(jacobian_mse=float(np.mean(diff**2)),weak_error_squared=float(np.sum((diff@weak)**2)),action_mse=float(np.mean((a-oracle)**2)),next_motion_mse=float(np.mean((data['query'][row,:n]@j.T-data['target'][row])**2)))
                rows.append(dict(case=int(case),controller=controller,dof=n,step=step,phase=phase,weak_directions=weak.shape[1],goal_error=float(np.linalg.norm(error)),metrics=metrics))
    save(ROOT/'rows.json',rows);summary={}
    for controller in ('constant','learned_ensemble'):
        rr=[r for r in rows if r['controller']==controller];summary[controller]={k:dict(jacobian_rmse=float(np.sqrt(np.mean([r['metrics'][k]['jacobian_mse'] for r in rr]))),action_rmse_rad=float(np.sqrt(np.mean([r['metrics'][k]['action_mse'] for r in rr]))),weak_error_squared=float(np.mean([r['metrics'][k]['weak_error_squared'] for r in rr]))) for k in ('learned','constant','uniform','prior')}
    save(ROOT/'summary.json',summary);save(ROOT/'protocol.json',dict(scope='Clearly privileged OFFLINE diagnostic only: replay saved pre-action fitting states to obtain exact MuJoCo position/tool-axis Jacobian. No privileged value enters controller, training or fresh tests. Compare full-Jacobian error and identical-state clipped DLS action against exact-Jacobian reference, not executed action quality or success. Everyfourth clean measured history on constant and learned E144 controllers; skip first state and fixed probes. Weak directions have history Gram eigenvalue<1%largest or1e-10. Error units mix position and scaled rotation; no causal success claim.',rows=len(rows)));shutil.copy2(__file__,ROOT/'source.py');print(summary,flush=True)

if __name__=='__main__':run()
