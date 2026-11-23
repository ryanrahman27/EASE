"""E141: learned transition weights inside a measured-history least-squares model."""
import shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .expanded_structured_pairs import ROOT as DATA_ROOT,COLLECT
from .abstaining_task_value_gate import read,save,fingerprint,DATA
from .arm_pick_place import ROT_SCALE

ROOT=Path('outputs/ease_learned_history_weights_v1')
SEEDS=(141100,141101,141102)

def uniform(dq,dx):
    w=.98**torch.arange(dq.shape[1]-1,-1,-1,device=dq.device,dtype=dq.dtype)
    g=dq.transpose(1,2)@(dq*w[None,:,None])+1e-6*torch.eye(7,device=dq.device)[None]
    c=dq.transpose(1,2)@(dx*w[None,:,None]);j=torch.linalg.solve(g,c)
    return j,g,w

def token_features(dq,dx,mask):
    j,g,w=uniform(dq,dx);res=dx-dq@j;distance=torch.flip(torch.cumsum(torch.flip(dq,[1]),dim=1),[1])-.5*dq;n=mask.sum(1).clamp_min(1).sqrt();leverage=torch.sum(dq*torch.linalg.solve(g,dq.transpose(1,2)).transpose(1,2),-1)
    norm=lambda z:torch.linalg.vector_norm(z,dim=-1)
    age=torch.linspace(1,0,dq.shape[1],device=dq.device)[None].expand(len(dq),-1)
    f=torch.stack((torch.log1p(norm(dq)/.005),torch.log1p(norm(dx[:,:,:3])/.001),torch.log1p(norm(dx[:,:,3:])/.001),(norm(distance)/n[:,None]/.05).clamp_max(10),age,torch.log1p(norm(res[:,:,:3])/.0001),torch.log1p(norm(res[:,:,3:])/.0001),leverage),-1)
    return f

class HistoryWeights(nn.Module):
    def __init__(self):
        super().__init__();self.input=nn.Linear(8,32);layer=nn.TransformerEncoderLayer(32,4,64,dropout=0.,activation='gelu',batch_first=True,norm_first=True);self.encoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False);self.output=nn.Linear(32,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias);self.ridge_log=nn.Parameter(torch.zeros(()))
    def forward(self,dq,dx,mask,features=None):
        f=token_features(dq,dx,mask) if features is None else features;logw=np.log(5)*torch.tanh(self.output(self.encoder(self.input(f))).squeeze(-1));w=logw.exp()*(.98**torch.arange(dq.shape[1]-1,-1,-1,device=dq.device,dtype=dq.dtype))[None];logridge=np.log(10)*torch.tanh(self.ridge_log);ridge=1e-6*logridge.exp();g=dq.transpose(1,2)@(dq*w[:,:,None])+ridge*torch.eye(7,device=dq.device)[None];c=dq.transpose(1,2)@(dx*w[:,:,None]);j=torch.linalg.solve(g,c)*mask[:,:,None];return j,logw,logridge

def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('models','source'):(ROOT/folder).mkdir()
    inputs=[DATA_ROOT/'dataset.npz',DATA_ROOT/'rows.json',DATA_ROOT/'cases.json',DATA_ROOT/'fresh_cases.json',Path('outputs/ease_paired_effect_classifier_v1/summary.json')]
    save(ROOT/'protocol.json',dict(experiment='E141',inputs={str(p):fingerprint(p) for p in inputs},
        design='Direct motion-estimator study after unsuccessful terminal-task gates. Learn transition weights inside an explicit least-squares solve; clearly a learned/analytical hybrid, not the original inverse-free MotionTransformer or flow policy. Same2208 clean phase-entry histories, same four fitting-family folds; next observed joint motion and tool motion are supervision only. No final success labels in loss, no privileged Jacobian, no body IDs or object/contact features.',
        observations='Last32 measured dq/dx transitions, padded7 joints; dx is tool XYZ increment plus ROT_SCALE times tool-up-vector increment. Weight network sees only eight scalar per-transition features: log dq norm, log position-motion norm, log rotation-motion norm, joint midpoint distance to current pose divided by sqrt(dof), age, log uniform-model position/rotation residual norms, leverage. All available from past history. No future action/motion enters the weight network.',
        architecture='8->32 embedding; two32-wide4-head Transformer encoder layers, feedforward64,GELU,dropout0; scalar transition log-weight head initializedzero. Weights exp(log5*tanh(head)) multiply fixed.98 recency. Ridge1e-6*exp(log10*tanh(parameter)), initialized1e-6. Solve weighted normal equations explicitly. Initial estimator exactly32-transition .98-weighted ridge. Learned weights shared across all joints and both motion modalities.',
        training='Four family folds x3 seeds141100..141102;1500 AdamW steps,lr3e-4,weight_decay1e-4,batch128 starts, both on/off one-step measured transitions per start, clipgrad1. Loss normalized measured forward-motion MSE plus.001 mean log-weight squared plus.001 log-ridge squared. Group motion scales (XYZ/rotation) from training rows only, RMS floor1e-5. No success labels, checkpoint/seed selection or teacher-Jacobian target.',
        evaluation='Compare out-of-fold next-motion predictions with uniform32 history and original analytical Gram/cross. Report position/rotation errors separately, per seed and joint count. Histories and paired queries are dependent, not independent episode counts. No task rollout or fresh/held test in this prediction stage.',
        advancement='Only consider full-task control if pooled position MSE improves at least10% over uniform32, each seed is no worse, each joint-count pooled MSE is no worse, and pooled rotation MSE is <=1.05x uniform32. Otherwise diagnose before controller rollout. E120 immutable.'))
    shutil.copy2(DATA_ROOT/'cases.json',ROOT/'cases.json');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)

def extract():
    import mujoco
    meta=read(DATA_ROOT/'rows.json');base=np.load(DATA_ROOT/'dataset.npz');hqs=[];hxs=[];masks=[];queries=[];targets=[];analytic=[];hashes={};cache={}
    def get(p):hashes[str(p)]=fingerprint(p);return read(p)
    for ix,r in enumerate(meta):
        case,seed,phase=r['case'],r['seed'],r['phase'];source=DATA/'collection' if case<56 else COLLECT;key=(case,seed)
        if key not in cache:
            stem=f'{case}_on_{seed}_frozen_bias';xml=source/'states'/f'{stem}.xml';m=mujoco.MjModel.from_xml_path(str(xml.resolve()));state=mujoco.MjData(m);site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');npz=source/'states'/f'{stem}.npz';hashes[str(xml)]=fingerprint(xml);hashes[str(npz)]=fingerprint(npz);cache[key]=(m,state,site,np.load(npz)['states'],get(source/'branches'/f'{case}_on_{seed}.json'))
        m,state,site,onstates,branch=cache[key];i=branch['entries'][phase]['step'];sp=source/'snapshots'/f'{case}_on_{seed}_{phase}.npz';s=np.load(sp);hashes[str(sp)]=fingerprint(sp);dq,dx=s['history_dq'],s['history_dx'];assert dq.shape[0]==32 and dx.shape==(32,6);n=dq.shape[1];hq=np.zeros((32,7));hq[:,:n]=dq;mask=np.arange(7)<n
        mujoco.mj_setState(m,state,s['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,state);q0=state.qpos[:n].copy();p0=state.site_xpos[site].copy();up0=state.site_xmat[site].reshape(3,3)[:,2].copy();op=source/'states'/f'{case}_off_{phase}_{seed}_frozen_bias.npz';hashes[str(op)]=fingerprint(op);offstate=np.load(op)['states'][i];qs=[];ys=[]
        for value in (onstates[i],offstate):
            state.qpos[:]=value[:m.nq];state.qvel[:]=value[m.nq:];mujoco.mj_forward(m,state);q=np.zeros(7);q[:n]=state.qpos[:n]-q0;qs.append(q);ys.append(np.r_[state.site_xpos[site]-p0,ROT_SCALE*(state.site_xmat[site].reshape(3,3)[:,2]-up0)])
        ja=np.zeros((7,6));ja[:n]=np.linalg.solve(s['gram'],s['cross']);hqs.append(hq);hxs.append(dx);masks.append(mask);queries.append(qs);targets.append(ys);analytic.append(ja)
        if ix%250==0:print('E141 extracted',ix,'histories',flush=True)
    np.savez_compressed(ROOT/'dataset.npz',dq=np.asarray(hqs,np.float32),dx=np.asarray(hxs,np.float32),mask=masks,query=np.asarray(queries,np.float32),target=np.asarray(targets,np.float32),analytical=np.asarray(analytic,np.float32),fold=base['fold'],family=base['family']);save(ROOT/'input_hashes.json',hashes);save(ROOT/'rows.json',meta);print('E141 extracted',len(meta),'histories and',2*len(meta),'paired next-motion labels',flush=True)

def train():
    torch.set_num_threads(4);d=np.load(ROOT/'dataset.npz');v={k:torch.tensor(d[k],device='cuda') for k in ('dq','dx','mask','query','target')};fold=d['fold'];family=d['family'];n=len(fold)
    with torch.no_grad():f=token_features(v['dq'],v['dx'],v['mask']);u=uniform(v['dq'],v['dx'])[0];up=v['query']@u
    np.savez_compressed(ROOT/'references.npz',uniform=up.cpu().numpy(),analytical=np.asarray(d['query']@d['analytical']))
    pred=np.empty((3,n,2,6),np.float32);audit=[]
    for split in range(4):
        ids=np.flatnonzero(fold!=split);va=np.flatnonzero(fold==split);assert not set(family[ids])&set(family[va]);target=d['target'][ids];scale=np.r_[np.full(3,max(float(np.sqrt(np.mean(target[:,:,:3]**2))),1e-5)),np.full(3,max(float(np.sqrt(np.mean(target[:,:,3:]**2))),1e-5))].astype(np.float32);directory=ROOT/'models'/f'fold_{split}';directory.mkdir();np.savez(directory/'normalization.npz',scale=scale,training=ids);sc=torch.tensor(scale,device='cuda')
        for si,seed in enumerate(SEEDS):
            torch.manual_seed(seed);rng=np.random.default_rng(seed);m=HistoryWeights().cuda();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);losses=[]
            for step in range(1500):
                idx=rng.choice(ids,128,replace=True);j,logw,logridge=m(v['dq'][idx],v['dx'][idx],v['mask'][idx],f[idx]);response=v['query'][idx]@j;mse=torch.mean(((response-v['target'][idx])/sc)**2);loss=mse+.001*torch.mean(logw**2)+.001*logridge**2;opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
                if step%100==0:losses.append(dict(step=step,loss=float(loss.detach()),mse=float(mse.detach())))
            m.eval();torch.save(m.state_dict(),directory/f'{seed}.pt')
            with torch.no_grad():
                for start in range(0,len(va),256):
                    ii=va[start:start+256];j,_,_=m(v['dq'][ii],v['dx'][ii],v['mask'][ii],f[ii]);pred[si,ii]=(v['query'][ii]@j).cpu().numpy()
            audit.append(dict(fold=split,seed=seed,losses=losses));print('E141 trained',split,seed,flush=True)
    np.savez_compressed(ROOT/'oof.npz',prediction=pred,fold=fold);save(ROOT/'training.json',audit);summarize()
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'dataset.npz',ROOT/'oof.npz',ROOT/'references.npz',ROOT/'training.json',*(ROOT/'models').rglob('*'),*(ROOT/'source').glob('*.py')] if p.is_file()})

def summarize():
    d=np.load(ROOT/'dataset.npz');o=np.load(ROOT/'oof.npz')['prediction'];ref=np.load(ROOT/'references.npz');truth=d['target'];dof=d['mask'].sum(1);metrics={}
    for name,p in [('uniform32',ref['uniform'][None]),('analytical',ref['analytical'][None]),('learned',o)]:
        metrics[name]={}
        for count in ('all',6,7):
            ids=np.ones(len(dof),bool) if count=='all' else dof==count;e=p[:,ids]-truth[None,ids];metrics[name][str(count)]=dict(position_rmse_mm=float(np.sqrt(np.mean(e[:,:,:,:3]**2))*1000),rotation_rmse_mm=float(np.sqrt(np.mean(e[:,:,:,3:]**2))*1000),position_mse_by_seed=np.mean(e[:,:,:,:3]**2,axis=(1,2,3)).tolist())
    a,b=metrics['learned'],metrics['uniform32'];ratio=(a['all']['position_rmse_mm']/b['all']['position_rmse_mm'])**2;advance=ratio<=.9 and all(x<=b['all']['position_mse_by_seed'][0] for x in a['all']['position_mse_by_seed']) and all(a[str(k)]['position_rmse_mm']<=b[str(k)]['position_rmse_mm'] for k in (6,7)) and a['all']['rotation_rmse_mm']**2<=1.05*b['all']['rotation_rmse_mm']**2
    summary=dict(metrics=metrics,position_mse_ratio=ratio,qualifies_for_control=bool(advance),task_rollouts=0,fresh_evaluated=False,E120_unchanged=True);save(ROOT/'summary.json',summary);print(summary,flush=True)

if __name__=='__main__':
    import sys
    {'prepare':prepare,'extract':extract,'train':train,'summarize':summarize}[sys.argv[1]]()
