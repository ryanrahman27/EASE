"""Unchanged E144 estimator logic without fitting-case archive assertions."""
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .history_estimator_control import broad,history,token_features,PHASES,ROT_SCALE,save
ROOT=Path('outputs/ease_unseen_estimator_v1')

def evaluate(case,t,method,models,learned,choice):
    previous=broad.previous;core=previous.core;original_predict=core.LocalPose.predict;original_rollout=core.probe_rollout;cache={};logs=[];contexts=[];correction_ids={id(models[f'correction_{s}']) for s in range(3)}
    def prediction(c):
        if method=='e120_ensemble':
            group='correction' if id(c.model) in correction_ids else 'reference';key=(group,c.n,c.gram.tobytes(),c.cross.tobytes())
            if cache.get(group+'_key')!=key:
                old=c.model;js=[]
                try:
                    for s in range(3):c.model=models[f'{group}_{s}'];js.append(original_predict(c)[0])
                finally:c.model=old
                # Keep both same-history roles cached through controller.update.
                if cache.get(group+'_key')!=key:cache[group+'_key']=key;cache[group+'_j']=np.mean(js,axis=0)
            return cache[group+'_j'].copy(),c.bias
        if method=='analytical':return original_predict(c)
        hq,hx=history(c);key=(hq.tobytes(),hx.tobytes(),c.n)
        if cache.get('key')!=key:
            q=torch.tensor(hq[None],device='cuda');x=torch.tensor(hx[None],device='cuda');mask=torch.arange(7,device='cuda')[None]<c.n
            with torch.no_grad():
                if method=='learned_ensemble':
                    features=token_features(q,x,mask);j=torch.stack([m(q,x,mask,features)[0] for m in learned]).mean(0)
                else:
                    forgetting,ridge=(.98,1e-6) if method=='uniform32' else (choice['forgetting'],choice['ridge']);w=forgetting**torch.arange(31,-1,-1,device='cuda',dtype=torch.float64);qq,xx=q.double(),x.double();g=qq.transpose(1,2)@(qq*w[None,:,None])+ridge*torch.eye(7,device='cuda',dtype=torch.float64)[None];cross=qq.transpose(1,2)@(xx*w[None,:,None]);j=torch.linalg.solve(g,cross)
            cache['key']=key;cache['j']=j[0,:c.n].cpu().numpy().T.copy()
        return cache['j'].copy(),c.bias
    def observer(env,home,c,**kwargs):
        original_step=env.step
        def step(action,aperture):
            before=env.observe();phase=env.phase;clean=not env.bad;hq,hx=history(c);g=np.zeros((7,7));g[:c.n,:c.n]=c.gram;cross=np.zeros((7,6));cross[:c.n]=c.cross
            after,command,info=original_step(action,aperture);dq=np.zeros(7);dq[:c.n]=after['q']-before['q'];dx=np.r_[after['position']-before['position'],ROT_SCALE*(after['rotation'][:,2]-before['rotation'][:,2])]
            contexts.append(dict(dq_history=hq,dx_history=hx,query=dq.astype(np.float32),target=dx.astype(np.float32),gram=g.astype(np.float32),cross=cross.astype(np.float32),clean=clean,phase=(*PHASES,'probe').index(phase)))
            logs.append(dict(phase=phase,q_before=before['q'].tolist(),position_before=before['position'].tolist(),nominal_action=np.asarray(action).tolist(),executed_command=command.tolist(),aperture=float(aperture),bad=info['bad']));return after,command,info
        env.step=step;return original_rollout(env,home,c,**kwargs)
    with patch.object(core.LocalPose,'predict',prediction),patch.object(core,'probe_rollout',observer):
        if method=='e120_ensemble':r,_=previous.evaluate(case,t,method,models['reference_0'],models['correction_0'],'bounded',save=True)
        else:r=core.evaluate(case,t,method,None,'frozen_bias',save_states=True)
    stem=f'{case}_{method}_frozen_bias';save(ROOT/'trials'/f'{stem}.json',r);save(ROOT/'actions'/f'{stem}.json',logs);arrays={k:np.stack([c[k] for c in contexts]) for k in contexts[0]};np.savez_compressed(ROOT/'histories'/f'{stem}.npz',**arrays,mask=np.arange(7)<len(t['home']))
    return {k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')}
