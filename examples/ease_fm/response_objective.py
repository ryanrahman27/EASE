"""E61 differentiable frozen response correction on measured contexts."""
import json
from pathlib import Path
import numpy as np
import torch
from .benchmark import fingerprint
from .control_training import batch,response
from .local_action_training import action_context,local_loss


def load_contexts(data,response_data,device):
    root=Path(data);rr=Path(response_data)
    assert fingerprint(root/'contexts.npz')==json.loads((root/'audit.json').read_text())['contexts_sha256']
    assert fingerprint(rr/'transitions.npz')==json.loads((rr/'audit.json').read_text())['data_sha256']
    provenance=json.loads((root/'provenance.json').read_text());source=np.load(root/'contexts.npz');trans=np.load(rr/'transitions.npz')
    selected=[i for i,r in enumerate(provenance) if r['driver'] in ['analytical','original_0','original_1','original_2']]
    values={k:source[k][selected] for k in source.files};alpha=[];offset=[]
    for i in selected:
        row=provenance[i];mask=(trans['body']==row['body'])&(trans['driver']==row['driver'])&(trans['step']==row['step'])
        x=trans['x'][mask];fold=np.load(rr/f"fold_{row['body']}_history.npz")
        assert len(x)==int(source['mask'][i].sum())
        np.testing.assert_allclose(x[:,2],source['q'][i,:len(x)],atol=1e-7)
        np.testing.assert_allclose(x[:,5],source['bias'][i,:len(x)],atol=1e-7)
        coef=fold['weight'][1:]/fold['scale'];intercept=fold['weight'][0]-fold['mean']@coef
        aa=np.zeros(7);bb=np.zeros(7);aa[:len(x)]=1+coef[0]
        bb[:len(x)]=-x[:,5]+intercept+x[:,1:]@coef[1:]
        alpha.append(aa);offset.append(bb)
    values['alpha']=np.asarray(alpha,dtype=np.float32);values['offset']=np.asarray(offset,dtype=np.float32)
    values={k:torch.tensor(v,device=device) for k,v in values.items()}
    train=torch.where(~values['validation'])[0];validation=torch.where(values['validation'])[0]
    assert len(train)==3600 and len(validation)==1200
    return values,train,validation


def context_batch(values,indices):
    g,c,m,j=[values[k][indices] for k in ['g','c','mask','j']]
    return (g,c,m,j,j),tuple(values[k][indices] for k in ['error','bias','q','alpha','offset'])


def mixed_batch(values,indices,generator,synthetic_generator,context_generator):
    selected=indices[torch.randint(len(indices),(64,),device=indices.device,generator=generator)]
    measured,mc=context_batch(values,selected);synthetic=batch(64,str(indices.device),synthetic_generator)
    sc=action_context(synthetic[4],synthetic[2],context_generator)
    sc=(*sc,synthetic[2].float(),-sc[1])
    return tuple(torch.cat([a,b],0) for a,b in zip(synthetic,measured)),tuple(torch.cat([a,b],0) for a,b in zip(sc,mc))


def corrected_loss(prediction,truth,mask,context):
    error,bias,q,alpha,offset=context
    motion=(response(prediction)@(.35*error[:,:,None])).squeeze(-1)
    command=(q+(motion+bias).clamp(-.04,.04)).clamp(-2.5,2.5)-q
    displacement=(alpha*command+offset).masked_fill(~mask,0)
    residual=error-(truth.transpose(1,2)@displacement[:,:,None]).squeeze(-1)
    return (residual.square().sum(-1)/(error.square().sum(-1)+.002**2)).mean()


def losses(prediction,target,truth,mask,teacher,context):
    fit=((prediction-target).square()*mask[:,:,None]).sum()/(6*mask.sum())
    weight=torch.where(mask.sum(-1)==6,2.,.5)[:,None,None]
    anchor=((prediction-teacher).square()*mask[:,:,None]*weight).sum()/(6*mask.sum())
    return fit,anchor,local_loss(prediction,truth,mask,context[:3]),corrected_loss(prediction,truth,mask,context)
