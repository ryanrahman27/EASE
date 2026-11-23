"""E35: fixed-budget noisy/short-history fine tuning and matched delay control."""
import argparse
import copy
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .benchmark import fingerprint
from .paired_data import register
from .motion_transformer import MotionTransformer
from .robustness import CONDITIONS, evaluate


def synthetic_batch(profiles, batch, device):
    dofs=torch.randint(6,8,(batch,),device=device)
    mask=torch.arange(7,device=device)[None]<dofs[:,None]
    length=torch.randint(1,33,(batch,),device=device)
    time=torch.arange(32,device=device)[None]
    valid=time<length[:,None]
    dq=profiles[torch.randint(len(profiles),(batch,),device=device)].clone()
    mix=torch.randn((batch,7,7),device=device)/7**.5
    gaussian=torch.randn_like(dq)*.008
    dq=torch.where((torch.rand(batch,1,1,device=device)<.5),dq,gaussian)@mix
    dq*=torch.exp(torch.randn((batch,1,7),device=device)*.4)
    dq=dq.masked_fill(~mask[:,None],0)
    jac=torch.randn((batch,7,3),device=device)*(.15+.65*torch.rand(batch,1,1,device=device))
    dx=dq@jac
    # Differentiate noisy positions so adjacent transition errors are correlated.
    sigma=torch.tensor([0.,.0005,.001,.002,.005],device=device)[torch.randint(5,(batch,),device=device)]
    qnoise=torch.randn((batch,33,7),device=device)*sigma[:,None,None]
    xnoise=torch.randn((batch,33,3),device=device)*sigma[:,None,None]
    measured_dq=(dq+qnoise[:,1:]-qnoise[:,:-1]).masked_fill(~mask[:,None],0)
    measured_dx=dx+xnoise[:,1:]-xnoise[:,:-1]
    weights=.98**(length[:,None]-1-time).clamp_min(0)*valid
    # Same prior recurrence as online estimation: .98*1e-6+2e-8 == 1e-6.
    clean_g=dq.transpose(1,2)@(dq*weights[:,:,None])+1e-6*torch.eye(7,device=device)[None]
    clean_c=dq.transpose(1,2)@(dx*weights[:,:,None])
    target=torch.linalg.solve(clean_g,clean_c).masked_fill(~mask[:,:,None],0)
    gram=measured_dq.transpose(1,2)@(measured_dq*weights[:,:,None])+1e-6*torch.eye(7,device=device)[None]
    cross=measured_dq.transpose(1,2)@(measured_dx*weights[:,:,None])
    return gram,cross,mask,target


def fit(data,manifest,entry,root,steps,device):
    seed=entry['seed'];torch.manual_seed(35000+seed);root.mkdir()
    profiles=[]
    for c in manifest['calibrations']:
        assert c['split']=='train'
        h=np.load(Path(data)/c['file']);p=np.zeros((32,7),np.float32);p[:,:h.shape[1]]=h[:,:,3];profiles.append(p)
    profiles=torch.tensor(np.stack(profiles),device=device)
    model=MotionTransformer().to(device)
    model.load_state_dict(torch.load(Path(entry['folder'])/'policy.pt',map_location=device,weights_only=True))
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=.0001);losses=[]
    for step in range(steps):
        g,c,mask,target=synthetic_batch(profiles,128,device)
        prediction=model(g,c,mask)
        loss=((prediction-target).square()*mask[:,:,None]).sum()/(mask.sum()*3)
        assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        losses.append(loss.item())
        if (step+1)%1000==0:print('trained',seed,step+1,loss.item(),flush=True)
    model.eval();torch.save(model.state_dict(),root/'policy.pt')
    (root/'training.json').write_text(json.dumps(dict(seed=seed,steps=steps,losses=losses,
        policy_sha256=fingerprint(root/'policy.pt'),parent_sha256=entry['policy_sha256'],
        manifest_sha256=fingerprint(Path(data)/'manifest.json'),profile_count=len(profiles)),indent=2))
    return model


def run(data,evaluation,replication,output,steps=5000,device='cuda'):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    train=register(data);manifest=register(evaluation)
    entries=json.loads((Path(replication)/'index.json').read_text())
    for e in entries:assert fingerprint(Path(e['folder'])/'policy.pt')==e['policy_sha256']
    protocol=dict(experiment='E35',steps=steps,lr=.0001,batch=128,parents=entries,
        train=str(data),evaluation=str(evaluation),train_sha256=fingerprint(Path(data)/'manifest.json'),
        evaluation_sha256=fingerprint(Path(evaluation)/'manifest.json'),test_evaluated=False,
        training='Fine-tune all three existing seeds for 5000 updates each. Training-only excitation profiles mixed with Gaussian motions. Uniform history lengths 1..32; position noise sigma in {0,.0005,.001,.002,.005}, temporally correlated differences; clean weighted ridge target. Same network and controller parameters.',
        delay='Shared delay-aware gain .5/(1+delay_steps), no state predictor. Evaluate old neural and analytical controls with this gain to separate controller gains from training.',
        evaluation_scope='All seven E34 conditions, all 1024 tasks, fixed final checkpoints; observed development suite, not an independent confirmation. No selection or tuning after results.',conditions=CONDITIONS)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'source').mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,root/'source'/p.name)
    models={}
    for e in entries:models[f'new_{e["seed"]}']=fit(data,train,e,root/f'seed_{e["seed"]}',steps,device)
    original_solve=np.linalg.solve
    def guard(a,b):
        assert a.shape[-2:]==(3,3)
        return original_solve(a,b)
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
        for base in CONDITIONS:
            condition=copy.deepcopy(base);condition['delay_aware']=True
            methods=dict(models)
            if base.get('delay'):
                methods['analytical']=None
                for e in entries:
                    m=MotionTransformer().to(device);m.load_state_dict(torch.load(Path(e['folder'])/'policy.pt',map_location=device,weights_only=True));methods[f'old_{e["seed"]}']=m.eval()
            for name,model in methods.items():
                if model is None:rows=evaluate(manifest,model,condition,device)
                else:
                    with patch.object(np.linalg,'solve',side_effect=guard),patch.object(torch.linalg,'solve',side_effect=RuntimeError('Neural inverse forbidden')):
                        rows=evaluate(manifest,model,condition,device)
                assert len(rows)==1024
                (root/f'{base["name"]}__{name}.json').write_text(json.dumps(rows,indent=2))
                print(base['name'],name,sum(r['success'] for r in rows),'/1024',flush=True)
    for e in entries:assert fingerprint(Path(e['folder'])/'policy.pt')==e['policy_sha256']
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('data','evaluation','replication','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--steps',type=int,default=5000);p.add_argument('--device',default='cuda');run(**vars(p.parse_args()))
