"""E44: fixed-budget synthetic long-history motion-estimator fine tuning."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .motion_transformer import MotionTransformer
from .robust_training import synthetic_batch
from .paired_data import register
from .benchmark import fingerprint


def long_batch(batch,device):
    n=torch.randint(6,8,(batch,),device=device);mask=torch.arange(7,device=device)[None]<n[:,None]
    lengths=torch.randint(33,257,(batch,),device=device);time=torch.arange(256,device=device)[None]
    rank=torch.randint(1,4,(batch,),device=device)
    basis=torch.randn(batch,3,7,device=device)/7**.5
    coefficients=torch.randn(batch,256,3,device=device)
    coefficients*= (torch.arange(3,device=device)[None,None]<rank[:,None,None])
    tail=coefficients@basis
    probe=torch.randn(batch,256,7,device=device)
    dq=torch.where((time<32)[:,:,None],probe,tail)
    amplitude=torch.exp(torch.empty(batch,1,1,device=device).uniform_(np.log(.0002),np.log(.012)))
    dq*=amplitude;dq=dq.masked_fill(~mask[:,None],0)
    jac=torch.randn(batch,7,3,device=device)*torch.exp(torch.empty(batch,1,1,device=device).uniform_(np.log(.03),np.log(.8)))
    dx=dq@jac
    weights=.98**(lengths[:,None]-1-time).clamp_min(0)*(time<lengths[:,None])
    eye=1e-6*torch.eye(7,device=device)[None]
    clean_g=dq.transpose(1,2)@(dq*weights[:,:,None])+eye
    clean_c=dq.transpose(1,2)@(dx*weights[:,:,None])
    target=torch.linalg.solve(clean_g,clean_c).masked_fill(~mask[:,:,None],0)
    # Most manipulation observations are clean; retain small-noise examples.
    sigma=torch.tensor([0.,0.,0.,.0001,.0005],device=device)[torch.randint(5,(batch,),device=device)]
    qnoise=torch.randn(batch,257,7,device=device)*sigma[:,None,None]
    xnoise=torch.randn(batch,257,3,device=device)*sigma[:,None,None]
    mdq=(dq+qnoise[:,1:]-qnoise[:,:-1]).masked_fill(~mask[:,None],0)
    mdx=dx+xnoise[:,1:]-xnoise[:,:-1]
    return mdq.transpose(1,2)@(mdq*weights[:,:,None])+eye,mdq.transpose(1,2)@(mdx*weights[:,:,None]),mask,target


def run(data,parents,output,steps=2000,retention=False):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    manifest=register(data);profiles=[]
    for entry in manifest['calibrations']:
        assert entry['split']=='train'
        h=np.load(Path(data)/entry['file']);p=np.zeros((32,7),np.float32);p[:,:h.shape[1]]=h[:,:,3];profiles.append(p)
    profiles=torch.tensor(np.stack(profiles),device='cuda')
    hashes={str(Path(parents)/f'seed_{s}'/'policy.pt'):fingerprint(Path(parents)/f'seed_{s}'/'policy.pt') for s in range(3)}
    lr=.00002 if retention else .00005
    protocol=dict(experiment='E45' if retention else 'E44',steps=steps,batch=128,lr=lr,parents=hashes,data=data,
        retention=retention,anchor='If enabled: frozen corresponding E35 teacher prediction penalty on every synthetic batch, coefficient 2 for six joints / .5 for seven joints. Starts from E35, not E44. Architecture unchanged; no native data.',
        training_manifest_sha256=fingerprint(Path(data)/'manifest.json'),test_evaluated=False,
        training='Alternate E35 synthetic noisy short histories with synthetic 33-256-step histories: 32-step full-rank probe then rank-1/2/3 motion, .98 weighting, 1e-6 ridge; motion amplitudes .0002-.012, Jacobian scales .03-.8; mostly clean observations. Targets are clean weighted ridge estimates. No native robot rollouts or geometry in training. Fixed final checkpoints, no best-seed selection.',
        evaluation='Fixed synthetic validation batch and subsequent existing observed native development tasks; independent final test remains reserved.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'source').mkdir()
    for name in ['motion_lowrank.py','robust_training.py','motion_transformer.py']:
        shutil.copyfile(Path(__file__).with_name(name),root/'source'/name)
    torch.manual_seed(44999);validation=long_batch(1024,'cuda')
    for seed in range(3):
        torch.manual_seed(44000+seed);folder=root/f'seed_{seed}';folder.mkdir();parent=Path(parents)/f'seed_{seed}'/'policy.pt'
        model=MotionTransformer().cuda();model.load_state_dict(torch.load(parent,map_location='cuda',weights_only=True))
        teacher=None
        if retention:
            teacher=MotionTransformer().cuda();teacher.load_state_dict(model.state_dict());teacher.eval()
            for parameter in teacher.parameters():parameter.requires_grad_(False)
        def validation_loss():
            model.eval()
            with torch.no_grad():
                g,c,m,target=validation;return float(((model(g,c,m)-target).square()*m[:,:,None]).sum()/(3*m.sum()))
        initial=validation_loss();model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=lr);losses=[];fit_losses=[];anchor_losses=[]
        for step in range(steps):
            g,c,m,target=long_batch(128,'cuda') if step%2==0 else synthetic_batch(profiles,128,'cuda')
            prediction=model(g,c,m)
            fit_loss=((prediction-target).square()*m[:,:,None]).sum()/(3*m.sum())
            anchor_loss=prediction.sum()*0
            if teacher is not None:
                with torch.no_grad():old=teacher(g,c,m)
                weight=torch.where(m.sum(-1)==6,2.,.5)[:,None,None]
                anchor_loss=((prediction-old).square()*m[:,:,None]*weight).sum()/(3*m.sum())
            loss=fit_loss+anchor_loss;assert torch.isfinite(loss)
            fit_losses.append(float(fit_loss.detach()));anchor_losses.append(float(anchor_loss.detach()))
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
            if (step+1)%500==0:print('train',seed,step+1,float(loss.detach()),flush=True)
        final=validation_loss();torch.save(model.state_dict(),folder/'policy.pt')
        (folder/'training.json').write_text(json.dumps(dict(seed=seed,steps=steps,policy_sha256=fingerprint(folder/'policy.pt'),
            parent_sha256=hashes[str(parent)],validation_initial=initial,validation_final=final,losses=losses,
            fit_losses=fit_losses,anchor_losses=anchor_losses),indent=2))
        print('complete seed',seed,'synthetic validation',initial,final,flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['data','parents','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--steps',type=int,default=2000);p.add_argument('--retention',action='store_true');run(**vars(p.parse_args()))
