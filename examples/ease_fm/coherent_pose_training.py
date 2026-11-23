"""E57: matched training with coherent procedural contexts and synthetic retention."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .benchmark import fingerprint
from .motion_transformer import MotionTransformer
from .control_training import batch,predict
from .local_action_training import action_context,losses


def load_contexts(data,device):
    root=Path(data);audit=json.loads((root/'audit.json').read_text())
    assert json.loads((root/'status.json').read_text())['status']=='complete'
    assert fingerprint(root/'contexts.npz')==audit['contexts_sha256']
    with np.load(root/'contexts.npz') as f:
        values={k:torch.tensor(f[k],device=device) for k in f.files}
    train=torch.where(~values['validation'])[0];validation=torch.where(values['validation'])[0]
    assert len(train)==audit['train'] and len(validation)==audit['validation']
    assert set(train.cpu().tolist()).isdisjoint(validation.cpu().tolist())
    return values,train,validation


def context_batch(values,indices):
    g,c,m,j=[values[k][indices] for k in ['g','c','mask','j']]
    context=tuple(values[k][indices] for k in ['error','bias','q'])
    return (g,c,m,j,j),context


def mixed_batch(values,indices,generator,synthetic_generator,context_generator):
    selected=indices[torch.randint(len(indices),(64,),device=indices.device,generator=generator)]
    measured,mc=context_batch(values,selected)
    synthetic=batch(64,str(indices.device),synthetic_generator)
    sc=action_context(synthetic[4],synthetic[2],context_generator)
    return tuple(torch.cat([a,b],0) for a,b in zip(synthetic,measured)),tuple(torch.cat([a,b],0) for a,b in zip(sc,mc))


def run(parents, data, output, steps=2000):
    torch.set_num_threads(4)
    root=Path(output); root.mkdir(parents=True, exist_ok=False)
    hashes={str(Path(parents)/f'seed_{s}'/'policy.pt'): fingerprint(Path(parents)/f'seed_{s}'/'policy.pt') for s in range(3)}
    protocol=dict(experiment='E57', steps=steps, batch=128, lr=2e-5, control_weight=.01, parents=hashes,
        conditions=['prediction', 'control'], seeds=[0,1,2], test_evaluated=False, coherent_data=data, coherent_sha256=fingerprint(Path(data)/'contexts.npz'),
        data='Every batch: 64 synthetic E56 contexts plus 64 uniformly sampled coherent measured procedural training contexts. Same paired rows. Trajectory holdout, not unseen bodies. Procedural labels are current geometric Jacobians; synthetic labels remain clean-ridge estimates. No native data or geometry.',
        loss='Same E56 fixed fit/anchor loss and .01 normalized local-action term, lr 2e-5, 2000 updates, batch128. Prediction-only gets fit+anchor; control gets fit+anchor+.01 local. Synthetic retention mixture is fixed at 50 percent before training. Context/label distribution changes together versus E56, so cross-experiment differences do not isolate coherence alone.',
        selection='Fixed final checkpoints, no seed selection. Original 256 training tasks per seed first; all failures retained. Native development comparison requires six-joint retention before adoption. No final test.',
        evaluation='Fixed independent synthetic batch; original training control; same twelve observed E46 manipulation tasks, original controller/tool-Z objective. No E51 angular intervention.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (root/'source').mkdir()
    for name in ['coherent_pose_training.py','coherent_pose_data.py','local_action_training.py','control_training.py','motion_transformer.py']:
        shutil.copy2(Path(__file__).with_name(name), root/'source'/name)
    values,train_indices,validation_indices=load_contexts(data,'cuda')
    held_indices=validation_indices[torch.randperm(len(validation_indices),device='cuda',generator=torch.Generator(device='cuda').manual_seed(57999))[:512]]
    held,held_context=context_batch(values,held_indices)
    vg=torch.Generator(device='cuda').manual_seed(52999)
    validation=batch(512,'cuda',vg)
    vc=action_context(validation[4],validation[2],torch.Generator(device='cuda').manual_seed(56999))
    for seed in range(3):
        teacher=MotionTransformer().cuda()
        teacher.load_state_dict(torch.load(Path(parents)/f'seed_{seed}'/'policy.pt',map_location='cuda',weights_only=True));teacher.eval()
        for parameter in teacher.parameters():parameter.requires_grad_(False)
        models={};optimizers={};history={k:[] for k in ['prediction','control']}
        for condition in history:
            model=MotionTransformer().cuda();model.load_state_dict(teacher.state_dict());models[condition]=model
            optimizers[condition]=torch.optim.AdamW(model.parameters(),lr=2e-5)
        def metrics(model,coherent=False):
            with torch.no_grad():
                g,c,m,t,j=held if coherent else validation
                context=held_context if coherent else vc
                return [float(x) for x in losses(predict(model,g,c,m),t,j,m,predict(teacher,g,c,m),context)]
        initial=metrics(teacher);coherent_initial=metrics(teacher,True);row_gen=torch.Generator(device='cuda').manual_seed(57000+seed);gen=torch.Generator(device='cuda').manual_seed(52000+seed)
        context_gen=torch.Generator(device='cuda').manual_seed(56000+seed)
        digest=hashlib.sha256();context_digest=hashlib.sha256()
        for step in range(steps):
            (g,c,m,t,j),context=mixed_batch(values,train_indices,row_gen,gen,context_gen)
            for value in context:context_digest.update(value.detach().cpu().numpy().tobytes())
            # One generated batch shared directly by both optimizer steps.
            digest.update(g.detach().cpu().numpy().tobytes());digest.update(c.detach().cpu().numpy().tobytes())
            with torch.no_grad():old=predict(teacher,g,c,m)
            for condition,model in models.items():
                fit,anchor,control=losses(predict(model,g,c,m),t,j,m,old,context)
                loss=fit+anchor+(.01*control if condition=='control' else 0)
                assert torch.isfinite(loss)
                optimizer=optimizers[condition];optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                history[condition].append([float(fit.detach()),float(anchor.detach()),float(control.detach())])
            if (step+1)%250==0:print('seed',seed,'step',step+1,{k:v[-1] for k,v in history.items()},flush=True)
        for condition,model in models.items():
            folder=root/condition/f'seed_{seed}';folder.mkdir(parents=True)
            torch.save(model.state_dict(),folder/'policy.pt')
            meta=dict(seed=seed,steps=steps,policy_sha256=fingerprint(folder/'policy.pt'),parent_sha256=fingerprint(Path(parents)/f'seed_{seed}'/'policy.pt'),batch_sha256=digest.hexdigest(),context_sha256=context_digest.hexdigest(),validation_initial=initial,validation_final=metrics(model),coherent_initial=coherent_initial,coherent_final=metrics(model,True),losses=history[condition])
            (folder/'training.json').write_text(json.dumps(meta,indent=2))
            print('complete',condition,seed,meta['validation_final'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['parents','data','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--steps',type=int,default=2000)
    run(**vars(p.parse_args()))
