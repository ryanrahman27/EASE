"""E56: matched clipped, bias-aware local-action training; synthetic only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from .benchmark import fingerprint
from .motion_transformer import MotionTransformer
from .control_training import batch,predict,response


def action_context(truth,mask,generator):
    size=truth.shape[0];device=truth.device
    dq=torch.randn(size,7,device=device,generator=generator)*.02
    dq=dq.masked_fill(~mask,0)
    error=(truth.transpose(1,2)@dq[:,:,None]).squeeze(-1)
    # Include small reference residuals as well as larger tracking deviations.
    scale=torch.exp(torch.empty(size,1,device=device).uniform_(-2.3,.7,generator=generator))
    error=error*scale
    bias=(torch.randn(size,7,device=device,generator=generator)*.006).clamp(-.025,.025).masked_fill(~mask,0)
    q=torch.empty(size,7,device=device).uniform_(-2.5,2.5,generator=generator).masked_fill(~mask,0)
    return error,bias,q


def local_loss(prediction,truth,mask,context):
    error,bias,q=context
    motion=(response(prediction)@(.35*error[:,:,None])).squeeze(-1)
    action=(motion+bias).clamp(-.04,.04)
    command=(q+action).clamp(-2.5,2.5)-q
    displacement=(command-bias).masked_fill(~mask,0)
    residual=error-(truth.transpose(1,2)@displacement[:,:,None]).squeeze(-1)
    return (residual.square().sum(-1)/(error.square().sum(-1)+.002**2)).mean()


def losses(prediction,target,truth,mask,teacher,context):
    fit=((prediction-target).square()*mask[:,:,None]).sum()/(6*mask.sum())
    weight=torch.where(mask.sum(-1)==6,2.,.5)[:,None,None]
    anchor=((prediction-teacher).square()*mask[:,:,None]*weight).sum()/(6*mask.sum())
    return fit,anchor,local_loss(prediction,truth,mask,context)


def run(parents, output, steps=2000):
    torch.set_num_threads(4)
    root=Path(output); root.mkdir(parents=True, exist_ok=False)
    hashes={str(Path(parents)/f'seed_{s}'/'policy.pt'): fingerprint(Path(parents)/f'seed_{s}'/'policy.pt') for s in range(3)}
    protocol=dict(experiment='E56', steps=steps, batch=128, lr=2e-5, control_weight=.01, parents=hashes,
        conditions=['prediction', 'control'], seeds=[0,1,2], test_evaluated=False,
        data='Synthetic only. Shared XYZ/tool-Z statistics; 1–256 transitions, full first 32 then rank 1–3, .98 decay, 1e-6 ridge, 6/7 joints, tangent-plane tool-Z responses, mostly clean measurement noise. Same batches per paired model.',
        loss='Both: same clean-ridge MSE and frozen E35 anchor (6-joint weight 2, 7-joint .5) as E52. Control adds .01 times mean squared local residual divided by squared task error plus .002^2. Action uses gain .35, damping .015, clip .04 and limits +/-2.5; predicted displacement subtracts measured bias. Error J*dq with dq sigma .02 and scale exp U(-2.3,.7); bias clipped Gaussian sigma .006; q uniform +/-2.5. Independent context RNG; identical state batches to E52. No native data or geometry. Fixed coefficient, no sweep.',
        selection='Fixed final checkpoints, no seed selection. Original 256 training tasks per seed first; all failures retained. Native development comparison requires six-joint retention before adoption. No final test.',
        evaluation='Fixed independent synthetic batch; original training control; same twelve observed E46 manipulation tasks, original controller/tool-Z objective. No E51 angular intervention.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (root/'source').mkdir()
    for name in ['local_action_training.py','control_training.py','motion_transformer.py','joint_policy.py']:
        shutil.copy2(Path(__file__).with_name(name), root/'source'/name)
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
        def metrics(model):
            with torch.no_grad():
                g,c,m,t,j=validation
                return [float(x) for x in losses(predict(model,g,c,m),t,j,m,predict(teacher,g,c,m),vc)]
        initial=metrics(teacher);gen=torch.Generator(device='cuda').manual_seed(52000+seed)
        context_gen=torch.Generator(device='cuda').manual_seed(56000+seed)
        digest=hashlib.sha256();context_digest=hashlib.sha256()
        for step in range(steps):
            g,c,m,t,j=batch(128,'cuda',gen)
            context=action_context(j,m,context_gen)
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
            meta=dict(seed=seed,steps=steps,policy_sha256=fingerprint(folder/'policy.pt'),parent_sha256=fingerprint(Path(parents)/f'seed_{seed}'/'policy.pt'),batch_sha256=digest.hexdigest(),context_sha256=context_digest.hexdigest(),validation_initial=initial,validation_final=metrics(model),losses=history[condition])
            (folder/'training.json').write_text(json.dumps(meta,indent=2))
            print('complete',condition,seed,meta['validation_final'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['parents','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--steps',type=int,default=2000)
    run(**vars(p.parse_args()))
