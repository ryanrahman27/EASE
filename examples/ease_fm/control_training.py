"""E52: matched prediction-only/control-aware synthetic fine tuning."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from .benchmark import fingerprint
from .motion_transformer import MotionTransformer


def batch(size, device, generator):
    def randn(*shape): return torch.randn(*shape, device=device, generator=generator)
    def randint(low, high, shape): return torch.randint(low, high, shape, device=device, generator=generator)
    mask = torch.arange(7, device=device)[None] < randint(6, 8, (size, 1))
    lengths = randint(1, 257, (size,))
    time = torch.arange(256, device=device)[None]
    rank = randint(1, 4, (size, 1, 1))
    basis = randn(size, 3, 7) / 7**.5
    coeff = randn(size, 256, 3) * (torch.arange(3, device=device)[None, None] < rank)
    dq = torch.where((time < 32)[:, :, None], randn(size, 256, 7), coeff @ basis)
    amplitude = torch.exp(torch.empty(size, 1, 1, device=device).uniform_(-8.5172, -4.4228, generator=generator))
    dq = (dq * amplitude).masked_fill(~mask[:, None], 0)
    xyz = randn(size, 7, 3) * .3
    z = randn(size, 3); z = z / z.norm(dim=-1, keepdim=True)
    tangent = torch.eye(3, device=device)[None] - z[:, :, None] * z[:, None, :]
    direction = (randn(size, 7, 3) @ tangent) * .15
    truth = torch.cat([xyz, direction], -1).masked_fill(~mask[:, :, None], 0)
    dx = dq @ truth
    weights = .98**(lengths[:, None]-1-time).clamp_min(0) * (time < lengths[:, None])
    ridge = 1e-6 * torch.eye(7, device=device)[None]
    clean_g = dq.transpose(1, 2) @ (dq * weights[:, :, None]) + ridge
    clean_c = dq.transpose(1, 2) @ (dx * weights[:, :, None])
    target = torch.linalg.solve(clean_g, clean_c)
    sigma = torch.tensor([0., 0., 0., .0001, .0005], device=device)[randint(0, 5, (size,))]
    qnoise = randn(size, 257, 7) * sigma[:, None, None]
    xnoise = randn(size, 257, 6) * sigma[:, None, None]
    mdq = (dq + qnoise[:, 1:] - qnoise[:, :-1]).masked_fill(~mask[:, None], 0)
    mdx = dx + xnoise[:, 1:] - xnoise[:, :-1]
    return mdq.transpose(1, 2) @ (mdq * weights[:, :, None]) + ridge, mdq.transpose(1, 2) @ (mdx * weights[:, :, None]), mask, target, truth


def predict(model, g, c, mask):
    return torch.cat([model(g, c[:, :, :3], mask), model(g, c[:, :, 3:], mask)], -1)


def response(jac):
    """Damped task-to-joint operator; training only, never neural inference."""
    return torch.linalg.solve(jac.transpose(1, 2) @ jac + .015**2 * torch.eye(6, device=jac.device)[None], jac.transpose(1, 2)).transpose(1, 2)


def losses(prediction, target, truth, mask, teacher):
    fit = ((prediction-target).square()*mask[:, :, None]).sum()/(6*mask.sum())
    weight = torch.where(mask.sum(-1)==6, 2., .5)[:, None, None]
    anchor = ((prediction-teacher).square()*mask[:, :, None]*weight).sum()/(6*mask.sum())
    # Compare achieved local task response to a damped true-model controller.
    # This does not demand tracking the unobservable direction-normal component.
    desired = truth.transpose(1, 2) @ response(truth)
    achieved = truth.transpose(1, 2) @ response(prediction)
    control = (achieved-desired).square().mean()
    return fit, anchor, control


def run(parents, output, steps=2000):
    torch.set_num_threads(4)
    root=Path(output); root.mkdir(parents=True, exist_ok=False)
    hashes={str(Path(parents)/f'seed_{s}'/'policy.pt'): fingerprint(Path(parents)/f'seed_{s}'/'policy.pt') for s in range(3)}
    protocol=dict(experiment='E52', steps=steps, batch=128, lr=2e-5, control_weight=.1, parents=hashes,
        conditions=['prediction', 'control'], seeds=[0,1,2], test_evaluated=False,
        data='Synthetic only. Shared XYZ/tool-Z statistics; 1–256 transitions, full first 32 then rank 1–3, .98 decay, 1e-6 ridge, 6/7 joints, tangent-plane tool-Z responses, mostly clean measurement noise. Same batches per paired model.',
        loss='Both: clean ridge prediction MSE plus frozen E35 anchor (six-joint weight 2, seven-joint .5). Control adds .1 times mean squared achieved task-response operator error using damping .015 and true synthetic Jacobian. No native data or geometry. Unclipped linear response surrogate; not a differentiable contact simulator.',
        selection='Fixed final checkpoints, no seed selection. Original 256 training tasks per seed first; all failures retained. Native development comparison requires six-joint retention before adoption. No final test.',
        evaluation='Fixed independent synthetic batch; original training control; same twelve observed E46 manipulation tasks, original controller/tool-Z objective. No E51 angular intervention.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (root/'source').mkdir()
    for name in ['control_training.py','motion_transformer.py','joint_policy.py']:
        shutil.copy2(Path(__file__).with_name(name), root/'source'/name)
    vg=torch.Generator(device='cuda').manual_seed(52999)
    validation=batch(512,'cuda',vg)
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
                return [float(x) for x in losses(predict(model,g,c,m),t,j,m,predict(teacher,g,c,m))]
        initial=metrics(teacher);gen=torch.Generator(device='cuda').manual_seed(52000+seed)
        digest=hashlib.sha256()
        for step in range(steps):
            g,c,m,t,j=batch(128,'cuda',gen)
            # One generated batch shared directly by both optimizer steps.
            digest.update(g.detach().cpu().numpy().tobytes());digest.update(c.detach().cpu().numpy().tobytes())
            with torch.no_grad():old=predict(teacher,g,c,m)
            for condition,model in models.items():
                fit,anchor,control=losses(predict(model,g,c,m),t,j,m,old)
                loss=fit+anchor+(.1*control if condition=='control' else 0)
                assert torch.isfinite(loss)
                optimizer=optimizers[condition];optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
                history[condition].append([float(fit.detach()),float(anchor.detach()),float(control.detach())])
            if (step+1)%250==0:print('seed',seed,'step',step+1,{k:v[-1] for k,v in history.items()},flush=True)
        for condition,model in models.items():
            folder=root/condition/f'seed_{seed}';folder.mkdir(parents=True)
            torch.save(model.state_dict(),folder/'policy.pt')
            meta=dict(seed=seed,steps=steps,policy_sha256=fingerprint(folder/'policy.pt'),parent_sha256=fingerprint(Path(parents)/f'seed_{seed}'/'policy.pt'),batch_sha256=digest.hexdigest(),validation_initial=initial,validation_final=metrics(model),losses=history[condition])
            (folder/'training.json').write_text(json.dumps(meta,indent=2))
            print('complete',condition,seed,meta['validation_final'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['parents','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--steps',type=int,default=2000)
    run(**vars(p.parse_args()))
