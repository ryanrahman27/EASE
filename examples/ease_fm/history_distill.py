"""Train measured-history conditioning against a frozen privileged controller."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .benchmark import fingerprint
from .budget_experiment import gate
from .diverse_oracle import oracle_tokens
from .full_control import add_recovery, rollout
from .joint_policy import JointFlowPolicy
from .mixed_data import MixedData
from .mixed_sim import Arm, MixedReach, explore


def encoder_key(name):
    return name.startswith('history_') or name=='empty'


def make_student(teacher,contexts,device):
    student=copy.deepcopy(teacher)
    torch.manual_seed(0)
    fresh=JointFlowPolicy(history=True).to(device).state_dict()
    state=student.state_dict()
    for name in state:
        if encoder_key(name): state[name]=fresh[name]
    student.load_state_dict(state)
    h=torch.tensor(np.concatenate([c.reshape(-1,11) for c in contexts]),device=device)
    student.history_mean.copy_(h.mean(0))
    student.history_std.copy_(h.std(0,unbiased=False).clamp_min(.01))
    for name,param in student.named_parameters(): param.requires_grad_(encoder_key(name))
    return student


def frozen_equal(student,teacher):
    reference=teacher.state_dict()
    return all(torch.equal(value,reference[name]) for name,value in student.state_dict().items() if not encoder_key(name))


def run(data,source,teacher,output,steps=10000,device='cuda'):
    torch.set_num_threads(4)
    root=Path(output); source=Path(source); teacher_path=Path(teacher)
    dataset=MixedData(data)
    shared=json.loads((source/'shared_recovery.json').read_text())
    for relative,digest in shared['hashes'].items(): assert fingerprint(source/relative)==digest
    dataset=add_recovery(dataset,[source/f'original_{kind}/recovery.npz' for kind in ['flow','direct']])
    meta=json.loads((teacher_path.parent/'training.json').read_text())
    manifest_hash=fingerprint(Path(data)/'manifest.json')
    assert meta['manifest_sha256']==manifest_hash and meta['policy_sha256']==fingerprint(teacher_path)
    assert json.loads((teacher_path.parent/'gate.json').read_text())['passed']
    root.mkdir(parents=True,exist_ok=False)
    teacher_model=JointFlowPolicy(history=True).to(device)
    teacher_model.load_state_dict(torch.load(teacher_path,map_location=device,weights_only=True))
    teacher_model.eval().requires_grad_(False)
    student=make_student(teacher_model,dataset.contexts,device)
    arms=[Arm(**a) for a in dataset.manifest['arms'] if a['split']=='train']
    ids={a.seed:i for i,a in enumerate(arms)}
    masks=torch.tensor([[j<a.dof for j in range(7)] for a in arms],device=device)
    physical=torch.tensor(np.stack([oracle_tokens(a,padded=True) for a in arms]),device=device)
    with torch.no_grad(): target=teacher_model.encode(physical,torch.full((len(arms),),32,device=device),masks)
    row_body=torch.tensor([ids[dataset.episode_body[e]] for e in dataset.row_episode],device=device)
    protocol={'experiment':'E17','scope':'Training-body latent distillation; no unseen-body evaluation',
        'teacher':str(teacher_path),'teacher_sha256':fingerprint(teacher_path),'teacher_steps':meta['steps'],
        'manifest_sha256':manifest_hash,'recovery_hashes':shared['hashes'],'seed':0,'steps':steps,'batch':32,'lr':.0003,
        'loss':'Masked per-joint raw latent MSE against frozen privileged encoder targets; no action or task loss',
        'student_initialization':'Fresh seed-0 history modules; inherited frozen teacher controller and joint/task normalization',
        'trainable_parameters':sum(p.numel() for p in student.parameters() if p.requires_grad),
        'total_parameters':sum(p.numel() for p in student.parameters()),
        'history':'Same other-episode variable 0..32 context sampling as E15/E16; measured-history training normalization',
        'sampling':'Uniform original+recovery rows; recovery rows supply body/context selection only, not action supervision',
        'evaluation_modes':['correct32','empty','wrong_axes32'],
        'wrong_donor':'First training arm of same DoF and different axis family',
        'privileged_information':'Training target only; no physical parameters or identity input at student inference',
        'test_evaluated':False,'validation_evaluated':False,
        'interpretation':'Distillation diagnostic includes pretrained privileged-controller training; not matched total compute against end-to-end baseline'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    snapshot=root/'source'; snapshot.mkdir()
    for path in Path(__file__).parent.glob('*.py'): shutil.copyfile(path,snapshot/path.name)
    opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=.0003)
    rng=torch.Generator(device=device).manual_seed(1000); context_rng=np.random.default_rng(3000)
    losses=[]
    for step in range(steps):
        idx=torch.randint(len(row_body),(32,),device=device,generator=rng)
        body=row_body[idx]; mask=masks[body]
        history,lengths=dataset.histories(idx.cpu().numpy(),context_rng)
        z=student.encode(history.to(device),lengths.to(device),mask)
        loss=((z-target[body]).square()*mask[...,None]).sum()/(mask.sum()*student.width)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite latent loss')
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad],1.); opt.step()
        losses.append(loss.item())
        if (step+1)%1000==0: print('distill',step+1,'latent loss',losses[-1],flush=True)
    assert frozen_equal(student,teacher_model)
    student.eval()
    torch.save(student.state_dict(),root/'policy.pt')
    (root/'training.json').write_text(json.dumps({'kind':'direct','seed':0,'steps':steps,'losses':losses,
        'manifest_sha256':manifest_hash,'policy_sha256':fingerprint(root/'policy.pt'),
        'controller_bitwise_unchanged':True,'trainable_parameters':protocol['trainable_parameters']},indent=2))
    # Independent calibration episodes, paired across bodies; no physical values enter student.encode.
    probes=[]
    with torch.no_grad():
        for seed in [710001,710002,710003,710004]:
            histories=torch.tensor(np.stack([np.pad(explore(MixedReach(a,synchronize=True),seed,32),
                ((0,0),(0,7-a.dof),(0,0))) for a in arms]),device=device)
            donors=[next(i for i,b in enumerate(arms) if b.dof==a.dof and b.family!=a.family) for a in arms]
            for mode in ['correct32','empty','wrong_axes32']:
                z=student.encode(histories[donors] if mode=='wrong_axes32' else histories,
                    torch.full((len(arms),),0 if mode=='empty' else 32,device=device),masks)
                for i,arm in enumerate(arms):
                    probes.append({'seed':seed,'body_seed':arm.seed,'family':arm.family,'mode':mode,
                        'latent_mse':float((z[i,:arm.dof]-target[i,:arm.dof]).square().mean())})
    (root/'latent_probe.json').write_text(json.dumps(probes,indent=2))
    rows=[]
    for mode in protocol['evaluation_modes']:
        records=rollout(dataset,student,'direct',root,device,mode=mode)
        result=gate(records)
        row={'mode':mode,**result,'collisions':sum(r['collision'] for r in records)}
        rows.append(row); print(row,flush=True)
        (root/'summary.json').write_text(json.dumps(rows,indent=2))
    assert frozen_equal(student,teacher_model) and fingerprint(teacher_path)==protocol['teacher_sha256']


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['data','source','teacher','output']: parser.add_argument('--'+name,required=True)
    parser.add_argument('--steps',type=int,default=10000); parser.add_argument('--device',default='cuda')
    run(**vars(parser.parse_args()))
