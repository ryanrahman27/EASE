"""Post-hoc sampling diagnostics on the frozen recovery-trained tiny flow policy."""
import json
from pathlib import Path
import shutil
import torch
from examples.ease_fm import overfit
from examples.ease_fm.mixed_data import MixedData
from examples.ease_fm.mixed_sim import Arm
from examples.ease_fm.joint_policy import JointFlowPolicy
from examples.ease_fm.benchmark import fingerprint

def main():
    torch.set_num_threads(4)
    root=Path('outputs/ease_overfit_sampler_v1'); root.mkdir(parents=True,exist_ok=False)
    source=Path('outputs/ease_overfit_retrained_v1')
    protocol=json.loads((source/'protocol.json').read_text())
    checkpoint=source/'flow/policy_5000.pt'
    (root/'protocol.json').write_text(json.dumps({'scope':'Post-hoc frozen-checkpoint diagnostics, no training or held-out evaluation',
        'checkpoint_sha256':fingerprint(checkpoint),'modes':['euler50','zero_noise10','mean8_noise10'],
        'zero_noise_repeats':'Identical deterministic repeats retained by shared evaluator; count unique episodes only',
        'mean8':'Mean of eight sampled bounded first commands; eightfold flow compute'},indent=2))
    shutil.copyfile(__file__,root/'source.py')
    d=MixedData('outputs/ease_mixed_v1')
    accepted=[e for e in d.manifest['episodes'] if e['split']=='train' and e['success'] and not e['collision']]
    selected=[i for i,e in enumerate(accepted) if e['file'] in protocol['episode_files']]
    indices=[i for i,eid in enumerate(d.row_episode) if eid in selected]
    arm=next(Arm(**a) for a in d.manifest['arms'] if a['seed']==protocol['body_seed'])
    q,task,mask,action,valid=[t[indices].to('cuda') for t in d.tensors]
    q=q[:,:arm.dof]; action=action[:,:,:arm.dof]
    model=JointFlowPolicy().to('cuda').eval(); model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=True))
    original=overfit.predict
    for mode in ['euler50','zero_noise10','mean8_noise10']:
        generator=torch.Generator(device='cuda').manual_seed(4401)
        def diagnostic(model,kind,q,task,mask,noise,solver=10):
            if mode=='euler50':
                return original(model,kind,q,task,mask,noise,solver=50)
            if mode=='zero_noise10':
                return original(model,kind,q,task,mask,torch.zeros_like(noise))
            noises=[noise]+[torch.randn(noise.shape,device=noise.device,generator=generator) for _ in range(7)]
            return torch.stack([original(model,kind,q,task,mask,n) for n in noises]).mean(0)
        overfit.predict=diagnostic
        result=overfit.evaluate(model,'flow',q,task,action,arm,[accepted[i] for i in selected],{g['id']:g for g in d.manifest['goals']['train']},'cuda')
        (root/(mode+'.json')).write_text(json.dumps(result,indent=2))
        records=result['records'] if mode!='zero_noise10' else [r for r in result['records'] if r['noise_repeat']==0]
        print(mode,sum(r['success'] for r in records),'/',len(records),flush=True)
    overfit.predict=original

if __name__=='__main__':
    main()
