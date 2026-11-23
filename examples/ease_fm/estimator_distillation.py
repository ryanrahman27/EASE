"""E66: isolate geometric versus measured analytical targets on procedural history."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .benchmark import fingerprint
from .motion_transformer import MotionTransformer
from .control_training import batch,predict
from .coherent_pose_training import load_contexts

ROOT=Path('outputs/ease_estimator_distillation_v1')
DATA=Path('outputs/ease_coherent_pose_data_v1')
PARENTS=Path('outputs/ease_robust_training_v1')


def analytical_targets(g,c,mask):
    # Padded joints have zero Gram entries; fill only their unused diagonal.
    safe=g.double()+torch.diag_embed((~mask).double())
    return torch.linalg.solve(safe,c.double()).to(g.dtype).masked_fill(~mask[:,:,None],0)


def fit_anchor(prediction,target,mask,parent):
    fit=((prediction-target).square()*mask[:,:,None]).sum()/(6*mask.sum())
    weight=torch.where(mask.sum(-1)==6,2.,.5)[:,None,None]
    anchor=((prediction-parent).square()*mask[:,:,None]*weight).sum()/(6*mask.sum())
    return fit,anchor


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir()
    hashes={str(PARENTS/f'seed_{s}'/'policy.pt'):fingerprint(PARENTS/f'seed_{s}'/'policy.pt') for s in range(3)}
    protocol=dict(experiment='E66',steps=2000,batch=128,lr=2e-5,seeds=[0,1,2],conditions=['geometric','analytical'],parents=hashes,
        contexts_sha256=fingerprint(DATA/'contexts.npz'),provenance_sha256=fingerprint(DATA/'provenance.json'),
        design='Matched final-checkpoint fine tuning from E35. Each batch has 64 synthetic E52 contexts plus 64 measured procedural contexts, restricted to analytical/original_0/original_1 training trajectories. Original_2 trajectories are validation (same eight bodies and goals). 3600 training, 1200 validation rows. Identical batches and optimizer budgets. Only measured target changes: current geometric Jacobian versus analytical solve of the observed Gram/cross statistics. Synthetic clean-ridge labels unchanged. No native data, response fit, control surrogate or privileged information at inference.',
        loss='Masked six-output mean squared fit plus fixed E35 anchor (six-joint weight2, seven-joint .5). AdamW lr2e-5, gradient norm clip1, 2000 updates per condition/seed. No best-checkpoint or seed selection. Analytical targets generated in float64 with padded inactive diagonals filled; network inference remains inverse-free.',
        evaluation='First all original 256 reaching tasks per checkpoint with existing bias correction. Then all twelve observed native common schedules and all twelve contact-enabled manipulation cases for all candidates, regardless of failures, reported diagnostically. Full six-/seven-joint retention required before promotion. Fixed E63/E46 E35 and analytical baselines; no final test.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for name in ['estimator_distillation.py','control_training.py','coherent_pose_training.py','motion_transformer.py','joint_policy.py']:
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    values,_,_=load_contexts(DATA,'cuda')
    provenance=json.loads((DATA/'provenance.json').read_text())
    selected=torch.tensor([i for i,r in enumerate(provenance) if r['driver'] in ['analytical','original_0','original_1','original_2']],device='cuda')
    values={k:v[selected] for k,v in values.items()}
    train=torch.where(~values['validation'])[0];val=torch.where(values['validation'])[0]
    assert len(train)==3600 and len(val)==1200
    values['analytical']=analytical_targets(values['g'],values['c'],values['mask'])
    # Independent NumPy solve on every active joint block verifies padding/dtype.
    max_error=0.
    for i in range(len(selected)):
        n=int(values['mask'][i].sum());g=values['g'][i,:n,:n].cpu().numpy().astype(float);c=values['c'][i,:n].cpu().numpy().astype(float)
        actual=values['analytical'][i,:n].cpu().numpy();expected=np.linalg.solve(g,c)
        np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5);max_error=max(max_error,float(np.max(np.abs(actual-expected))))
    (ROOT/'target_audit.json').write_text(json.dumps(dict(rows=4800,max_absolute_error=max_error,train=3600,validation=1200,validation_scope='trajectory holdout on same bodies',test_evaluated=False),indent=2))
    held=val[torch.randperm(len(val),device='cuda',generator=torch.Generator(device='cuda').manual_seed(66999))[:512]]
    validation=batch(512,'cuda',torch.Generator(device='cuda').manual_seed(52999))
    for seed in range(3):
        teacher=MotionTransformer().cuda();teacher.load_state_dict(torch.load(PARENTS/f'seed_{seed}'/'policy.pt',weights_only=True,map_location='cuda'));teacher.eval()
        for p in teacher.parameters():p.requires_grad_(False)
        models={};opts={};hist={k:[] for k in ['geometric','analytical']}
        for condition in hist:
            model=MotionTransformer().cuda();model.load_state_dict(teacher.state_dict());models[condition]=model
            opts[condition]=torch.optim.AdamW(model.parameters(),lr=2e-5)
        def metrics(model):
            with torch.no_grad():
                p=predict(model,values['g'][held],values['c'][held],values['mask'][held]);m=values['mask'][held]
                scores={k:float(((p-values[k][held]).square()*m[:,:,None]).sum()/(6*m.sum())) for k in ['j','analytical']}
                g,c,m,t,_=validation;scores['synthetic']=float(((predict(model,g,c,m)-t).square()*m[:,:,None]).sum()/(6*m.sum()))
                return scores
        initial=metrics(teacher);gen=torch.Generator(device='cuda').manual_seed(52000+seed);rows_gen=torch.Generator(device='cuda').manual_seed(66000+seed);digest=hashlib.sha256()
        for update in range(2000):
            indices=train[torch.randint(len(train),(64,),device='cuda',generator=rows_gen)]
            sg,sc,sm,st,_=batch(64,'cuda',gen)
            g=torch.cat([sg,values['g'][indices]]);c=torch.cat([sc,values['c'][indices]]);m=torch.cat([sm,values['mask'][indices]])
            for x in [indices,g,c,m,st]:digest.update(x.detach().cpu().numpy().tobytes())
            with torch.no_grad():parent=predict(teacher,g,c,m)
            for condition,model in models.items():
                target=torch.cat([st,values['j' if condition=='geometric' else 'analytical'][indices]])
                fit,anchor=fit_anchor(predict(model,g,c,m),target,m,parent);loss=fit+anchor
                assert torch.isfinite(loss)
                opts[condition].zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opts[condition].step()
                hist[condition].append([float(fit.detach()),float(anchor.detach())])
            if (update+1)%250==0:print(seed,update+1,{k:v[-1] for k,v in hist.items()},flush=True)
        for condition,model in models.items():
            folder=ROOT/condition/f'seed_{seed}';folder.mkdir(parents=True);torch.save(model.state_dict(),folder/'policy.pt')
            meta=dict(seed=seed,steps=2000,policy_sha256=fingerprint(folder/'policy.pt'),parent_sha256=hashes[str(PARENTS/f'seed_{seed}'/'policy.pt')],batch_sha256=digest.hexdigest(),initial=initial,final=metrics(model),losses=hist[condition])
            (folder/'training.json').write_text(json.dumps(meta,indent=2));print('complete',condition,seed,meta['final'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (ROOT/'status.json').write_text(json.dumps(dict(status='trained',test_evaluated=False)))


if __name__=='__main__':run()
