"""Matched held-history phase audit for all parent/candidate checkpoints."""
import json,shutil
from pathlib import Path
import numpy as np
import torch
from .training_estimator_repair import ROOT,setup,predict,analytical_targets,action_tensor
from .motion_transformer import MotionTransformer

torch.set_num_threads(4);_,models,_=setup();f=np.load(ROOT/'contexts.npz');ids=np.where(f['clean']&f['validation'])[0]
g=torch.tensor(f['g'][ids],device='cuda');c=torch.tensor(f['c'][ids],device='cuda');mask=torch.tensor(f['mask'][ids],device='cuda');error=torch.tensor(f['error'][ids],device='cuda');bias=torch.tensor(f['bias'][ids],device='cuda');dq=torch.tensor(f['dq'][ids],device='cuda');dx=torch.tensor(f['dx'][ids],device='cuda');target=analytical_targets(g,c,mask);action_target=action_tensor(target,error,bias,mask)
for seed in range(3):
    m=MotionTransformer().cuda();m.load_state_dict(torch.load(ROOT/f'seed_{seed}/policy.pt',map_location='cuda',weights_only=True));models[f'candidate_{seed}']=m.eval()
rows=[]
with torch.no_grad():
    for method,model in models.items():
        if model is None:continue
        xyz=[];action=[]
        for start in range(0,len(ids),256):
            sl=slice(start,start+256);p=predict(model,g[sl],c[sl],mask[sl]);a=action_tensor(p,error[sl],bias[sl],mask[sl]);response=torch.einsum('bnd,bn->bd',p,dq[sl])
            xyz.extend(torch.linalg.vector_norm(response[:,:3]-dx[sl,:3],dim=-1).cpu().tolist());action.extend(torch.sqrt(((a-action_target[sl]).square()*mask[sl]).sum(-1)/mask[sl].sum(-1)).cpu().tolist())
        xyz=np.array(xyz);action=np.array(action)
        for n in (6,7):
            for phase in sorted(set(f['phase'][ids])):
                subset=(f['mask'][ids].sum(-1)==n)&(f['phase'][ids]==phase)
                if subset.any():rows.append(dict(method=method,dof=n,phase=phase,n=int(subset.sum()),xyz_mm=float(xyz[subset].mean()*1000),action_mrad=None if phase=='probe' else float(action[subset].mean()*1000)))
(ROOT/'held_phase_metrics.json').write_text(json.dumps(rows,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
print('Saved',len(rows),'held phase/DOF/checkpoint groups')
