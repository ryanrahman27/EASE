"""Post-training diagnostic: does lower response loss merely suppress motion?"""
import argparse
import json
from pathlib import Path
import torch
from .control_training import batch,predict,response
from .motion_transformer import MotionTransformer
from .benchmark import fingerprint


def run(training):
    torch.set_num_threads(4);root=Path(training)
    protocol=json.loads((root/'protocol.json').read_text());parents=list(protocol['parents'])
    g,c,m,t,j=batch(512,'cuda',torch.Generator(device='cuda').manual_seed(52999))
    ideal=j.transpose(1,2)@response(j)
    rows=[]
    for seed in range(3):
        for condition in ['original','prediction','control']:
            path=Path(parents[seed]) if condition=='original' else root/condition/f'seed_{seed}'/'policy.pt'
            before=fingerprint(path)
            model=MotionTransformer().cuda();model.load_state_dict(torch.load(path,map_location='cuda',weights_only=True));model.eval()
            with torch.no_grad():
                prediction=predict(model,g,c,m);operator=response(prediction)
                achieved=j.transpose(1,2)@operator
                for dof in [6,7]:
                    selected=m.sum(-1)==dof
                    def avg(x):return float(x[selected].mean())
                    rows.append(dict(condition=condition,seed=seed,dof=dof,count=int(selected.sum()),
                        jacobian_norm_ratio=avg(prediction.norm(dim=(1,2))/j.norm(dim=(1,2))),
                        action_operator_norm_ratio=avg(operator.norm(dim=(1,2))/response(j).norm(dim=(1,2))),
                        response_trace=avg(achieved.diagonal(dim1=1,dim2=2).sum(-1)),
                        ideal_response_trace=avg(ideal.diagonal(dim1=1,dim2=2).sum(-1)),
                        response_mse=avg((achieved-ideal).square().mean(dim=(1,2)))))
            assert fingerprint(path)==before
    (root/'response_diagnostic.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--training',required=True);run(**vars(p.parse_args()))
