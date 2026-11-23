"""Post-training control regression on the original training arms."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .benchmark import fingerprint
from .motion_transformer import MotionTransformer,evaluate
from .paired_data import register


def run(data,models,output):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    manifest=register(data);solve=np.linalg.solve;summary=[]
    def guarded(a,b):
        assert a.shape[-2:]==(3,3)
        return solve(a,b)
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')),patch.object(np.linalg,'solve',side_effect=guarded),patch.object(torch.linalg,'solve',side_effect=RuntimeError('Neural inverse forbidden')):
        for seed in range(3):
            folder=Path(models)/f'seed_{seed}';meta=json.loads((folder/'training.json').read_text())
            assert fingerprint(folder/'policy.pt')==meta['policy_sha256']
            m=MotionTransformer().cuda();m.load_state_dict(torch.load(folder/'policy.pt',map_location='cuda',weights_only=True));m.eval()
            rows=evaluate(manifest,m,'train','empty_neural_online','cuda')
            (root/f'seed_{seed}.json').write_text(json.dumps(rows,indent=2))
            summary.append(dict(seed=seed,success=sum(r['success'] for r in rows),count=len(rows),policy_sha256=meta['policy_sha256'],records_sha256=fingerprint(root/f'seed_{seed}.json')))
            print(summary[-1],flush=True)
    (root/'audit.json').write_text(json.dumps(dict(summary=summary,manifest_sha256=fingerprint(Path(data)/'manifest.json'),test_evaluated=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('data','models','output'):p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))
