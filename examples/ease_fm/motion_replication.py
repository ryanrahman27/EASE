"""E31 fixed-protocol replication and context controls for E30."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import torch
import mujoco
from .motion_transformer import MotionTransformer, run, evaluate
from .paired_data import register
from .benchmark import fingerprint


def replicate(data, pilot, output, device='cuda'):
    torch.set_num_threads(4)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    manifest=register(data)
    protocol=dict(experiment='E31',seeds=[0,1,2],pilot=pilot,steps=15000,
        manifest_sha256=fingerprint(Path(data)/'manifest.json'),probes=[930000,940000,950000],
        modes=['neural_frozen','neural_online','hybrid_online','empty_neural_online','twin_neural_online'],
        scope='Reuse observed E30 seed0 and train seeds1/2 unchanged; no validation tuning. Random-network control uses one probe per model seed. Empty starts without calibration and learns statistics only during task execution.',
        test_evaluated=False)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    source=root/'source';source.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,source/p.name)
    index=[]
    for seed in (0,1,2):
        folder=Path(pilot) if seed==0 else root/f'seed_{seed}'
        if seed:run(data,str(folder),device=device,seed=seed)
        meta=json.loads((folder/'training.json').read_text())
        assert fingerprint(folder/'policy.pt')==meta['policy_sha256']
        model=MotionTransformer().to(device)
        model.load_state_dict(torch.load(folder/'policy.pt',map_location=device,weights_only=True));model.eval()
        records=[]
        with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
            for probe in protocol['probes']:
                for mode in protocol['modes']:
                    rows=evaluate(manifest,model,'validation',mode,device,probe)
                    if probe==930000 and mode in ('neural_frozen','neural_online','hybrid_online'):
                        assert rows==json.loads((folder/f'validation_{mode}.json').read_text()),'Primary replay changed'
                    records.extend(rows)
                    (root/f'validation_seed_{seed}.json').write_text(json.dumps(records,indent=2))
                    print('replicate',seed,probe,mode,sum(r['success'] for r in rows),'/',len(rows),flush=True)
            torch.manual_seed(20000+seed); random_model=MotionTransformer().to(device).eval()
            rows=evaluate(manifest,random_model,'validation','neural_online',device)
            for row in rows:row['mode']='random_neural_online'
            records.extend(rows)
            (root/f'validation_seed_{seed}.json').write_text(json.dumps(records,indent=2))
            print('random',seed,sum(r['success'] for r in rows),'/',len(rows),flush=True)
        index.append(dict(seed=seed,folder=str(folder),policy_sha256=meta['policy_sha256']))
        (root/'index.json').write_text(json.dumps(index,indent=2))
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('data','pilot','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--device',default='cuda')
    replicate(**vars(parser.parse_args()))
