"""Post-hoc history interventions for the frozen E12 direct policy on training tasks."""
import json
from pathlib import Path
import shutil
import torch
from .benchmark import fingerprint
from .mixed_data import MixedData
from .joint_policy import JointFlowPolicy
from .full_control import rollout


def main():
    torch.set_num_threads(4)
    root=Path('outputs/ease_full_control_v1')
    folder=root/'recovery_direct'
    meta=json.loads((folder/'training.json').read_text())
    if fingerprint(folder/'policy.pt')!=meta['policy_sha256']:
        raise ValueError('Checkpoint mismatch')
    with (root/'history_probe_protocol.json').open('x') as f:
        json.dump({'scope':'Post-hoc training-body history diagnostic; no new training or test-body policy evaluation',
            'checkpoint_sha256':meta['policy_sha256'],'modes':['empty','wrong32'],
            'reference':'recovery_direct/correct32.json','donor':'First different training body with same joint count',
            'paired':'Same tasks, resets, checkpoint and calibration seeds as correct32; only context intervention changes',
            'warning':'Seen-body result cannot establish adaptation to new kinematics'},f,indent=2)
    shutil.copyfile(__file__,root/'history_probe_source.py')
    dataset=MixedData('outputs/ease_mixed_sync_v2')
    if meta['manifest_sha256']!=fingerprint(Path('outputs/ease_mixed_sync_v2/manifest.json')):
        raise ValueError('Dataset mismatch')
    model=JointFlowPolicy(history=True).to('cuda').eval()
    model.load_state_dict(torch.load(folder/'policy.pt',map_location='cuda',weights_only=True))
    for mode in ['empty','wrong32']:
        records=rollout(dataset,model,'direct',folder,'cuda',mode=mode)
        print(mode,sum(r['success'] for r in records),'/',len(records),flush=True)


if __name__=='__main__':
    main()
