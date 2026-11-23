"""E24: retain all seeds, directly train reactive policies, vary probe seeds."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .benchmark import fingerprint
from .mixed_data import MixedData
from .paired_data import register
from .paired_experiment import run as paired_run, evaluate, gate
from .full_control import fit
from .joint_policy import JointFlowPolicy
from .pose_policy import PosePolicy


class ReactiveData(MixedData):
    """Same supervised action rows; no calibration or body input to policy."""
    def __init__(self, data):
        super().__init__(data)
        self.contexts = [np.zeros((32, 7, 11), np.float32)]

    def histories(self, indices, rng):
        return torch.zeros((len(indices), 32, 7, 11)), torch.zeros(len(indices), dtype=torch.long)


def run(data, pilot, output, device='cuda'):
    torch.set_num_threads(4)
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    manifest = register(data)
    protocol = {'experiment': 'E24', 'seeds': [0, 1, 2], 'pilot': pilot,
        'manifest_sha256': fingerprint(Path(data)/'manifest.json'),
        'teacher_steps': 10000, 'encoder_steps': 5000, 'reactive_steps': 15000,
        'probe_seeds': [930000, 940000, 950000], 'final_test_evaluated': False,
        'gate_policy': 'Retain every failed gate. Do not fit students after failed teacher gate. Reactive validation is diagnostic when its training gate fails.',
        'reactive_design': 'Same direct Transformer action trunk, learned constant empty latent, no history/body/initial-pose context. All action-producing weights trained on supervised action chunks.',
        'budget': 'Reactive 15k updates equals teacher 10k plus one encoder 5k in update count, not FLOPs; same batch32 AdamW lr0.0003.',
        'capacity': 'Same allocated 244289 parameters; history input/spatial modules have no influence at length zero. Temporal/output modules process the learned empty token, yielding a trainable constant. Effective trainable capacity differs.',
        'replication': 'Seed0 reuses immutable pilot. Seeds1/2 fresh teacher and encoder initialization/sampling. Fixed dataset; no tuning.'}
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2))
    src = root/'source'; src.mkdir()
    for p in Path(__file__).parent.glob('*.py'): shutil.copyfile(p, src/p.name)
    index = []
    for seed in protocol['seeds']:
        run_root = Path(pilot) if seed == 0 else root/f'seed_{seed}'
        if seed:
            paired_run(data, str(run_root), device=device, seed=seed)
        teacher_gate = json.loads((run_root/'teacher'/'gate.json').read_text())
        folder = root/f'reactive_{seed}'
        model = fit(ReactiveData(data), folder, 'direct', 15000, device, protocol['manifest_sha256'], seed=seed)
        for split in ('train', 'validation'):
            rows = evaluate(data, manifest, model, split, 'reactive', device)
            (folder/f'{split}.json').write_text(json.dumps(rows, indent=2))
            if split == 'train':
                reactive_gate = gate(rows)
                (folder/'gate.json').write_text(json.dumps(reactive_gate, indent=2))
            print('REACTIVE', seed, split, gate(rows), flush=True)
        checks = []
        if teacher_gate['passed']:
            for name, cls in [('history', JointFlowPolicy), ('pose', PosePolicy)]:
                checkpoint = run_root/name/'policy.pt'
                metadata = json.loads((run_root/name/'training.json').read_text())
                assert fingerprint(checkpoint) == metadata['policy_sha256']
                model = cls(history=True).to(device)
                model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)); model.eval()
                for probe in protocol['probe_seeds'][1:]:
                    for mode in ('correct', 'empty', 'twin'):
                        rows = evaluate(data, manifest, model, 'validation', mode, device, calibration_seed=probe)
                        filename = f'seed_{seed}_{name}_{probe}_{mode}.json'
                        (root/filename).write_text(json.dumps(rows, indent=2))
                        checks.append({'model': name, 'probe': probe, 'mode': mode, 'file': filename})
                        print('PROBE', seed, name, probe, mode, gate(rows)['overall'], flush=True)
        index.append({'seed': seed, 'run': str(run_root), 'reactive': str(folder),
            'teacher_gate': teacher_gate, 'reactive_gate': reactive_gate, 'extra_probes': checks})
        (root/'index.json').write_text(json.dumps(index, indent=2))
        print('COMPLETED SEED', seed, flush=True)
    (root/'status.json').write_text(json.dumps({'status': 'complete', 'test_evaluated': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('data', 'pilot', 'output'): parser.add_argument('--'+key, required=True)
    parser.add_argument('--device', default='cuda')
    run(**vars(parser.parse_args()))
