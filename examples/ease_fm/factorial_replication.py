"""E27: fixed-data replication of E26, including calibration cost."""
import argparse
import json
import shutil
from pathlib import Path
import torch
from .benchmark import fingerprint
from .paired_data import register
from .paired_replication import ReactiveData
from .factorial_experiment import PrefixData
from .full_control import fit, add_recovery
from .paired_experiment import evaluate, gate


def run(data, pilot, output, device='cuda'):
    torch.set_num_threads(4)
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    pilot = Path(pilot); manifest = register(data)
    shared = json.loads((pilot/'shared_recovery.json').read_text())
    paths = [Path(p.replace('\\', '/')) for p in shared['files']]
    for path, digest in zip(paths, shared['files'].values()):
        assert fingerprint(path) == digest
    manifest_hash = fingerprint(Path(data)/'manifest.json')
    recovery_hash = fingerprint(pilot/'shared_recovery.json')
    protocol = dict(experiment='E27', seeds=[0, 1, 2], steps_per_model=15000,
        manifest_sha256=manifest_hash, recovery_sha256=recovery_hash,
        pilot=str(pilot), test_evaluated=False, calibration_seed=930000,
        scope='Seed 0 reused from E26; two new seeds for all four policies. Fixed recovery union; no tuning or filtering failed gates.',
        efficiency='Task steps capped at 150. Correct/twin history costs 32 extra steps per task, or one 32-step calibration reused over 12 goals per body/posture. Reactive/empty require zero calibration, regardless of ignored evaluator probes.',
        success_budgets=[25, 50, 100, 150, 182])
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2))
    snapshot = root/'source'; snapshot.mkdir()
    for path in Path(__file__).parent.glob('*.py'): shutil.copyfile(path, snapshot/path.name)
    index = []
    for seed in (0, 1, 2):
        for phase in ('original', 'shared_recovery'):
            for name, cls in [('reactive', ReactiveData), ('history', PrefixData)]:
                folder = pilot/f'{phase}_{name}' if seed == 0 else root/f'seed_{seed}_{phase}_{name}'
                if seed:
                    dataset = cls(data)
                    if phase == 'shared_recovery': dataset = add_recovery(dataset, paths)
                    model = fit(dataset, folder, 'direct', 15000, device, manifest_hash,
                                recovery_hash if phase == 'shared_recovery' else None, seed=seed)
                    for split in ('train', 'validation'):
                        for mode in (['reactive'] if name == 'reactive' else ['correct', 'empty', 'twin']):
                            records = evaluate(data, manifest, model, split, mode, device)
                            (folder/f'{split}_{mode}.json').write_text(json.dumps(records, indent=2))
                            print(seed, phase, name, split, mode, gate(records), flush=True)
                meta = json.loads((folder/'training.json').read_text())
                assert meta['seed'] == seed and meta['steps'] == 15000
                assert meta['manifest_sha256'] == manifest_hash
                assert meta['recovery_sha256'] == (recovery_hash if phase == 'shared_recovery' else None)
                assert fingerprint(folder/'policy.pt') == meta['policy_sha256']
                index.append(dict(seed=seed, phase=phase, model=name, folder=str(folder), policy_sha256=meta['policy_sha256']))
                (root/'index.json').write_text(json.dumps(index, indent=2))
    assert fingerprint(pilot/'shared_recovery.json') == recovery_hash
    assert all(fingerprint(p) == h for p, h in zip(paths, shared['files'].values()))
    (root/'status.json').write_text(json.dumps(dict(status='complete', test_evaluated=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('data', 'pilot', 'output'): parser.add_argument('--'+key, required=True)
    parser.add_argument('--device', default='cuda')
    run(**vars(parser.parse_args()))
