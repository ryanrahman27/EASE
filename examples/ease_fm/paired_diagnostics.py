"""E25 frozen privileged transfer, history interventions, and reactive fitting."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .benchmark import fingerprint
from .paired_data import register
from .paired_experiment import evaluate
from .history_audit import intervene
from .joint_policy import JointFlowPolicy
from .mixed_sim import Arm
from .diverse_oracle import oracle_tokens
from .full_control import commands


@torch.no_grad()
def fitting(data, manifest, model, privileged, device):
    arms = {a['seed']: Arm(**a) for a in manifest['arms']}
    errors = {'initial': [], 'later': []}; targets = {}
    for episode in manifest['episodes']:
        arm = arms[episode['body_seed']]
        with np.load(Path(data)/episode['file']) as values:
            n = len(values['action']); q = np.zeros((n, 7, 2), np.float32)
            q[:, :arm.dof] = values['joints']
            mask = torch.tensor([[j < arm.dof for j in range(7)]], device=device)
            h = oracle_tokens(arm, padded=True) if privileged else np.zeros((32, 7, 11), np.float32)
            z = model.encode(torch.tensor(h[None], device=device), torch.tensor([32 if privileged else 0], device=device), mask)
            predictions = []
            for start in range(0, n, 128):
                end = min(start+128, n); count = end-start
                pred = commands(model, 'direct', torch.tensor(q[start:end], device=device),
                    torch.tensor(values['task'][start:end], device=device), mask.expand(count, -1),
                    torch.zeros((count, 16, 7), device=device), z.expand(count, -1, -1))
                predictions.append(pred[:, :arm.dof].cpu().numpy())
            error = np.concatenate(predictions)-values['action']
            errors['initial'].extend(error[:1].ravel().tolist())
            errors['later'].extend(error[1:].ravel().tolist())
            targets[(arm.seed, episode['goal_id'])] = (values['joints'][0].copy(), values['task'][0].copy(), values['action'][0].copy())
    conflict = []
    for pair in manifest['pairs']:
        if pair['split'] != 'train': continue
        a, b = pair['bodies']
        for goal in manifest['goals']['train']:
            if goal['pair'] != pair['id']: continue
            left, right = targets[(a, goal['id'])], targets[(b, goal['id'])]
            assert np.array_equal(left[0], right[0]) and np.array_equal(left[1], right[1])
            conflict.extend((left[2]-right[2]).tolist())
    return {'action_error': {band: {'joint_samples': len(v), 'mae_rad': float(np.abs(v).mean()),
                'rmse_rad': float(np.sqrt(np.square(v).mean()))} for band, v in errors.items()},
            'initial_twin_target_rms_difference_rad': float(np.sqrt(np.square(conflict).mean())),
            'minimum_initial_deterministic_action_mse_rad2': float(np.square(conflict).mean()/4)}


def run(data, replication, output, device='cuda'):
    torch.set_num_threads(4)
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    manifest = register(data); entries = json.loads((Path(replication)/'index.json').read_text())
    modes = ['correct32', 'shuffle_actions', 'shuffle_transitions', 'static_only', 'repeat_initial_static']
    protocol = {'experiment': 'E25', 'seeds': [e['seed'] for e in entries], 'modes': modes,
        'probe_seeds': [930000, 940000, 950000], 'manifest_sha256': fingerprint(Path(data)/'manifest.json'),
        'weights_updated': False, 'test_evaluated': False,
        'interventions': 'Existing E20 transforms: fixed action permutation; full transition permutation; q/ee trajectory only; repeated first q/ee',
        'limitations': 'Frozen input interventions may be out of distribution. Pose trajectory contains motion information. Privileged controller is not a formal upper bound.',
        'fitting': 'Original training expert states, first-action errors; identical twin initial inputs imply a deterministic supervised MSE floor, not a control impossibility proof.'}
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2))
    src = root/'source'; src.mkdir()
    for p in Path(__file__).parent.glob('*.py'): shutil.copyfile(p, src/p.name)
    summary = []; audits = []; checkpoints = {}
    dofs = {a['seed']: len(a['lengths']) for a in manifest['arms']}
    for entry in entries:
        seed = entry['seed']; run_root = Path(entry['run'])
        for name in ('teacher', 'reactive', 'history'):
            folder = Path(entry['reactive']) if name == 'reactive' else run_root/name
            meta = json.loads((folder/'training.json').read_text())
            path = folder/'policy.pt'; digest = fingerprint(path)
            assert digest == meta['policy_sha256']; checkpoints[str(path)] = digest
            model = JointFlowPolicy(history=True).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True)); model.eval()
            if name in ('teacher', 'reactive'):
                audits.append({'seed': seed, 'model': name, **fitting(data, manifest, model, name == 'teacher', device)})
                (root/'fitting.json').write_text(json.dumps(audits, indent=2))
            if name == 'reactive': continue
            probes = [None] if name == 'teacher' else protocol['probe_seeds']
            for probe in probes:
                for mode in (['oracle'] if name == 'teacher' else modes):
                    rows = evaluate(data, manifest, model, 'validation', mode, device,
                        calibration_seed=930000 if probe is None else probe,
                        intervention=None if name == 'teacher' else intervene)
                    if mode == 'correct32':
                        reference = run_root/'history'/'validation_correct.json' if probe == 930000 else Path(replication)/next(
                            r['file'] for r in entry['extra_probes'] if (r['model'], r['probe'], r['mode']) == ('history', probe, 'correct'))
                        old = {(r['body_seed'], r['goal_id']): r for r in json.loads(reference.read_text())}
                        assert all(all(r[k] == old[(r['body_seed'], r['goal_id'])][k] for k in ('success', 'collision', 'error', 'steps')) for r in rows)
                    filename = f'{seed}_{name}_{probe}_{mode}.json'
                    (root/filename).write_text(json.dumps(rows, indent=2))
                    row = {'seed': seed, 'model': name, 'probe': probe, 'mode': mode, 'file': filename,
                        'success': sum(r['success'] for r in rows), 'trials': len(rows),
                        'six_success': sum(r['success'] for r in rows if dofs[r['body_seed']] == 6),
                        'seven_success': sum(r['success'] for r in rows if dofs[r['body_seed']] == 7),
                        'collisions': sum(r['collision'] for r in rows)}
                    summary.append(row); print(row, flush=True)
                    (root/'summary.json').write_text(json.dumps(summary, indent=2))
            assert fingerprint(path) == digest
    assert all(fingerprint(p) == h for p, h in checkpoints.items())
    (root/'verification.json').write_text(json.dumps({'checkpoint_hashes': checkpoints, 'intact_replays_exact': True,
        'test_evaluated': False, 'records': sum(r['trials'] for r in summary)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('data', 'replication', 'output'): parser.add_argument('--'+name, required=True)
    parser.add_argument('--device', default='cuda'); run(**vars(parser.parse_args()))
