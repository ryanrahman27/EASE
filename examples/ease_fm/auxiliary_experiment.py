"""Prespecified, capacity-matched auxiliary response pilot and diagnostic."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .benchmark import fingerprint
from .joint_policy import JointFlowPolicy
from .mixed_data import load
from .mixed_experiment import train, evaluate
from .mixed_sim import Arm, MixedReach, explore


def response_probe(data, folder, device):
    metadata = json.loads((folder/'training.json').read_text())
    model = JointFlowPolicy(history=True, auxiliary=True).to(device)
    model.load_state_dict(torch.load(folder/'policy.pt', map_location=device, weights_only=True))
    model.eval()
    records = []
    manifest = load(data)
    for item in manifest['arms']:
        if item['split'] not in {'validation', 'validation_topology'}:
            continue
        arm = Arm(**item)
        donor = next(Arm(**a) for a in manifest['arms'] if a['split'] == 'train' and len(a['lengths']) == arm.dof)
        for repeat in range(8):
            # Independent query and calibration episodes; query transitions never enter context.
            query = explore(MixedReach(arm), 810000+repeat, 32)
            joints = torch.tensor(query[:, :, :2], device=device)
            commands = torch.tensor(query[:, :, 2], device=device)
            target = torch.tensor(query[:, 0, 8:11], device=device)
            mask = torch.ones((len(query), arm.dof), dtype=torch.bool, device=device)
            for mode, body, length in [('empty', arm, 0), ('correct32', arm, 32), ('wrong32', donor, 32)]:
                context = np.zeros((32, arm.dof, 11), np.float32)
                if length:
                    context[:] = explore(MixedReach(body), 820000+repeat, 32)
                with torch.no_grad():
                    z = model.encode(torch.tensor(context, device=device)[None].expand(len(query), -1, -1, -1),
                                     torch.full((len(query),), length, device=device), mask)
                    prediction = model.response(joints, commands, mask, z)*model.response_scale
                records.append({'family': arm.family, 'body_seed': arm.seed, 'repeat': repeat, 'mode': mode,
                    'mse_m2': float((prediction-target).square().mean()),
                    'zero_prediction_mse_m2': float(target.square().mean())})
    (folder/'response_probe.json').write_text(json.dumps({'protocol': 'Independent random-excitation query episodes; no updates; 32 transitions per episode; MSE averaged over XYZ',
        'checkpoint_sha256': metadata['policy_sha256'], 'records': records}, indent=2))


def summarize(root):
    rows = []
    for name in ['flow_only', 'flow_response']:
        evaluation = json.loads((root/name/'validation.json').read_text())
        probe = json.loads((root/name/'response_probe.json').read_text())
        for family in ['six', 'seven', 'six_novel']:
            for mode in ['empty', 'correct8', 'correct32', 'wrong32']:
                cases = [r for r in evaluation['records'] if r['family'] == family and r['mode'] == mode]
                diagnostic = [r for r in probe['records'] if r['family'] == family and r['mode'] == mode]
                row = {'variant': name, 'family': family, 'mode': mode,
                    'successes': sum(r['success'] for r in cases), 'trials': len(cases),
                    'mean_final_error_m': float(np.mean([r['error'] for r in cases]))}
                if diagnostic:
                    row['response_rmse_m'] = float(np.sqrt(np.mean([r['mse_m2'] for r in diagnostic])))
                    row['zero_response_rmse_m'] = float(np.sqrt(np.mean([r['zero_prediction_mse_m2'] for r in diagnostic])))
                rows.append(row)
    (root/'summary.json').write_text(json.dumps(rows, indent=2))
    lines = ['# Auxiliary response pilot', '', 'Single training seed, one validation body per family. Descriptive results only.', '',
             '| Variant | Family | Context | Success | Final error (m) | Response RMSE (m) |',
             '|---|---|---|---:|---:|---:|']
    for r in rows:
        error = f"{r['response_rmse_m']:.6f}" if 'response_rmse_m' in r else '-'
        lines.append(f"| {r['variant']} | {r['family']} | {r['mode']} | {r['successes']}/{r['trials']} | {r['mean_final_error_m']:.4f} | {error} |")
    (root/'summary.md').write_text('\n'.join(lines)+'\n')


def pilot(data, output, steps=10000, device='cuda'):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    protocol = {'seed': 0, 'steps': steps, 'batch_size': 32, 'weights': {'flow_only': 0., 'flow_response': .1},
        'primary': 'correct32 minus wrong32 success on held-out six-DoF validation arm',
        'secondary': 'novel-axis success and independent measured-response prediction',
        'normalization': 'Per-axis response RMS from accepted training transitions only, floor 1e-4 m',
        'controls': 'Identical architecture, initial weights, batch/noise/time/context random streams, budgets and evaluation cases',
        'loss': 'masked flow MSE + weight * normalized measured delta-EE MSE',
        'context': 'Same training body, different episode; no goal in history or response head',
        'scope': 'Single-seed development ablation, no hyperparameter sweep; final test untouched',
        'manifest_sha256': fingerprint(Path(data)/'manifest.json'), 'device': device}
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2))
    for name, weight in protocol['weights'].items():
        folder = root/name
        train(data, folder, history=True, seed=0, steps=steps, device=device, auxiliary_weight=weight)
        evaluate(data, folder/'policy.pt', folder/'validation.json', device=device)
        response_probe(data, folder, device)
    summarize(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--device', default='cuda')
    pilot(**vars(parser.parse_args()))
