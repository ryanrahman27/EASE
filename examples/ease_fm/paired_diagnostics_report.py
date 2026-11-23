"""Summarize all frozen diagnostics without dropping seeds or interventions."""
import argparse
import json
from pathlib import Path
import numpy as np
from .benchmark import fingerprint
from .paired_data import register


def report(output, data=None):
    root = Path(output)
    rows = json.loads((root/'summary.json').read_text())
    verification = json.loads((root/'verification.json').read_text())
    assert len(rows) == 48 and verification['records'] == 4608
    assert all(fingerprint(p) == h for p, h in verification['checkpoint_hashes'].items())
    lines = ['# E25: frozen paired-arm diagnostics', '',
        '| Seed | Privileged six-joint | Privileged seven-joint |', '|---|---:|---:|']
    for seed in range(3):
        row = next(r for r in rows if r['seed'] == seed and r['model'] == 'teacher')
        lines.append(f"| {seed} | {row['six_success']}/48 | {row['seven_success']}/48 |")
    modes = ['correct32', 'shuffle_actions', 'shuffle_transitions', 'static_only', 'repeat_initial_static']
    lines += ['', '| Seed | Probe | Intact | Shuffled actions | Shuffled transitions | Pose trajectory | Initial pose |',
              '|---|---|---:|---:|---:|---:|---:|']
    for seed in range(3):
        for probe in (930000, 940000, 950000):
            selected = [next(r for r in rows if (r['seed'], r['probe'], r['mode']) == (seed, probe, mode)) for mode in modes]
            lines.append(f'| {seed} | {probe} | '+' | '.join(f"{r['success']}/96" for r in selected)+' |')
    fitting = json.loads((root/'fitting.json').read_text())
    lines += ['', '| Seed | Model | Initial action RMSE (rad) | Later expert-state RMSE (rad) |', '|---|---|---:|---:|']
    for row in fitting:
        lines.append(f"| {row['seed']} | {row['model']} | {row['action_error']['initial']['rmse_rad']:.6f} | {row['action_error']['later']['rmse_rad']:.6f} |")
    floor = fitting[0]['minimum_initial_deterministic_action_mse_rad2']
    lines += ['', f'Twin initial-input deterministic squared-action-error floor: {floor:.9g} rad² per valid joint.',
        'This is a supervised first-action ambiguity bound under equal twin weighting, not a lower bound on achievable task success.',
        'Frozen intervention inputs may be out of distribution. Privileged teachers are diagnostics, not performance upper bounds.',
        'All intact replays match prior outcomes exactly. Checkpoints unchanged; final test not evaluated.']
    if data:
        manifest = register(data); coverage = []
        for dof in (6, 7):
            states = []
            for episode in manifest['episodes']:
                if episode['dof'] != dof: continue
                with np.load(Path(data)/episode['file']) as values:
                    states.append(values['joints'][:, :, 0].copy())
            q = np.concatenate(states); lower, upper = q.min(0), q.max(0)
            homes = [a['home'] for a in manifest['arms'] if a['split'] == 'validation' and len(a['lengths']) == dof]
            outside = sorted(set(int(j)+1 for h in homes for j in np.flatnonzero((np.asarray(h) < lower-1e-6) | (np.asarray(h) > upper+1e-6))))
            coverage.append({'dof': dof, 'training_q_min': lower.tolist(), 'training_q_max': upper.tolist(),
                             'validation_homes': homes, 'out_of_range_joints_one_based': outside})
        (root/'reset_coverage.json').write_text(json.dumps(coverage, indent=2))
        lines += ['', '## Reset-state coverage', '',
            'Post-hoc diagnostic: training demonstration joint ranges versus validation home positions (tolerance 1e-6 rad).']
        for row in coverage:
            lines.append(f"- {row['dof']}-joint validation: out-of-range joints {row['out_of_range_joints_one_based']}.")
        lines += ['The six-joint split changes both kinematics and reset posture; this confounds interpreting failure as kinematic transfer alone.']
    (root/'report.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True); parser.add_argument('--data')
    report(**vars(parser.parse_args()))
