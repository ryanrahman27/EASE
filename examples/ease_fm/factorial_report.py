"""Verify the E26 recovery union and all factorial evaluation conditions."""
import argparse
import json
from pathlib import Path
import numpy as np
from .benchmark import fingerprint
from .paired_experiment import gate
from .paired_data import register


def read(path): return json.loads(Path(path).read_text())


def report(data, output):
    root = Path(output); manifest = register(data)
    protocol = read(root/'protocol.json'); shared = read(root/'shared_recovery.json')
    assert read(root/'status.json')['status'] == 'complete'
    assert protocol['manifest_sha256'] == fingerprint(Path(data)/'manifest.json')
    arms = {a['seed']: a for a in manifest['arms']}
    episodes = manifest['episodes']; recovered = 0
    for path, digest in shared['files'].items():
        path = Path(path); assert fingerprint(path) == digest
        metadata = read(path.parent/'queries.json'); assert metadata['sha256'] == digest
        assert sum(not q['branch_collision'] for q in metadata['queries']) == metadata['accepted']
        assert all(episodes[q['episode']]['body_seed'] == q['body'] for q in metadata['queries'])
        with np.load(path) as values:
            assert len(values['episode']) == metadata['accepted']
            assert all(0 <= int(i) < len(episodes) and episodes[int(i)]['split'] == 'train' for i in values['episode'])
            assert np.isfinite(values['action']).all() and np.isfinite(values['joints']).all()
            assert np.all(values['valid'] <= values['mask'][:, None, :])
            assert np.all(values['action'][values['valid'] == 0] == 0)
            expected_mask = np.array([[j < episodes[int(i)]['dof'] for j in range(7)] for i in values['episode']])
            np.testing.assert_array_equal(values['mask'], expected_mask)
            recovered += len(values['episode'])
    assert all(fingerprint(p) == h for p, h in shared['source_checkpoints'].items())
    conditions = ['seen_axes_seen_posture', 'new_axes_seen_posture', 'seen_axes_shifted_posture', 'new_axes_shifted_posture']
    rows = []
    def summarize(records, **identity):
        assert len(records) == 384
        assert len({(r['body_seed'], r['goal_id']) for r in records}) == 384
        for condition in conditions:
            selected = [r for r in records if r['condition'] == condition]
            assert len(selected) == 96
            rows.append({**identity, 'condition': condition, 'success': sum(r['success'] for r in selected),
                'trials': 96, 'six_success': sum(r['success'] for r in selected if len(arms[r['body_seed']]['lengths']) == 6),
                'seven_success': sum(r['success'] for r in selected if len(arms[r['body_seed']]['lengths']) == 7),
                'collisions': sum(r['collision'] for r in selected)})
    lines = ['# E26: crossed kinematics/posture and shared recovery', '',
             f'Recovery union: {recovered} accepted expert chunks from two frozen source policies.', '',
             'Single fresh training seed. Each action policy receives 15,000 updates. No final-test rollout.', '',
             '| Data | Policy | Training success | Training gate |', '|---|---|---:|---|']
    summary = read(root/'summary.json')
    for phase in ('original', 'shared_recovery'):
        for name in ('reactive', 'history'):
            folder = root/f'{phase}_{name}'; meta = read(folder/'training.json')
            assert fingerprint(folder/'policy.pt') == meta['policy_sha256']
            assert meta['manifest_sha256'] == protocol['manifest_sha256'] and meta['steps'] == protocol['steps_per_model']
            if phase == 'shared_recovery': assert meta['recovery_sha256'] == fingerprint(root/'shared_recovery.json')
            primary = 'reactive' if name == 'reactive' else 'correct'
            g = gate(read(folder/f'train_{primary}.json'))
            recorded = next(r for r in summary if (r['phase'], r['model'], r['split'], r['mode']) == (phase, name, 'train', primary))
            assert g['overall'] == recorded['overall'] and g['passed'] == recorded['passed']
            lines.append(f"| {phase} | {name} | {g['overall'][0]}/{g['overall'][1]} | {g['passed']} |")
            for mode in ([primary] if name == 'reactive' else ['correct', 'empty', 'twin']):
                summarize(read(folder/f'validation_{mode}.json'), phase=phase, model=name, mode=mode, training_gate_passed=g['passed'])
    lines += ['', '| Data | Policy/context | Seen axes / seen posture | New axes / seen posture | Seen axes / shifted posture | New axes / shifted posture |',
              '|---|---|---:|---:|---:|---:|']
    for phase in ('original', 'shared_recovery'):
        for name, mode in [('reactive', 'reactive'), ('history', 'correct'), ('history', 'empty'), ('history', 'twin')]:
            selected = [next(r for r in rows if (r['phase'], r['model'], r['mode'], r['condition']) == (phase, name, mode, c)) for c in conditions]
            lines.append(f'| {phase} | {name}/{mode} | '+' | '.join(f"{r['success']}/96" for r in selected)+' |')
    lines += ['', '| Frozen old teacher seed | Seen/seen | New/seen | Seen/shifted | New/shifted |', '|---|---:|---:|---:|---:|']
    for seed in range(3):
        artifact = read(root/f'privileged_{seed}.json')
        summarize(artifact['records'], phase='frozen_old_teacher', model='oracle', mode='oracle', seed=seed)
        selected = [next(r for r in rows if r.get('seed') == seed and r['condition'] == c) for c in conditions]
        lines.append(f'| {seed} | '+' | '.join(f"{r['success']}/96" for r in selected)+' |')
    lines += ['', 'Privileged references were trained on the previous dataset, not on these new demonstrations.',
        'Every failed gate is retained. Validation is diagnostic when training control fails.',
        'Goal offsets and physical parameters are matched across factors; Cartesian states necessarily vary with posture/axes.',
        'One recovery round and one training seed do not establish general adaptation.']
    (root/'comparison.json').write_text(json.dumps(rows, indent=2))
    paired = []
    for phase in ('original', 'shared_recovery'):
        reactive = {(r['body_seed'], r['goal_id']): r for r in read(root/f'{phase}_reactive'/'validation_reactive.json')}
        history = {(r['body_seed'], r['goal_id']): r for r in read(root/f'{phase}_history'/'validation_correct.json')}
        assert reactive.keys() == history.keys()
        for condition in conditions:
            keys = [k for k, r in history.items() if r['condition'] == condition]
            paired.append({'phase': phase, 'condition': condition,
                'history_only': sum(history[k]['success'] and not reactive[k]['success'] for k in keys),
                'reactive_only': sum(reactive[k]['success'] and not history[k]['success'] for k in keys)})
    (root/'paired_wins.json').write_text(json.dumps(paired, indent=2))
    (root/'report.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    configs = [(phase, name, 'reactive' if name == 'reactive' else 'correct')
               for phase in ('original', 'shared_recovery') for name in ('reactive', 'history')]
    for ax, metric, title in zip(axes, ('six_success', 'seven_success'), ('Six-joint arms', 'Seven-joint arms')):
        values = np.array([[next(r[metric] for r in rows if (r['phase'], r['model'], r['mode'], r['condition']) == (phase, name, mode, c))
                           for c in conditions] for phase, name, mode in configs])
        im = ax.imshow(values/48*100, vmin=0, vmax=100, cmap='Blues', aspect='auto')
        for i in range(4):
            for j in range(4): ax.text(j, i, f'{values[i,j]}/48', ha='center', va='center', color='white' if values[i,j] > 25 else 'black')
        ax.set_xticks(range(4), ['Seen axes\nSeen posture', 'New axes\nSeen posture', 'Seen axes\nShifted posture', 'New axes\nShifted posture'], fontsize=8)
        ax.set_yticks(range(4), ['Original reactive', 'Original history', 'Recovery reactive', 'Recovery history'])
        ax.set_title(title)
    fig.suptitle('E26: factorial validation — single-seed, directly supervised policies')
    fig.text(.5, .015, 'Fixed shared recovery union. Training-gate failures retained; see report.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .94]); fig.savefig(root/'comparison.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--output', required=True)
    report(**vars(p.parse_args()))
