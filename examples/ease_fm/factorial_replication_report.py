"""Audited E27 outcomes and explicit deployment-interaction accounting."""
import argparse
import json
from pathlib import Path
import numpy as np
from .benchmark import fingerprint
from .paired_experiment import gate
from .paired_data import register

CONDITIONS = ['seen_axes_seen_posture', 'new_axes_seen_posture',
              'seen_axes_shifted_posture', 'new_axes_shifted_posture']
BUDGETS = [25, 50, 100, 150, 182]


def efficiency(records, mode):
    """Failures retain their full horizon; reuse is aggregate cost, not latency."""
    assert records
    assert all(1 <= r['steps'] <= 150 for r in records)
    assert all(r['success'] or r['steps'] == 150 for r in records)
    cost = 32 if mode in ('correct', 'twin') else 0
    groups = {(r['body_seed'], r['condition']) for r in records}
    steps = sum(r['steps'] for r in records)
    return dict(task_steps=steps, mean_task_steps=steps/len(records),
        single_use_steps=steps+cost*len(records),
        mean_single_use_steps=steps/len(records)+cost,
        reused_steps=steps+cost*len(groups),
        mean_reused_steps=(steps+cost*len(groups))/len(records),
        calibration_groups=len(groups),
        success_within_budget={str(b): sum(bool(r['success']) and r['steps']+cost <= b for r in records) for b in BUDGETS})


def read(path): return json.loads(Path(path).read_text())


def report(data, output):
    root = Path(output); manifest = register(data)
    protocol = read(root/'protocol.json'); entries = read(root/'index.json')
    assert read(root/'status.json')['status'] == 'complete'
    assert len(entries) == 12
    assert {(e['seed'], e['phase'], e['model']) for e in entries} == {
        (s, p, m) for s in (0, 1, 2) for p in ('original', 'shared_recovery') for m in ('reactive', 'history')}
    assert protocol['manifest_sha256'] == fingerprint(Path(data)/'manifest.json')
    pilot = Path(protocol['pilot']); shared = read(pilot/'shared_recovery.json')
    assert fingerprint(pilot/'shared_recovery.json') == protocol['recovery_sha256']
    assert all(fingerprint(Path(p.replace('\\', '/'))) == h for p, h in shared['files'].items())
    assert all(fingerprint(Path(p.replace('\\', '/'))) == h for p, h in shared['source_checkpoints'].items())
    dofs = {a['seed']: len(a['lengths']) for a in manifest['arms']}
    pairs = {p['id']: p['bodies'] for p in manifest['pairs']}
    expected = {(body, g['id'], g['condition']) for g in manifest['goals']['validation'] for body in pairs[g['pair']]}
    # Reuse is justified only for exactly the same robot/reset and probe seed.
    for condition in CONDITIONS:
        for body in {b for b, _, c in expected if c == condition}:
            goals = [g for g in manifest['goals']['validation'] if g['condition'] == condition and body in pairs[g['pair']]]
            assert len(goals) == 12
            assert len({(g['pair'], tuple(g['home'])) for g in goals}) == 1
    rows = []; training = []; reference_keys = None; raw = {}; evidence = {}
    for entry in entries:
        folder = Path(entry['folder']); meta = read(folder/'training.json')
        assert fingerprint(folder/'policy.pt') == entry['policy_sha256'] == meta['policy_sha256']
        assert meta['steps'] == 15000 and meta['seed'] == entry['seed']
        assert meta['parameters'] == 244289 and meta['kind'] == 'direct'
        assert meta['transitions'] == (6635 if entry['phase'] == 'shared_recovery' else 3424)
        assert meta['manifest_sha256'] == protocol['manifest_sha256']
        assert meta['recovery_sha256'] == (protocol['recovery_sha256'] if entry['phase'] == 'shared_recovery' else None)
        identity = {k: entry[k] for k in ('seed', 'phase', 'model')}
        for mode in (['reactive'] if entry['model'] == 'reactive' else ['correct', 'empty', 'twin']):
            train = read(folder/f'train_{mode}.json')
            for split in ('train', 'validation'):
                path = folder/f'{split}_{mode}.json'
                evidence[str(path)] = fingerprint(path)
            assert len(train) == 256 and len({(r['body_seed'], r['goal_id']) for r in train}) == 256
            training.append(dict(**identity, mode=mode, **gate(train)))
            records = read(folder/f'validation_{mode}.json')
            keys = {(r['body_seed'], r['goal_id'], r['condition']) for r in records}
            assert len(records) == len(keys) == 384
            assert keys == expected
            if reference_keys is None: reference_keys = keys
            assert keys == reference_keys
            raw[(entry['seed'], entry['phase'], mode)] = records
            for condition in CONDITIONS:
                selected = [r for r in records if r['condition'] == condition]
                assert len(selected) == 96
                assert len({r['body_seed'] for r in selected}) == 8
                rows.append(dict(**identity, mode=mode, condition=condition,
                    successes=sum(r['success'] for r in selected), trials=len(selected),
                    six_successes=sum(r['success'] for r in selected if dofs[r['body_seed']] == 6),
                    seven_successes=sum(r['success'] for r in selected if dofs[r['body_seed']] == 7),
                    collisions=sum(r['collision'] for r in selected), **efficiency(selected, mode)))
    paired = []
    for seed in (0, 1, 2):
        for phase in ('original', 'shared_recovery'):
            correct = {(r['body_seed'], r['goal_id']): r for r in raw[seed, phase, 'correct']}
            for mode in ('reactive', 'empty'):
                for condition in CONDITIONS:
                    pairs = [(correct[r['body_seed'], r['goal_id']], r) for r in raw[seed, phase, mode] if r['condition'] == condition]
                    both = [(a, b) for a, b in pairs if a['success'] and b['success']]
                    paired.append(dict(seed=seed, phase=phase, comparator=mode, condition=condition,
                        history_only=sum(a['success'] and not b['success'] for a, b in pairs),
                        comparator_only=sum(b['success'] and not a['success'] for a, b in pairs),
                        both_success=len(both),
                        both_success_mean_task_step_difference=float(np.mean([a['steps']-b['steps'] for a, b in both])) if both else None))
    (root/'comparison.json').write_text(json.dumps(dict(validation=rows, training=training, paired=paired), indent=2))
    (root/'audit.json').write_text(json.dumps(dict(policy_checkpoints=12,
        validation_records=sum(len(v) for v in raw.values()),
        training_records=len(training)*256, test_evaluated=False,
        evaluation_sha256=evidence, report_source_sha256=fingerprint(Path(__file__))), indent=2))
    lines = ['# E27: fixed-data replication and interaction cost', '',
        'Three training seeds; seed 0 is the E26 pilot. Eight new 15,000-update policies use the unchanged training data and fixed 3,211-chunk recovery union. No final-test evaluation or validation tuning.', '',
        'Each cell below lists successes for seeds 0 / 1 / 2, each out of 96 matched cases. Training gates are retained even when failed.', '',
        '| Data | Context | Seen axes, seen posture | New axes, seen posture | Seen axes, shifted posture | New axes, shifted posture |',
        '|---|---|---|---|---|---|']
    for phase in ('original', 'shared_recovery'):
        for mode in ('reactive', 'correct', 'empty', 'twin'):
            cells = [' / '.join(str(next(r['successes'] for r in rows if r['seed'] == seed and r['phase'] == phase and r['mode'] == mode and r['condition'] == c)) for seed in (0, 1, 2)) for c in CONDITIONS]
            lines.append('| '+phase+' | '+mode+' | '+' | '.join(cells)+' |')
    lines += ['', '## Overall validation and cost', '',
        'Success columns are out of 384 per seed. Costs average all cases and seeds, including full-horizon failures.', '',
        '| Data | Context | Seed 0 | Seed 1 | Seed 2 | Mean task steps | Mean single-use steps | Mean reused steps |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for phase in ('original', 'shared_recovery'):
        for mode in ('reactive', 'correct', 'empty'):
            selected = [r for r in rows if r['phase'] == phase and r['mode'] == mode]
            totals = [sum(r['successes'] for r in selected if r['seed'] == seed) for seed in (0, 1, 2)]
            means = [np.mean([r[k] for r in selected]) for k in ('mean_task_steps', 'mean_single_use_steps', 'mean_reused_steps')]
            lines.append('| '+phase+' | '+mode+' | '+' | '.join(str(v) for v in totals)+' | '+' | '.join(f'{v:.1f}' for v in means)+' |')
    lines += ['', '## Recovery-trained transfer by joint count', '',
        'Each entry lists seed 0 / 1 / 2 successes out of 48.', '',
        '| Context | Joint count | New axes, seen posture | New axes, shifted posture |', '|---|---:|---|---|']
    for mode in ('reactive', 'correct', 'empty'):
        for dof, key in [(6, 'six_successes'), (7, 'seven_successes')]:
            cells = [' / '.join(str(next(r[key] for r in rows if r['seed'] == seed and r['phase'] == 'shared_recovery' and r['mode'] == mode and r['condition'] == c)) for seed in (0, 1, 2)) for c in (CONDITIONS[1], CONDITIONS[3])]
            lines.append(f'| {mode} | {dof} | '+ ' | '.join(cells)+' |')
    lines += ['', '## Training control', '', '| Data | Context | Seed | Success / 256 | Gate |', '|---|---|---:|---:|---|']
    for row in training:
        lines.append(f"| {row['phase']} | {row['mode']} | {row['seed']} | {row['overall'][0]} | {row['passed']} |")
    lines += ['', '## Interaction accounting', '',
        'Costs are control steps (25 Hz), excluding resets and computation. All unsuccessful episodes consume the full 150-step horizon. Correct/twin history requires a 32-step probe. Reactive/empty needs no probe: the evaluator may generate an ignored probe for empty, so zero cost describes the minimal deployment input, not literal evaluator runtime.', '',
        'Single-use cost adds 32 steps per task. Reuse cost adds one probe for each body/posture group of 12 goals (8 probes per 96-case cell); reuse is aggregate suite accounting, not per-task latency. Frozen context is goal-independent and identical within each group. Report success jointly with cost; lower capped cost alone does not establish better control.', '',
        'The table averages over three seeds; success is mean successes / 96. Budget success includes calibration and counts only successful tasks finished by that budget. Full seed-level costs, six/seven-joint results, collisions, and paired wins are in comparison.json.', '',
        '| Data | Context | Condition | Success / 96 | Mean task steps | Single use | Reused | Success by 50 | Success by 100 | Success by 150 |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for phase in ('original', 'shared_recovery'):
        for mode in ('reactive', 'correct', 'empty'):
            for condition in CONDITIONS:
                selected = [r for r in rows if r['phase'] == phase and r['mode'] == mode and r['condition'] == condition]
                means = [np.mean([r[k] for r in selected]) for k in ('successes', 'mean_task_steps', 'mean_single_use_steps', 'mean_reused_steps')]
                budgets = [np.mean([r['success_within_budget'][str(b)] for r in selected]) for b in (50, 100, 150)]
                lines.append('| '+phase+' | '+mode+' | '+condition+' | '+' | '.join(f'{v:.1f}' for v in means+budgets)+' |')
    lines += ['', 'Three training seeds do not represent three independent robot-family samples. Seed 0 was already observed when replication was proposed. The fixed recovery union tests training randomness, not data-collection randomness. Six-joint and shifted-posture failures remain relevant; no equivalence or broad embodiment-adaptation claim follows from close aggregate success.', '']
    (root/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, condition, title in zip(axes, (CONDITIONS[1], CONDITIONS[3]), ('New axes / seen posture', 'New axes / shifted posture')):
        for mode, label in [('reactive', 'Reactive'), ('correct', 'Correct history (+32 steps)'), ('empty', 'Empty history')]:
            selected = [r for r in rows if r['phase'] == 'shared_recovery' and r['mode'] == mode and r['condition'] == condition]
            ys = np.array([[r['success_within_budget'][str(b)]/96*100 for b in BUDGETS] for r in selected])
            line, = ax.plot(BUDGETS, ys.mean(0), marker='o', label=label)
            ax.fill_between(BUDGETS, ys.min(0), ys.max(0), color=line.get_color(), alpha=.12)
        ax.set(title=title, xlabel='Total interaction budget (control steps)', ylim=(0, 100))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Successful tasks (%)'); axes[1].legend(fontsize=8)
    fig.suptitle('E27 shared recovery: single-use calibration included; bands = seed range')
    fig.tight_layout(); fig.savefig(root/'interaction_efficiency.png', dpi=160); plt.close(fig)
    print('\n'.join(lines[:17]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True); parser.add_argument('--output', required=True)
    report(**vars(parser.parse_args()))
