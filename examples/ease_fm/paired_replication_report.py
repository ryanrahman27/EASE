"""E24 audit: report all seeds, probe conditions, and failed gates."""
import argparse
import json
from pathlib import Path
import torch
from .benchmark import fingerprint
from .history_distill import encoder_key
from .paired_experiment import gate


def read(path):
    return json.loads(Path(path).read_text())


def report(output):
    root = Path(output); protocol = read(root/'protocol.json'); entries = read(root/'index.json')
    assert [e['seed'] for e in entries] == protocol['seeds']
    rows = []; comparisons = []
    lines = ['# E24: paired-arm replication and reactive baseline', '',
        'Fixed dataset. Seed 0 includes the earlier pilot; seeds 1/2 are new. All failures retained.', '',
        '| Seed | Teacher training | History training | Pose training | Reactive training |',
        '|---|---:|---:|---:|---:|']
    def checkpoint(folder):
        meta = read(folder/'training.json')
        assert fingerprint(folder/'policy.pt') == meta['policy_sha256']
        if 'manifest_sha256' in meta: assert meta['manifest_sha256'] == protocol['manifest_sha256']
        return torch.load(folder/'policy.pt', map_location='cpu', weights_only=True)
    def add(seed, name, probe, mode, records, training_gate):
        assert len(records) == 96
        keys = {(r['body_seed'], r['goal_id']) for r in records}
        assert len(keys) == 96 and all(r['split'] == 'validation' for r in records)
        rows.append({'seed': seed, 'model': name, 'probe_seed': probe, 'mode': mode,
            'success': sum(r['success'] for r in records), 'trials': len(records),
            'six_success': sum(r['success'] for r in records if r['body_seed'] < 82012),
            'seven_success': sum(r['success'] for r in records if r['body_seed'] >= 82012),
            'collisions': sum(r['collision'] for r in records), 'training_gate_passed': training_gate})
        return {(r['body_seed'], r['goal_id']): r for r in records}
    for entry in entries:
        seed = entry['seed']; run = Path(entry['run']); reactive = Path(entry['reactive'])
        teacher = checkpoint(run/'teacher'); checkpoint(reactive)
        assert json.loads(json.dumps(gate(read(run/'teacher'/'train.json')))) == entry['teacher_gate']
        rg = gate(read(reactive/'train.json'))
        assert rg['overall'] == entry['reactive_gate']['overall'] and rg['passed'] == entry['reactive_gate']['passed']
        reactive_cases = add(seed, 'reactive', None, 'reactive', read(reactive/'validation.json'), rg['passed'])
        training = {}
        if entry['teacher_gate']['passed']:
            for name in ('history', 'pose'):
                state = checkpoint(run/name)
                assert all(torch.equal(v, teacher[k]) for k, v in state.items() if not encoder_key(k))
                training[name] = gate(read(run/name/'train_correct.json'))
                for probe in protocol['probe_seeds']:
                    modes = {}
                    for mode in ('correct', 'empty', 'twin'):
                        path = run/name/f'validation_{mode}.json' if probe == 930000 else root/next(
                            e['file'] for e in entry['extra_probes'] if (e['model'], e['probe'], e['mode']) == (name, probe, mode))
                        modes[mode] = add(seed, name, probe, mode, read(path), training[name]['passed'])
                    assert modes['correct'].keys() == modes['twin'].keys() == reactive_cases.keys()
                    if name == 'pose':
                        assert all(all(r[k] == modes['twin'][key][k] for k in ('success', 'error', 'collision', 'steps')) for key, r in modes['correct'].items())
                    else:
                        for comparator, cases in [('twin', modes['twin']), ('reactive', reactive_cases)]:
                            comparisons.append({'seed': seed, 'probe_seed': probe, 'comparator': comparator,
                                'history_only': sum(r['success'] and not cases[k]['success'] for k, r in modes['correct'].items()),
                                'comparator_only': sum(cases[k]['success'] and not r['success'] for k, r in modes['correct'].items())})
        def cell(g):
            return f"{g['overall'][0]}/{g['overall'][1]} ({'pass' if g['passed'] else 'FAIL'})" if g else 'not trained'
        lines.append(f"| {seed} | {cell(entry['teacher_gate'])} | {cell(training.get('history'))} | {cell(training.get('pose'))} | {cell(rg)} |")
    lines += ['', '| Seed | Probe seed | History | Empty | Twin | Pose | Reactive |', '|---|---|---:|---:|---:|---:|---:|']
    for entry in entries:
        seed = entry['seed']
        def count(name, probe, mode):
            matches = [r for r in rows if (r['seed'], r['model'], r['probe_seed'], r['mode']) == (seed, name, probe, mode)]
            return f"{matches[0]['success']}/96" if matches else 'not trained'
        for probe in protocol['probe_seeds']:
            values = [count('history', probe, mode) for mode in ('correct', 'empty', 'twin')]
            values += [count('pose', probe, 'correct'), count('reactive', None, 'reactive')]
            lines.append(f'| {seed} | {probe} | '+ ' | '.join(values)+' |')
    lines += ['', 'Reactive values repeat across probe rows because reactive inference uses no probe.',
        'Repeated goals/bodies/probes are dependent observations, not independent embodiment samples.',
        'Update counts are matched, not FLOPs or effective parameter counts. Failed training gates limit transfer interpretation.',
        'No final-test evaluation. Checkpoint hashes, frozen student controllers, and pose correct/twin invariance verified.']
    (root/'comparison.json').write_text(json.dumps({'conditions': rows, 'paired_wins': comparisons}, indent=2))
    (root/'report.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, title in zip(axes, ['six_success', 'seven_success'], ['Held-out six-joint arms', 'Held-out seven-joint arms']):
        for name, mode, color, label in [('history', 'correct', '#167d9a', 'Correct history'),
                ('history', 'twin', '#bb5555', 'Twin history'), ('pose', 'correct', '#bb8b16', 'Pose only'),
                ('reactive', 'reactive', '#555555', 'Reactive')]:
            shift = {'Correct history': -.24, 'Twin history': -.08, 'Pose only': .08, 'Reactive': .24}[label]
            for seed in protocol['seeds']:
                selected = [r[key]/48*100 for r in rows if r['seed'] == seed and r['model'] == name and r['mode'] == mode]
                if not selected: continue
                center = sum(selected)/len(selected)
                ax.errorbar(seed+shift, center, yerr=[[center-min(selected)], [max(selected)-center]],
                            fmt='o', color=color, capsize=3, label=label if seed == 0 else None)
        ax.set_xticks([0, 1, 2]); ax.set_xlabel('Training seed'); ax.set_ylabel('Success (%)')
        ax.set_ylim(-3, 103); ax.set_title(title); ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle('E24: means and ranges across three calibration seeds (not confidence intervals)')
    fig.text(.5, .015, 'Reactive uses no calibration. Failed training gates retained; see report.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .05, 1, .94]); fig.savefig(root/'comparison.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    report(**vars(parser.parse_args()))
