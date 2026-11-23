"""Verify and summarize the paired pilot, including a failed teacher gate."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from .benchmark import fingerprint
from .history_distill import encoder_key


def report(output):
    root = Path(output)
    protocol = json.loads((root/'protocol.json').read_text())
    status = json.loads((root/'status.json').read_text())
    teacher_meta = json.loads((root/'teacher'/'training.json').read_text())
    assert fingerprint(root/'teacher'/'policy.pt') == teacher_meta['policy_sha256']
    assert protocol['manifest_sha256'] == teacher_meta['manifest_sha256']
    gate = json.loads((root/'teacher'/'gate.json').read_text())
    lines = ['# E23 paired-arm pilot', '', f"Status: {status['status']}", '',
             f"Privileged training control: {gate['overall'][0]}/{gate['overall'][1]}; gate passed: {gate['passed']}.", '',
             'Single seed; local reaching goals; no final-test policy evaluation.', '']
    if not gate['passed']:
        assert status['status'] == 'teacher_gate_failed'
        lines += ['Student fitting was not run. Resolve teacher control before testing history transfer.']
    else:
        teacher = torch.load(root/'teacher'/'policy.pt', map_location='cpu', weights_only=True)
        summary = json.loads((root/'summary.json').read_text())
        assert len(summary) == 12
        lines += ['| Model | Split | Context | Success | Training-style gate | Mean error at step 10 (n) |',
                  '|---|---|---|---:|---|---:|']
        for name in ('history', 'pose'):
            meta = json.loads((root/name/'training.json').read_text())
            assert meta['policy_sha256'] == fingerprint(root/name/'policy.pt')
            state = torch.load(root/name/'policy.pt', map_location='cpu', weights_only=True)
            assert all(torch.equal(v, teacher[k]) for k, v in state.items() if not encoder_key(k))
            for split in ('train', 'validation'):
                by_mode = {}
                for mode in ('correct', 'empty', 'twin'):
                    records = json.loads((root/name/f'{split}_{mode}.json').read_text())
                    assert len(records) == (256 if split == 'train' else 96)
                    by_mode[mode] = {(r['body_seed'], r['goal_id']): r for r in records}
                    row = next(r for r in summary if (r['model'], r['split'], r['mode']) == (name, split, mode))
                    errors = [r['error_at_10'] for r in records if r['error_at_10'] is not None]
                    metric = f'{np.mean(errors):.5f} ({len(errors)})' if errors else 'N/A (0)'
                    lines.append(f"| {name} | {split} | {mode} | {row['overall'][0]}/{row['overall'][1]} | {row['passed']} | {metric} |")
                assert by_mode['correct'].keys() == by_mode['twin'].keys()
                if name == 'pose':
                    for key, left in by_mode['correct'].items():
                        right = by_mode['twin'][key]
                        assert all(left[k] == right[k] for k in ('success', 'collision', 'error', 'steps', 'error_at_10'))
        lines += ['', 'Frozen controllers and hashes verified. Pose-only correct/twin outcomes match exactly.',
                  'Step-10 means exclude already-ended episodes; differing sample counts can confound comparisons.',
                  'Validation gates above are descriptive, not training gates or confirmation of adaptation.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    report(**vars(parser.parse_args()))
