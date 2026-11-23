"""Paired E21 comparisons, retaining every prespecified seed and condition."""
import argparse
import json
from pathlib import Path
import torch
from .benchmark import fingerprint
from .history_distill import encoder_key


def report(output):
    root = Path(output)
    entries = json.loads((root/'index.json').read_text())
    assert [e['seed'] for e in entries] == [0, 1, 2]
    rows = []
    for entry in entries:
        teacher = torch.load(entry['teacher'], map_location='cpu', weights_only=True)
        pose_path = Path(entry['pose'])/'policy.pt'
        assert fingerprint(pose_path) == entry['pose_sha256']
        assert fingerprint(entry['teacher']) == entry['teacher_sha256']
        pose = torch.load(pose_path, map_location='cpu', weights_only=True)
        assert all(torch.equal(value, teacher[key]) for key, value in pose.items() if not encoder_key(key))
        artifacts = [json.loads(Path(entry[key]).read_text()) for key in ('full_validation', 'validation')]
        for artifact, checkpoint in zip(artifacts, [Path(entry['full_history'])/'policy.pt', pose_path]):
            assert artifact['checkpoint_sha256'] == fingerprint(checkpoint)
            assert not artifact['test_evaluated']
            assert len(artifact['records']) == 784
        def key(row):
            return row['body_seed'], row['goal_id'], row['task_seed'], row['mode']
        maps = [{key(r): r for r in a['records']} for a in artifacts]
        assert maps[0].keys() == maps[1].keys()
        for split in ('validation', 'validation_topology'):
            for mode in sorted({r['mode'] for r in artifacts[0]['records']}):
                cases = [k for k, r in maps[0].items() if r['split'] == split and r['mode'] == mode]
                rows.append({'seed': entry['seed'], 'split': split, 'mode': mode, 'trials': len(cases),
                    'full_success': sum(maps[0][k]['success'] for k in cases),
                    'pose_success': sum(maps[1][k]['success'] for k in cases),
                    'full_only': sum(maps[0][k]['success'] and not maps[1][k]['success'] for k in cases),
                    'pose_only': sum(maps[1][k]['success'] and not maps[0][k]['success'] for k in cases)})
    (root/'comparison.json').write_text(json.dumps(rows, indent=2))
    lines = ['# E21: Matched trained pose-only baseline', '',
             'All seeds retained. Seed 1 inherits a teacher that failed the training gate. No final-test policy evaluation.', '',
             '| Seed | Split | Context | Full history | Pose only | Full-only / pose-only wins |',
             '|---|---|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['seed']} | {r['split']} | {r['mode']} | {r['full_success']}/{r['trials']} | {r['pose_success']}/{r['trials']} | {r['full_only']} / {r['pose_only']} |")
    lines += ['', 'Training gates:', '']
    for e in entries:
        lines.append(f"- Seed {e['seed']}: full {e['full_training_gate']}; pose {e['pose_training_gate']}.")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([r for r in rows if r['mode'] == 'correct32'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    report(**vars(parser.parse_args()))
