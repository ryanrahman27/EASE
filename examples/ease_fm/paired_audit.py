"""Independent dataset replay and twin information-boundary checks."""
import argparse
import json
from pathlib import Path
import numpy as np
from .benchmark import fingerprint
from .paired_data import register
from .mixed_sim import Arm, MixedReach


def audit(data, output):
    root = Path(data); manifest = register(root)
    arms = {a['seed']: Arm(**a) for a in manifest['arms']}
    assert len(arms) == len(manifest['arms']) == 24
    assert not manifest['goals']['test']
    assert all(e['split'] == 'train' for e in manifest['episodes'])
    assert all(e['split'] == 'train' for e in manifest['calibrations'])
    axes = {split: {tuple(manifest['axis_catalog'][a.family]) for a in arms.values() if a.split == split}
            for split in ('train', 'validation', 'test')}
    assert not (axes['train'] & axes['validation'] or axes['train'] & axes['test'] or axes['validation'] & axes['test'])
    goals = {g['id']: g for g in manifest['goals']['train']}
    replay = 0
    for entry in manifest['episodes']:
        env = MixedReach(arms[entry['body_seed']], synchronize=True)
        obs = env.reset(0, goals[entry['goal_id']]['xyz'], spread=0)
        with np.load(root/entry['file']) as episode:
            for i, action in enumerate(episode['action']):
                np.testing.assert_allclose(obs['joints'], episode['joints'][i], atol=2e-6)
                np.testing.assert_allclose(obs['task'], episode['task'][i], atol=2e-6)
                obs, done, info = env.step(action)
            assert done and info['success'] and not info['collision']
        replay += 1
    checked = 0
    for pair in manifest['pairs']:
        if pair['split'] == 'test': continue
        a, b = pair['bodies']
        for i in range(16 if pair['split'] == 'train' else 1):
            with np.load(root/f'calibration_{a}_{i}.npz') as left, np.load(root/f'calibration_{b}_{i}.npz') as right:
                assert np.array_equal(left['commands'], right['commands'])
                assert np.array_equal(left['history'][0, :, [0, 5, 6, 7]], right['history'][0, :, [0, 5, 6, 7]])
                assert not np.array_equal(left['history'], right['history'])
            checked += 1
    for name, digest in manifest['source_hashes'].items(): assert fingerprint(root/'source'/name) == digest
    result = {'expert_replay_successes': replay, 'training_episodes': len(manifest['episodes']),
        'paired_calibrations_checked': checked, 'unique_body_ids': len(arms),
        'assets_and_sources_verified': True, 'axis_splits_disjoint': True, 'test_rollouts': 0}
    Path(output).write_text(json.dumps(result, indent=2)); print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True); parser.add_argument('--output', required=True)
    audit(**vars(parser.parse_args()))
