"""Independent replay and factorial-design invariants for E26."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from .paired_data import register
from .factorial_data import check_pair, home
from .mixed_sim import Arm, MixedReach
from .benchmark import fingerprint


def audit(data, output):
    root = Path(data); m = register(root)
    arms = {a['seed']: Arm(**a) for a in m['arms']}
    pairs = {p['id']: p for p in m['pairs']}
    assert len(arms) == len(m['arms']) == 24 and not m['goals']['test']
    for pid in range(4):
        left = [arms[b] for b in pairs[pid]['bodies']]
        right = [arms[b] for b in pairs[pid+4]['bodies']]
        for a, b in zip(left, right):
            for key in ('lengths', 'masses', 'strength', 'damping', 'home'): assert getattr(a, key) == getattr(b, key)
        for aa in (left, right):
            check_pair(aa); check_pair([replace(a, home=home(a.dof, True)) for a in aa])
    train_goals = {g['id']: g for g in m['goals']['train']}
    for episode in m['episodes']:
        assert episode['split'] == 'train'
        arm = arms[episode['body_seed']]; env = MixedReach(arm, synchronize=True)
        obs = env.reset(0, train_goals[episode['goal_id']]['xyz'], spread=0)
        with np.load(root/episode['file']) as values:
            for i, action in enumerate(values['action']):
                np.testing.assert_allclose(obs['joints'], values['joints'][i], atol=2e-6)
                np.testing.assert_allclose(obs['task'], values['task'][i], atol=2e-6)
                obs, done, info = env.step(action)
            assert done and info['success'] and not info['collision']
    groups = {}
    for g in m['goals']['validation']:
        aa = [replace(arms[b], home=tuple(g['home'])) for b in pairs[g['pair']]['bodies']]
        origin = check_pair(aa)
        groups.setdefault((g['physical_group'], g['offset_id']), []).append((g['condition'], np.asarray(g['xyz'])-origin))
        observations = [MixedReach(a, synchronize=True).reset(0, g['xyz'], spread=0) for a in aa]
        assert all(np.array_equal(observations[0][k], observations[1][k]) for k in observations[0])
    for values in groups.values():
        assert len(values) == len({c for c, _ in values}) == 4
        for _, delta in values: np.testing.assert_allclose(delta, values[0][1], atol=1e-12)
    for name, digest in m['source_hashes'].items(): assert fingerprint(root/'source'/name) == digest
    result = {'expert_replays': len(m['episodes']), 'validation_cases': len(m['goals']['validation'])*2,
        'matched_offset_groups': len(groups), 'physical_parameters_matched': True, 'pair_observations_equal': True,
        'jacobians_rank_three': True, 'seen_home_vectors_identical_across_axes': True,
        'test_rollouts': 0, 'hashes_verified': True}
    Path(output).write_text(json.dumps(result, indent=2)); print(result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--output', required=True)
    audit(**vars(p.parse_args()))
