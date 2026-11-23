"""E23: paired goals and calibration with identical initial observations."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import numpy as np
from .ambiguous_fixture import PAIRS, pair
from .benchmark import fingerprint
from .mixed_sim import AXES, MixedReach, token
from .mixed_data import expert_rollout, load


def calibration(arm, seed):
    env = MixedReach(arm, synchronize=True)
    before = env.reset(0, [0, 0, .9], spread=0)
    rng = np.random.default_rng(seed)
    history, commands = [], []
    for step in range(32):
        if step % 4 == 0:
            command = rng.uniform(-.018, .018, arm.dof)
        after, _, info = env.step(command)
        if info['collision']:
            raise RuntimeError('Calibration contact; retain failed version')
        history.append(token(before, info['executed_action'], after))
        commands.append(command.copy())
        before = after
    return np.asarray(history), np.asarray(commands)


def register(directory):
    manifest = load(directory)
    AXES.update(manifest['axis_catalog'])
    return manifest


def build(output):
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 'ease_mixed_v1', 'benchmark': 'ease_paired_v1', 'experiment': 'E23',
        'synchronized_observations': True, 'arms': [], 'pairs': [], 'episodes': [],
        'calibrations': [], 'goals': {'train': [], 'validation': [], 'test': []},
        'axis_catalog': {}, 'feasibility_audit': [], 'test_evaluated': False,
        'protocol': {'train_goals_per_pair': 32, 'validation_goals_per_pair': 12,
            'reset_spread': 0, 'calibration_steps': 32, 'training_calibrations_per_body': 16,
            'history_sampling': 'Calibration prefixes only; no moving task-window poses',
            'goal_distribution': 'Pair-relative offsets; radius 0.06..0.12m, negative vertical offset',
            'test': 'Geometry reserved; no goal screening or rollout in this version',
            'selection': 'Keep candidate only if both twins succeed; retain all candidate outcomes'}}
    (root/'protocol.json').write_text(json.dumps(manifest['protocol'], indent=2))
    source = root/'source'; source.mkdir()
    for p in Path(__file__).parent.glob('*.py'): shutil.copyfile(p, source/p.name)
    gid = 0
    for split_index, (split, configurations) in enumerate(PAIRS.items()):
        for config_index, configuration in enumerate(configurations):
            for instance in range(2):
                pid = split_index*4 + config_index*2 + instance
                arms = [replace(a, seed=82000 + 2*pid + side) for side, a in enumerate(pair(split, configuration, instance))]
                manifest['arms'].extend(a.manifest() for a in arms)
                manifest['axis_catalog'].update({a.family: AXES[a.family] for a in arms})
                manifest['pairs'].append({'id': pid, 'split': split, 'bodies': [a.seed for a in arms]})
                for a in arms: (root/f'body_{a.seed}.xml').write_text(MixedReach(a).xml)
                if split == 'test': continue
                envs = [MixedReach(a, synchronize=True) for a in arms]
                initial = [e.reset(0, [0, 0, .9], spread=0) for e in envs]
                assert all(np.array_equal(initial[0][k], initial[1][k]) for k in initial[0])
                origin = envs[0].data.site_xpos[0].copy()
                rng = np.random.default_rng(91000 + pid)
                needed = 32 if split == 'train' else 12
                accepted = 0
                for candidate in range(needed*100):
                    delta = rng.uniform([-.1, -.1, -.1], [.1, .1, -.02])
                    if not .06 <= np.linalg.norm(delta) <= .12: continue
                    goal = origin + delta
                    trajectories = [expert_rollout(e, 0, goal, spread=0) for e in envs]
                    audit = {'pair': pid, 'candidate': candidate, 'xyz': goal.tolist(),
                             'outcomes': [r for _, r in trajectories]}
                    manifest['feasibility_audit'].append(audit)
                    with (root/'feasibility.jsonl').open('a') as f: f.write(json.dumps(audit)+'\n')
                    if not all(r['success'] and not r['collision'] for _, r in trajectories): continue
                    for key in ('joints', 'task'):
                        assert np.array_equal(trajectories[0][0][key][0], trajectories[1][0][key][0])
                    manifest['goals'][split].append({'id': gid, 'pair': pid, 'xyz': goal.tolist(), 'seed': 0})
                    if split == 'train':
                        for a, (trajectory, result) in zip(arms, trajectories):
                            filename = f'episode_{a.seed}_{gid}.npz'
                            np.savez_compressed(root/filename, **trajectory)
                            manifest['episodes'].append({'file': filename, 'body_seed': a.seed,
                                'family': a.family, 'split': split, 'dof': a.dof, 'goal_id': gid,
                                'variant': 'paired_zero_spread', **result})
                    gid += 1; accepted += 1
                    if accepted == needed: break
                assert accepted == needed, 'Insufficient paired feasible goals'
                for index in range(16 if split == 'train' else 1):
                    results = [calibration(a, 92000 + pid*100 + index) for a in arms]
                    assert np.array_equal(results[0][1], results[1][1])
                    assert np.array_equal(results[0][0][0, :, [0, 5, 6, 7]], results[1][0][0, :, [0, 5, 6, 7]])
                    for a, (history, commands) in zip(arms, results):
                        filename = f'calibration_{a.seed}_{index}.npz'
                        np.savez_compressed(root/filename, history=history, commands=commands)
                        if split == 'train':
                            hname = filename.replace('.npz', '.npy'); np.save(root/hname, history)
                            manifest['calibrations'].append({'file': hname, 'body_seed': a.seed, 'split': split})
                print('pair', pid, split, 'goals', accepted, flush=True)
    manifest['asset_hashes'] = {p.name: fingerprint(p) for p in root.iterdir() if p.is_file()}
    manifest['source_hashes'] = {p.name: fingerprint(p) for p in source.glob('*.py')}
    (root/'manifest.json').write_text(json.dumps(manifest, indent=2))
    (root/'manifest.sha256').write_text(fingerprint(root/'manifest.json'))
    print('completed', len(manifest['episodes']), 'training episodes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    build(**vars(parser.parse_args()))
