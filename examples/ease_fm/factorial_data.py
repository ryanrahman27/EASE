"""E26: crossed axis/posture validation and pair-shared reset configurations."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np
from .ambiguous_fixture import PAIRS, pair
from .paired_data import calibration
from .mixed_sim import AXES, MixedReach
from .mixed_data import expert_rollout
from .benchmark import fingerprint


def home(dof, shifted=False):
    q = np.zeros(dof); q[1] = .75 if shifted else .45; q[4] = -1.05 if shifted else -.65
    return tuple(q)


def check_pair(arms):
    envs = [MixedReach(a, synchronize=True) for a in arms]
    observations = [e.reset(0, [0, 0, .9], spread=0) for e in envs]
    assert all(np.array_equal(observations[0][k], observations[1][k]) for k in observations[0])
    for env in envs:
        jac = np.zeros((3, env.arm.dof)); mujoco.mj_jacSite(env.model, env.data, jac, None, 0)
        assert np.linalg.matrix_rank(jac, tol=1e-7) == 3
    return envs[0].data.site_xpos[0].copy()


def build(output):
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    m = {'schema': 'ease_mixed_v1', 'benchmark': 'ease_paired_factorial_v2', 'experiment': 'E26',
        'synchronized_observations': True, 'arms': [], 'pairs': [], 'axis_catalog': {},
        'episodes': [], 'calibrations': [], 'goals': {'train': [], 'validation': [], 'test': []},
        'protocol': {'training_goals_per_pair': 32, 'validation_offsets_per_physical_group': 12,
            'seen_home': 'q1=.45, q4=-.65 (zero-based); all other joints zero',
            'shifted_home': 'q1=.75, q4=-1.05; all other joints zero',
            'factorial': 'Seen/new axes crossed with seen/shifted home, matched lengths/masses/actuators and goal offsets',
            'seen_axes': 'Same training bodies, new validation goals; not new-body transfer',
            'screening': 'Validation offsets accepted only if every twin in all four factorial conditions succeeds; all rejected outcomes retained',
            'test': 'Reserved geometry only; no rollout or screening',
            'scope': 'Single fixed posture contrast, not broad posture coverage or isolated Cartesian-state control'}}
    (root/'protocol.json').write_text(json.dumps(m['protocol'], indent=2))
    source = root/'source'; source.mkdir()
    for p in Path(__file__).parent.glob('*.py'): shutil.copyfile(p, source/p.name)
    pairs = {}
    for si, split in enumerate(('train', 'validation', 'test')):
        for ci, config in enumerate(PAIRS[split]):
            for instance in range(2):
                pid = si*4 + ci*2 + instance
                arms = [replace(a, seed=84000+pid*2+side, home=home(a.dof)) for side, a in enumerate(pair(split, config, instance))]
                pairs[pid] = arms
                m['arms'].extend(a.manifest() for a in arms)
                m['pairs'].append({'id': pid, 'split': split, 'bodies': [a.seed for a in arms]})
                for a in arms:
                    m['axis_catalog'][a.family] = AXES[a.family]
                    (root/f'body_{a.seed}.xml').write_text(MixedReach(a).xml)
                if split != 'test':
                    check_pair(arms); check_pair([replace(a, home=home(a.dof, True)) for a in arms])
    gid = 0
    def candidate(rng):
        while True:
            delta = rng.uniform([-.1, -.1, -.1], [.1, .1, -.02])
            if .06 <= np.linalg.norm(delta) <= .12: return delta
    def audit(row):
        with (root/'feasibility.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    for pid in range(4):
        arms = pairs[pid]; origin = check_pair(arms); rng = np.random.default_rng(101000+pid)
        accepted = 0
        for attempt in range(3200):
            goal = origin+candidate(rng)
            trajectories = [expert_rollout(MixedReach(a, synchronize=True), 0, goal, spread=0) for a in arms]
            audit({'role': 'train', 'pair': pid, 'attempt': attempt, 'goal': goal.tolist(), 'outcomes': [r for _, r in trajectories]})
            if not all(r['success'] for _, r in trajectories): continue
            m['goals']['train'].append({'id': gid, 'pair': pid, 'xyz': goal.tolist(), 'seed': 0})
            for a, (trajectory, result) in zip(arms, trajectories):
                filename = f'episode_{a.seed}_{gid}.npz'; np.savez_compressed(root/filename, **trajectory)
                m['episodes'].append({'file': filename, 'body_seed': a.seed, 'family': a.family, 'dof': a.dof,
                    'split': 'train', 'goal_id': gid, 'variant': 'paired_zero_spread', **result})
            accepted += 1; gid += 1
            if accepted == 32: break
        assert accepted == 32
        for index in range(16):
            contexts = [calibration(a, 102000+pid*100+index) for a in arms]
            np.testing.assert_array_equal(contexts[0][1], contexts[1][1])
            for a, (history, commands) in zip(arms, contexts):
                filename = f'calibration_{a.seed}_{index}.npy'; np.save(root/filename, history)
                np.savez_compressed(root/filename.replace('.npy', '.npz'), history=history, commands=commands)
                m['calibrations'].append({'file': filename, 'body_seed': a.seed, 'split': 'train'})
        # Same physical parameters across seen/new kinematic pair; cross both postures.
        configurations = []
        for new_axes in (False, True):
            pair_id = pid+4 if new_axes else pid
            for shifted in (False, True):
                aa = [replace(a, home=home(a.dof, shifted)) for a in pairs[pair_id]]
                configurations.append((pair_id, shifted, aa, check_pair(aa)))
        rng = np.random.default_rng(103000+pid); accepted = 0
        for attempt in range(1200):
            delta = candidate(rng); outcomes = []
            for pair_id, shifted, aa, origin in configurations:
                for a in aa:
                    _, result = expert_rollout(MixedReach(a, synchronize=True), 0, origin+delta, spread=0)
                    outcomes.append({'body': a.seed, 'shifted': shifted, **result})
            audit({'role': 'validation', 'physical_group': pid, 'attempt': attempt, 'offset': delta.tolist(), 'outcomes': outcomes})
            if not all(r['success'] for r in outcomes): continue
            for pair_id, shifted, aa, origin in configurations:
                m['goals']['validation'].append({'id': gid, 'pair': pair_id, 'xyz': (origin+delta).tolist(),
                    'home': aa[0].home, 'seed': 0, 'physical_group': pid, 'offset_id': accepted,
                    'condition': ('new_axes' if pair_id >= 4 else 'seen_axes')+('_shifted_posture' if shifted else '_seen_posture')})
                gid += 1
            accepted += 1
            if accepted == 12: break
        assert accepted == 12
        print('physical group', pid, 'complete', flush=True)
    m['asset_hashes'] = {p.name: fingerprint(p) for p in root.iterdir() if p.is_file()}
    m['source_hashes'] = {p.name: fingerprint(p) for p in source.glob('*.py')}
    (root/'manifest.json').write_text(json.dumps(m, indent=2))
    (root/'manifest.sha256').write_text(fingerprint(root/'manifest.json'))
    print('completed', len(m['episodes']), 'training episodes;', len(m['goals']['validation'])*2, 'validation cases', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True); build(**vars(p.parse_args()))
