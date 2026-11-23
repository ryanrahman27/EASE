"""E22 fixture audit: identical initial observations, distinct joint axes."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from .mixed_sim import AXES, MixedReach, sample_arm

PAIRS = {
    'train': [('ZYYXYX', 'ZYXYYX', 4), ('ZYYXYXY', 'ZYXYYXX', 4)],
    'validation': [('ZYYXXY', 'ZYXXXY', 3), ('ZYXXYZX', 'ZYYXYZX', 4)],
    'test': [('ZYXYYY', 'ZYYXYY', 4), ('ZYXYXYY', 'ZYYYXYY', 3)],
}


def pair(split, sequences, instance):
    first, second, bend = sequences
    prefix = 'six' if len(first) == 6 else 'seven'
    home = np.zeros(len(first)); home[1] = .45; home[bend] = -.65
    arms = []
    for side, sequence in enumerate((first, second)):
        name = f'{prefix}_amb_{split}_{side}'
        AXES[name] = [dict(X='1 0 0', Y='0 1 0', Z='0 0 1')[a] for a in sequence]
        arms.append(replace(sample_arm(81000 + instance, name, split), home=tuple(home)))
    return arms


def audit(arms):
    envs = [MixedReach(a, synchronize=True) for a in arms]
    observations = [e.reset(0, [0, 0, .9], spread=0) for e in envs]
    identical = all(np.array_equal(observations[0][k], observations[1][k]) for k in observations[0])
    assert identical, 'Initial student observations differ'
    initial = envs[0].data.site_xpos[0].copy()
    jacobians = []
    for env in envs:
        jac = np.zeros((3, env.arm.dof))
        mujoco.mj_jacSite(env.model, env.data, jac, None, 0)
        jacobians.append(jac)
    ranks = [int(np.linalg.matrix_rank(j, tol=1e-7)) for j in jacobians]
    assert ranks == [3, 3], ranks
    trajectories = []
    for env in envs:
        env.reset(0, initial, spread=0)
        trajectory = []
        for joint in range(env.arm.dof):
            action = np.zeros(env.arm.dof); action[joint] = .01
            for _ in range(4):
                obs, _, info = env.step(action)
                assert not info['collision']
                trajectory.append(obs['task'][:3].copy())
        trajectories.append(np.asarray(trajectory))
    separation = np.linalg.norm(trajectories[0]-trajectories[1], axis=1)
    assert separation.max() > 1e-4
    candidates = []
    for offset in ([.04, 0, -.04], [0, .04, -.04], [-.04, 0, -.04], [0, -.04, -.04]):
        goal = initial + offset
        outcomes = []
        for env in envs:
            env.reset(0, goal, spread=0)
            done = False
            while not done:
                _, done, info = env.step(env.expert())
            outcomes.append({k: info[k] for k in ('success', 'error', 'collision')})
        candidates.append({'goal': goal.tolist(), 'outcomes': outcomes})
    assert any(all(o['success'] for o in c['outcomes']) for c in candidates)
    return {'identical_initial_observations': identical, 'jacobian_ranks': ranks,
            'jacobian_difference_norm': float(np.linalg.norm(jacobians[0]-jacobians[1])),
            'probe_max_ee_separation_m': float(separation.max()), 'goal_audit': candidates}


def run(output):
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, root/'source.py')
    shutil.copyfile(Path(__file__).with_name('mixed_sim.py'), root/'mixed_sim.py')
    rows = []
    for split, configurations in PAIRS.items():
        for configuration in configurations:
            for instance in range(2):
                arms = pair(split, configuration, instance)
                result = audit(arms)
                name = f'{split}_{arms[0].dof}_{instance}'
                for side, arm in enumerate(arms):
                    (root/f'{name}_{side}.xml').write_text(MixedReach(arm).xml)
                rows.append({'name': name, 'axes': configuration[:2],
                             'arms': [a.manifest() for a in arms], **result})
                print(name, result['probe_max_ee_separation_m'], flush=True)
    (root/'audit.json').write_text(json.dumps({'experiment': 'E22', 'pairs': rows,
        'scope': 'Fixture feasibility only; no learned policy trained or evaluated. Test geometry used for expert feasibility audit.',
        'limitations': 'Identical observations only within each pair at spread=0. Online observations may still permit reactive control without explicit history.'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    run(**vars(parser.parse_args()))
