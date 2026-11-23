"""Observation-only local kinematic identification and adaptive reaching diagnostic."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import numpy as np
import mujoco
from .paired_data import register, calibration
from .mixed_sim import Arm, MixedReach
from .benchmark import fingerprint


class IdentifiedController:
    """No robot description or simulator state enters this controller."""
    def __init__(self, history, online):
        dq = history[:, :, 3].astype(float)
        dx = history[:, 0, 8:11].astype(float)
        self.gram = dq.T @ dq + 1e-6*np.eye(dq.shape[1])
        self.cross = dq.T @ dx
        self.jac = np.linalg.solve(self.gram, self.cross).T
        self.bias = (history[:, :, 2]*.04-dq).mean(0)
        self.online = online

    def action(self, observation):
        error = observation['task'][3:]-observation['task'][:3]
        delta = self.jac.T @ np.linalg.solve(self.jac @ self.jac.T + .015**2*np.eye(3), .5*error)
        return np.clip(delta+self.bias, -.04, .04)

    def update(self, before, after, executed):
        if not self.online: return
        dq = (after['joints'][:, 0]-before['joints'][:, 0]).astype(float)
        dx = (after['task'][:3]-before['task'][:3]).astype(float)
        self.gram = .98*self.gram + np.outer(dq, dq) + 2e-8*np.eye(len(dq))
        self.cross = .98*self.cross + np.outer(dq, dx)
        self.jac = np.linalg.solve(self.gram, self.cross).T
        self.bias = .8*self.bias + .2*(executed-dq)


def run(data, output):
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    manifest = register(data)
    protocol = dict(experiment='E28', manifest_sha256=fingerprint(Path(data)/'manifest.json'),
        methods=['identified_frozen', 'identified_online', 'twin_online'],
        calibration_seeds=[930000, 940000, 950000], calibration_steps=32,
        ridge=1e-6, forgetting=.98, bias_update=.2, control_gain=.5, damping=.015,
        scope='Analytical observation-only identification/control diagnostic, not a trained Transformer result. Fixed E27 cases and success criterion. No parameter selection on validation.',
        privileged='True Jacobian is recorded only for diagnosis, never passed to the controller.',
        test_evaluated=False)
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2))
    source = root/'source'; source.mkdir()
    for p in Path(__file__).parent.glob('*.py'): shutil.copyfile(p, source/p.name)
    arms = {a['seed']: Arm(**a) for a in manifest['arms']}
    pairs = {p['id']: p for p in manifest['pairs']}
    records = []; diagnostics = []; cache = {}
    for split in ('train', 'validation'):
        for probe_seed in (protocol['calibration_seeds'] if split == 'validation' else [930000]):
            for mode in protocol['methods']:
                selected = []
                for goal in manifest['goals'][split]:
                    for body in pairs[goal['pair']]['bodies']:
                        arm = arms[body]
                        if 'home' in goal: arm = replace(arm, home=tuple(goal['home']))
                        donor = replace(arms[next(b for b in pairs[goal['pair']]['bodies'] if b != body)], home=arm.home) if mode == 'twin_online' else arm
                        key = (donor.seed, tuple(donor.home), probe_seed)
                        if key not in cache: cache[key] = calibration(donor, probe_seed+goal['pair'])[0]
                        controller = IdentifiedController(cache[key], mode != 'identified_frozen')
                        env = MixedReach(arm, synchronize=True)
                        before = env.reset(0, goal['xyz'], spread=0)
                        true_jac = np.zeros((3, arm.dof)); mujoco.mj_jacSite(env.model, env.data, true_jac, None, 0)
                        if mode != 'twin_online':
                            diagnostics.append(dict(split=split, body_seed=body, goal_id=goal['id'], mode=mode,
                                probe_seed=probe_seed, condition=goal.get('condition', 'train'),
                                jacobian_relative_error=float(np.linalg.norm(controller.jac-true_jac)/np.linalg.norm(true_jac)),
                                excitation_condition=float(np.linalg.cond(controller.gram))))
                        for _ in range(150):
                            action = controller.action(before)
                            after, done, info = env.step(action)
                            controller.update(before, after, info['executed_action'])
                            before = after
                            if done: break
                        row = dict(split=split, probe_seed=probe_seed, mode=mode, body_seed=body,
                            dof=arm.dof, goal_id=goal['id'], condition=goal.get('condition', 'train'),
                            steps=env.steps, **{k: info[k] for k in ('success', 'error', 'collision')})
                        selected.append(row)
                records.extend(selected)
                (root/'records.json').write_text(json.dumps(records, indent=2))
                (root/'identification.json').write_text(json.dumps(diagnostics, indent=2))
                print(split, probe_seed, mode, sum(r['success'] for r in selected), '/', len(selected), flush=True)
    (root/'status.json').write_text(json.dumps(dict(status='complete', test_evaluated=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True); parser.add_argument('--output', required=True)
    run(**vars(parser.parse_args()))
