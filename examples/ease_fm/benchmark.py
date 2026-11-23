"""Frozen development benchmark with common Cartesian goals and separate validation.

Expert feasibility screening is benchmark construction, never a student input.
All rejected candidates are retained in the audit. Test trajectories are not collected.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from .sim import ACTION_LIMIT, CONTROL_HZ, SUCCESS_HOLD_STEPS, Body, ReachEnv, sample_body

SCHEMA = "ease_reach_v2"
HISTORY_LENGTH = 32


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transition_token(observation, action, next_observation):
    """Only measured state and executed command; no goal, ID or physical parameters."""
    before = np.asarray(observation)[..., :17]
    delta = np.asarray(next_observation)[..., :17] - before
    delta = delta.copy()
    delta[..., :7] = (delta[..., :7] + np.pi) % (2 * np.pi) - np.pi
    return np.concatenate([before, np.asarray(action) / ACTION_LIMIT, delta], axis=-1).astype(np.float32)


def calibration(env, seed, length=HISTORY_LENGTH):
    """Bounded excitation in a separate episode, without privileged expert calls."""
    rng = np.random.default_rng(seed)
    obs = env.reset(seed)
    tokens = []
    for step in range(length):
        if step % 4 == 0:
            command = rng.uniform(-.018, .018, 7)
        new_obs, _, info = env.step(command)
        if info["collision"]:
            raise RuntimeError(f"Calibration collision on body {env.body.seed}, seed {seed}")
        tokens.append(transition_token(obs, info["executed_action"], new_obs))
        obs = new_obs
    return np.asarray(tokens, dtype=np.float32)


def expert_episode(env, seed, goal):
    obs = env.reset(seed, goal)
    states, actions, next_states = [], [], []
    done = False
    while not done:
        new_obs, done, info = env.step(env.expert())
        states.append(obs)
        actions.append(info["executed_action"])
        next_states.append(new_obs)
        obs = new_obs
    return {"observation": np.asarray(states), "action": np.asarray(actions),
            "next_observation": np.asarray(next_states)}, {
        "success": info["success"], "collision": env.episode_collision,
        "error": info["error"], "steps": env.steps}


def build(output, train_bodies=10, validation_bodies=3, test_bodies=3,
          train_goals=60, eval_goals=12, seed=20261016):
    if min(train_bodies, validation_bodies, test_bodies, train_goals, eval_goals) < 1:
        raise ValueError("All counts must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bodies = [sample_body(seed+i, "train") for i in range(train_bodies)]
    bodies += [sample_body(seed+10000+i, "validation") for i in range(validation_bodies)]
    bodies += [sample_body(seed+20000+i, "interpolation") for i in range(test_bodies)]
    bodies += [sample_body(seed+30000+i, "extrapolation") for i in range(test_bodies)]
    envs = [ReachEnv(body) for body in bodies]
    for env in envs:
        (output / f"body_{env.body.seed}.xml").write_text(env.xml)
    manifest = {"schema": SCHEMA, "seed": seed, "control_hz": CONTROL_HZ,
                "action_limit": ACTION_LIMIT, "success_threshold_m": .03,
                "success_hold_steps": SUCCESS_HOLD_STEPS, "horizon": 150,
                "contact_check": "every physics substep; any contact invalidates success",
                "mujoco": mujoco.__version__, "numpy": np.__version__,
                "bodies": [b.manifest() for b in bodies], "goals": {}, "episodes": [],
                "calibrations": [], "feasibility_audit": [],
                "test_policy": "No test demonstrations; use validation for development",
                "goal_selection": "Candidate XYZ box, expert succeeds on every body; benchmark-conditioned distribution"}
    rng = np.random.default_rng(seed)
    for role, count, offset in [("train", train_goals, 100000), ("validation", eval_goals, 200000),
                                 ("test", eval_goals, 300000)]:
        accepted = []
        for attempt in range(count * 100):
            # Common task-space volume near the shared reachable workspace.
            goal = rng.uniform([-.25, -.22, .85], [.30, .22, 1.00])
            task_seed = seed + offset + attempt
            failures = []
            cached = []
            for env in envs:
                trajectory, outcome = expert_episode(env, task_seed, goal)
                cached.append((env.body, trajectory, outcome))
                if not outcome["success"]:
                    failures.append(env.body.seed)
            manifest["feasibility_audit"].append({"role": role, "candidate": attempt,
                "goal": goal.tolist(), "task_seed": task_seed, "failed_bodies": failures})
            if failures:
                continue
            goal_id = len(accepted)
            accepted.append({"id": goal_id, "xyz": goal.tolist(), "seed": task_seed})
            for body, trajectory, outcome in cached:
                if role == "train" and body.split == "train":
                    filename = f"train_{body.seed}_{goal_id:04d}.npz"
                    np.savez_compressed(output / filename, **trajectory)
                    manifest["episodes"].append({"file": filename, "body_seed": body.seed,
                        "split": "train", "goal_id": goal_id, "episode_seed": task_seed, **outcome})
            if len(accepted) == count:
                break
        if len(accepted) != count:
            raise RuntimeError(f"Only {len(accepted)} feasible {role} goals; inspect workspace bounds")
        manifest["goals"][role] = accepted
        print(role, "accepted", len(accepted), "of", attempt+1, "candidates", flush=True)
    # Training-only calibration episodes. Validation/test contexts are acquired on demand.
    for env in envs:
        if env.body.split != "train":
            continue
        for index in range(8):
            context_seed = seed + 400000 + index
            tokens = calibration(env, context_seed)
            filename = f"calibration_{env.body.seed}_{index}.npy"
            np.save(output / filename, tokens)
            manifest["calibrations"].append({"file": filename, "body_seed": env.body.seed,
                                               "seed": context_seed, "split": "train"})
    manifest["asset_hashes"] = {p.name: fingerprint(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "manifest.sha256").write_text(fingerprint(output / "manifest.json"))
    print("benchmark", output, "episodes", len(manifest["episodes"]), flush=True)


def load_manifest(directory, verify_assets=False):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["schema"] != SCHEMA:
        raise ValueError("Requires a v2 benchmark; do not mix pilot contracts")
    expected = (directory / "manifest.sha256").read_text().strip()
    if fingerprint(directory / "manifest.json") != expected:
        raise ValueError("Benchmark manifest hash mismatch")
    if verify_assets:
        for name, digest in manifest["asset_hashes"].items():
            if fingerprint(directory / name) != digest:
                raise ValueError(f"Benchmark asset hash mismatch: {name}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-bodies", type=int, default=10)
    parser.add_argument("--validation-bodies", type=int, default=3)
    parser.add_argument("--test-bodies", type=int, default=3)
    parser.add_argument("--train-goals", type=int, default=60)
    parser.add_argument("--eval-goals", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20261016)
    build(**vars(parser.parse_args()))
