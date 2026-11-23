"""Collect episode-separated transitions and immutable body manifests."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .sim import ACTION_LIMIT, CONTROL_HZ, ReachEnv, sample_body


def collect(output, episodes=100, train_bodies=5, eval_bodies=2, seed=20261015):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    bodies = [sample_body(seed+i, "train") for i in range(train_bodies)]
    bodies += [sample_body(seed+10000+i, "interpolation") for i in range(eval_bodies)]
    bodies += [sample_body(seed+20000+i, "extrapolation") for i in range(eval_bodies)]
    manifest = {"schema": 1, "seed": seed, "control_hz": CONTROL_HZ, "action_limit": ACTION_LIMIT,
                "success_threshold_m": .03, "mujoco": mujoco.__version__, "numpy": np.__version__,
                "bodies": [b.manifest() for b in bodies], "episodes": []}
    for body in bodies:
        env = ReachEnv(body)
        (output / f"body_{body.seed}.xml").write_text(env.xml)
        for episode in range(episodes):
            # Paired seeds across bodies; goals are body-feasible, not identical Cartesian goals.
            episode_seed = seed + 100000 + episode
            obs = env.reset(episode_seed)
            states, actions, next_states = [], [], []
            done = False
            collision = False
            while not done:
                next_obs, done, info = env.step(env.expert())
                states.append(obs)
                actions.append(info["executed_action"])
                next_states.append(next_obs)
                obs = next_obs
                collision |= info["collision"]
            filename = f"{body.split}_{body.seed}_{episode:04d}.npz"
            np.savez_compressed(output / filename, observation=np.array(states), action=np.array(actions),
                                next_observation=np.array(next_states))
            manifest["episodes"].append({"file": filename, "body_seed": body.seed, "split": body.split,
                                         "episode_seed": episode_seed, "success": info["success"],
                                         "collision": collision, "error": info["error"], "steps": len(actions)})
        subset = [e for e in manifest["episodes"] if e["body_seed"] == body.seed]
        print(body.split, body.seed, "success", sum(e["success"] for e in subset), "/", episodes, flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--train-bodies", type=int, default=5)
    parser.add_argument("--eval-bodies", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    if min(args.episodes, args.train_bodies, args.eval_bodies) < 1:
        parser.error("Episode and body counts must be positive")
    collect(**vars(args))
