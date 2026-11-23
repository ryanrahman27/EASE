"""Add expert recoveries from perturbed TRAINING states, keeping test tasks fixed."""
import argparse
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from .benchmark import fingerprint, load_manifest
from .sim import Body, ReachEnv


def perturb_state(env, observation, seed, position_radius=.35, velocity_radius=.15):
    """Perturb every joint in the simulator, then recompute measured task state."""
    rng = np.random.default_rng(seed)
    env.reset(seed, observation[-3:])
    env.data.qpos[:] = np.clip(observation[:7] + rng.uniform(-position_radius, position_radius, 7), -2.3, 2.3)
    env.data.qvel[:] = observation[7:14] + rng.uniform(-velocity_radius, velocity_radius, 7)
    env.data.ctrl[:] = env.data.qpos
    mujoco.mj_forward(env.model, env.data)
    return env.observation(), bool(env.data.ncon)


def rollout_expert(env, observation):
    states, actions, next_states = [], [], []
    done = False
    while not done:
        after, done, info = env.step(env.expert())
        states.append(observation)
        actions.append(info["executed_action"])
        next_states.append(after)
        observation = after
    return {"observation": np.asarray(states), "action": np.asarray(actions),
            "next_observation": np.asarray(next_states)}, {
        "success": info["success"], "collision": env.episode_collision,
        "error": info["error"], "steps": env.steps}


def build(data, output, seed=20261017):
    source, output = Path(data), Path(output)
    manifest = load_manifest(source, verify_assets=True)
    output.mkdir(parents=True, exist_ok=False)
    for name in manifest["asset_hashes"]:
        shutil.copyfile(source / name, output / name)
    original = list(manifest["episodes"])
    bodies = {body["seed"]: Body(**body) for body in manifest["bodies"]}
    manifest["parent_manifest_sha256"] = fingerprint(source / "manifest.json")
    manifest["dataset_variant"] = "v2_all_joint_recovery_v1"
    manifest["recovery_protocol"] = {"seed": seed, "position_radius_rad": .35,
        "velocity_radius_rad_s": .15, "episodes_per_original": 1,
        "source": "Random state in each original training episode; all seven joints perturbed",
        "state_source_splits": ["train"], "original_data_preserved": True}
    audit = []
    for episode_id, entry in enumerate(original):
        if entry["split"] != "train":
            raise ValueError("Recovery collection must use training episodes only")
        env = ReachEnv(bodies[entry["body_seed"]])
        with np.load(source / entry["file"]) as trajectory:
            rng = np.random.default_rng(seed + episode_id)
            index = int(rng.integers(len(trajectory["observation"])))
            state = trajectory["observation"][index].copy()
        for attempt in range(20):
            perturb_seed = seed + 100000 + episode_id * 20 + attempt
            observation, collision = perturb_state(env, state, perturb_seed)
            audit.append({"source_episode": entry["file"], "source_step": index,
                          "seed": perturb_seed, "initial_collision": collision})
            if not collision:
                break
        else:
            raise RuntimeError(f"Could not find noncolliding perturbation for {entry['file']}")
        trajectory, outcome = rollout_expert(env, observation)
        filename = f"recovery_{entry['body_seed']}_{episode_id:04d}.npz"
        np.savez_compressed(output / filename, **trajectory)
        manifest["episodes"].append({"file": filename, "body_seed": entry["body_seed"],
            "split": "train", "goal_id": entry["goal_id"], "episode_seed": perturb_seed,
            "source_episode": entry["file"], "source_step": index, "recovery": True, **outcome})
        if (episode_id+1) % 100 == 0:
            print("recovery episodes", episode_id+1, "/", len(original), flush=True)
    manifest["recovery_state_audit"] = audit
    manifest["asset_hashes"] = {p.name: fingerprint(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "manifest.sha256").write_text(fingerprint(output / "manifest.json"))
    recovery = [entry for entry in manifest["episodes"] if entry.get("recovery")]
    print("expert recovery success", sum(r["success"] for r in recovery), "/", len(recovery), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    build(**vars(parser.parse_args()))
