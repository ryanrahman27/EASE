"""Mixed-topology reaching dataset: robot metadata stays outside policy tensors."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .benchmark import fingerprint
from .mixed_sim import MixedReach, explore, sample_arm, token


def expert_rollout(env, seed, goal, spread=.05):
    obs = env.reset(seed, goal, spread=spread)
    joints, tasks, actions, histories = [], [], [], []
    done = False
    while not done:
        after, done, info = env.step(env.expert())
        joints.append(obs["joints"])
        tasks.append(obs["task"])
        actions.append(info["executed_action"])
        histories.append(token(obs, info["executed_action"], after))
        obs = after
    return {"joints": np.asarray(joints), "task": np.asarray(tasks), "action": np.asarray(actions),
            "history": np.asarray(histories)}, {k: info[k] for k in ["success", "collision", "error"]}


def build(output, train_goals=40, eval_goals=8, seed=20261018):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    arms = [sample_arm(seed+i, "six", "train") for i in range(3)]
    arms += [sample_arm(seed+100+i, "seven", "train") for i in range(3)]
    arms += [sample_arm(seed+1000, "six", "validation"), sample_arm(seed+1100, "seven", "validation"),
             sample_arm(seed+1200, "six_novel", "validation_topology"), sample_arm(seed+2000, "six", "test")]
    envs = [MixedReach(arm) for arm in arms]
    manifest = {"schema": "ease_mixed_v1", "seed": seed, "arms": [arm.manifest() for arm in arms],
                "goals": {}, "episodes": [], "calibrations": [], "feasibility_audit": [],
                "observation": "joint q,dq; measured ee XYZ; goal XYZ; joint validity",
                "privileged_inputs": False, "success_threshold_m": .03, "hold_steps": 5,
                "control_hz": 25, "action_limit_rad": .04, "history_steps": 32,
                "scope": "Procedural serial arms; six_novel axis sequence excluded from training"}
    for env in envs:
        (output / f"body_{env.arm.seed}.xml").write_text(env.xml)
    rng = np.random.default_rng(seed)
    for role, count, offset in [("train", train_goals, 100000), ("validation", eval_goals, 200000),
                                ("test", eval_goals, 300000)]:
        accepted = []
        for attempt in range(count*100):
            xyz = rng.uniform([-.20, -.18, .78], [.25, .18, .95])
            task_seed = seed+offset+attempt
            cache, failures = [], []
            for env in envs:
                trajectory, result = expert_rollout(env, task_seed, xyz)
                cache.append((env, trajectory, result))
                if not result["success"]:
                    failures.append(env.arm.seed)
            manifest["feasibility_audit"].append({"role": role, "candidate": attempt,
                "goal": xyz.tolist(), "seed": task_seed, "failed_arms": failures})
            if failures:
                continue
            goal_id = len(accepted)
            accepted.append({"id": goal_id, "xyz": xyz.tolist(), "seed": task_seed})
            if role == "train":
                for env, trajectory, result in cache:
                    if env.arm.split != "train":
                        continue
                    for variant in ["nominal", "broad_start"]:
                        if variant == "broad_start":
                            trajectory, result = expert_rollout(env, task_seed+400000, xyz, spread=.35)
                        filename = f"{env.arm.seed}_{goal_id}_{variant}.npz"
                        np.savez_compressed(output / filename, **trajectory)
                        manifest["episodes"].append({"file": filename, "body_seed": env.arm.seed,
                            "split": "train", "dof": env.arm.dof, "goal_id": goal_id, "variant": variant, **result})
            if len(accepted) == count:
                break
        if len(accepted) != count:
            raise RuntimeError(f"Only {len(accepted)} feasible {role} goals")
        manifest["goals"][role] = accepted
        print(role, "accepted", count, "of", attempt+1, flush=True)
    for env in envs:
        if env.arm.split != "train":
            continue
        for index in range(8):
            context_seed = seed+500000+index
            filename = f"calibration_{env.arm.seed}_{index}.npy"
            np.save(output / filename, explore(env, context_seed))
            manifest["calibrations"].append({"file": filename, "body_seed": env.arm.seed, "split": "train"})
    manifest["asset_hashes"] = {p.name: fingerprint(p) for p in output.iterdir() if p.is_file()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (output / "manifest.sha256").write_text(fingerprint(output / "manifest.json"))
    print("training episodes", len(manifest["episodes"]), flush=True)


def load(directory):
    directory = Path(directory)
    if fingerprint(directory / "manifest.json") != (directory / "manifest.sha256").read_text():
        raise ValueError("Manifest changed")
    result = json.loads((directory / "manifest.json").read_text())
    if result["schema"] != "ease_mixed_v1":
        raise ValueError("Not a mixed-topology dataset")
    for name, digest in result["asset_hashes"].items():
        if fingerprint(directory / name) != digest:
            raise ValueError(f"Asset changed: {name}")
    return result


class MixedData:
    def __init__(self, directory):
        root = Path(directory)
        self.manifest = load(root)
        self.max_joints = max(arm["lengths"].__len__() for arm in self.manifest["arms"] if arm["split"] == "train")
        rows, tasks, masks, targets, valid, row_ep = [], [], [], [], [], []
        responses = []
        self.contexts, self.context_body, self.context_episode, self.episode_body = [], [], [], []
        for episode in self.manifest["episodes"]:
            if episode["split"] != "train" or not episode["success"] or episode["collision"]:
                continue
            eid = len(self.episode_body)
            self.episode_body.append(episode["body_seed"])
            with np.load(root / episode["file"]) as data:
                dof = data["joints"].shape[1]
                for step in range(len(data["action"])):
                    joint = np.zeros((self.max_joints, 2), np.float32)
                    joint[:dof] = data["joints"][step]
                    chunk = data["action"][step:step+16]
                    action = np.zeros((16, self.max_joints), np.float32)
                    action[:len(chunk), :dof] = chunk/.04
                    action_mask = np.zeros_like(action)
                    action_mask[:len(chunk), :dof] = 1
                    rows.append(joint)
                    tasks.append(data["task"][step])
                    masks.append(np.arange(self.max_joints) < dof)
                    targets.append(action)
                    valid.append(action_mask)
                    row_ep.append(eid)
                    responses.append(data["history"][step, 0, 8:11].copy())
                self.contexts.append(data["history"].copy())
                self.context_body.append(episode["body_seed"])
                self.context_episode.append(eid)
        for entry in self.manifest["calibrations"]:
            if entry["split"] != "train":
                raise ValueError("Nontraining calibration")
            self.contexts.append(np.load(root / entry["file"]))
            self.context_body.append(entry["body_seed"])
            self.context_episode.append(-1)
        self.tensors = tuple(torch.from_numpy(np.stack(values)) for values in (rows, tasks, masks, targets, valid))
        self.row_episode = np.asarray(row_ep)
        self.responses = torch.from_numpy(np.stack(responses))
        self.context_body, self.context_episode = np.asarray(self.context_body), np.asarray(self.context_episode)
        self.pools = {}
        for eid, body in enumerate(self.episode_body):
            allowed = (self.context_body == body) & (self.context_episode != eid)
            self.pools[eid] = [np.flatnonzero(allowed & (self.context_episode == -1)),
                               np.flatnonzero(allowed & (self.context_episode != -1))]
            if any(len(pool) == 0 for pool in self.pools[eid]):
                raise ValueError("Need at least two task episodes and calibration per body")

    def histories(self, indices, rng):
        history = np.zeros((len(indices), 32, self.max_joints, 11), np.float32)
        lengths = np.zeros(len(indices), np.int64)
        for row, idx in enumerate(indices):
            pool = self.pools[self.row_episode[idx]][rng.integers(2)]
            context = self.contexts[rng.choice(pool)]
            end = int(rng.integers(1, len(context)+1))
            length = int(rng.integers(0, min(end, 32)+1))
            if length:
                history[row, :length, :context.shape[1]] = context[end-length:end]
            lengths[row] = length
        return torch.from_numpy(history), torch.from_numpy(lengths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-goals", type=int, default=40)
    parser.add_argument("--eval-goals", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261018)
    build(**vars(parser.parse_args()))
