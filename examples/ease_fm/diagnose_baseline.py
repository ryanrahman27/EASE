"""Separate training fit, new-goal transfer, new-body transfer and solver resolution.

This is a validation diagnostic, not part of the prespecified primary comparison.
It deliberately does not inspect final-test bodies or final-test goals.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .baseline import FlowPolicy, HORIZON
from .benchmark import fingerprint, load_manifest
from .sim import Body, ReachEnv


def diagnose(data, checkpoint, output, goals=6, device="cpu"):
    torch.set_num_threads(4)
    manifest = load_manifest(data, verify_assets=True)
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint.parent / "training.json").read_text())
    if metadata["method"] != "baseline":
        raise ValueError("Diagnostic requires a history-free checkpoint")
    if metadata["manifest_sha256"] != fingerprint(Path(data) / "manifest.json"):
        raise ValueError("Benchmark mismatch")
    if metadata["policy_sha256"] != fingerprint(checkpoint):
        raise ValueError("Checkpoint mismatch")
    model = FlowPolicy().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    cases = []
    for condition, body_split, goal_split in [("training_fit", "train", "train"),
                                              ("new_goals", "train", "validation"),
                                              ("new_bodies", "validation", "validation")]:
        bodies = [Body(**b) for b in manifest["bodies"] if b["split"] == body_split][:3]
        for body in bodies:
            for goal in manifest["goals"][goal_split][:goals]:
                cases.append((condition, body, goal))
    noise = torch.stack([torch.randn(150, HORIZON, 7,
                         generator=torch.Generator().manual_seed(goal["seed"]+600000))
                         for _, _, goal in cases], dim=1).to(device)
    records = []
    for integration_steps in [10, 50]:
        envs = [ReachEnv(body) for _, body, _ in cases]
        observation = np.stack([env.reset(goal["seed"], goal["xyz"])
                                for env, (_, _, goal) in zip(envs, cases)])
        active = np.ones(len(cases), dtype=bool)
        for step in range(150):
            actions = model.sample(torch.from_numpy(observation).to(device), steps=integration_steps,
                                   noise=noise[step]).cpu().numpy()[:, 0]
            for idx in np.flatnonzero(active):
                observation[idx], done, info = envs[idx].step(actions[idx])
                if done:
                    condition, body, goal = cases[idx]
                    records.append({"condition": condition, "integration_steps": integration_steps,
                        "body_seed": body.seed, "goal_id": goal["id"], "success": info["success"],
                        "error": info["error"], "collision": envs[idx].episode_collision})
                    active[idx] = False
            if not active.any():
                break
        for condition in ["training_fit", "new_goals", "new_bodies"]:
            selected = [r for r in records if r["condition"] == condition and r["integration_steps"] == integration_steps]
            print(integration_steps, condition, sum(r["success"] for r in selected), "/", len(selected), flush=True)
    with Path(output).open("x") as file:
        json.dump({"training_steps": metadata["steps"], "training_seed": metadata["seed"],
                   "checkpoint_sha256": metadata["policy_sha256"], "records": records,
                   "purpose": "Post-hoc validation diagnostic; no final-test use"}, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--goals", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    diagnose(**vars(parser.parse_args()))
