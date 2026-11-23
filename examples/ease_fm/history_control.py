"""Analytical system-identification sanity check; NOT the learned FM policy.

Estimates a local Jacobian from measured transitions only. This checks whether
the history/sensing interface can support control before attributing failures to
neural adaptation. No MuJoCo Jacobians or physical parameters are accessed here.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .mixed_data import load
from .mixed_sim import Arm, MixedReach, explore, token


def estimate_jacobian(history, previous=None, ridge=1e-5):
    history = np.asarray(history)
    changes = history[:, :, 3].astype(float)
    displacement = history[:, 0, 8:11].astype(float)
    count = changes.shape[1]
    prior = np.zeros((3, count)) if previous is None else previous
    return np.linalg.solve(changes.T@changes + ridge*np.eye(count),
                           changes.T@displacement + ridge*prior.T).T


def evaluate(data, output):
    manifest = load(data)
    records = []
    for item in manifest["arms"]:
        if item["split"] not in {"validation", "validation_topology"}:
            continue
        arm = Arm(**item)
        for goal in manifest["goals"]["validation"]:
            history = list(explore(MixedReach(arm), goal["seed"]+500000, 32))
            jacobian = estimate_jacobian(history)
            env = MixedReach(arm)
            obs = env.reset(goal["seed"], goal["xyz"])
            done = False
            while not done:
                # Residual servo bias is also inferred from executed commands and
                # measured movement, never from gravity or actuator parameters.
                recent = np.asarray(history[-16:])
                bias = np.mean(recent[:, :, 2]*.04-recent[:, :, 3], axis=0)
                error = obs["task"][3:]-obs["task"][:3]
                action = jacobian.T@np.linalg.solve(jacobian@jacobian.T+.025**2*np.eye(3), .3*error)
                action = np.clip(action + bias, -.04, .04)
                after, done, info = env.step(action)
                history.append(token(obs, info["executed_action"], after))
                history = history[-32:]
                jacobian = estimate_jacobian(history, jacobian)
                obs = after
            records.append({"family": arm.family, "body_seed": arm.seed, "goal_id": goal["id"],
                "success": info["success"], "error": info["error"], "collision": info["collision"],
                "steps": env.steps, "calibration_actions": 32})
        rows = [r for r in records if r["body_seed"] == arm.seed]
        print("history-only analytical control", arm.family, sum(r["success"] for r in rows), "/", len(rows), flush=True)
    with Path(output).open("x") as file:
        json.dump({"method": "Online ridge local Jacobian + measured servo residual",
            "not_neural_policy": True, "privileged_parameters_or_jacobians": False,
            "context_updates_during_task": True, "records": records}, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    evaluate(**vars(parser.parse_args()))
