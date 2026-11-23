"""Post-hoc control diagnostics, not a deployable privileged-policy baseline."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .baseline import FlowPolicy, HORIZON
from .benchmark import fingerprint, load_manifest
from .sim import Body, ReachEnv


def audit(data, comparison, output, device="cpu"):
    manifest = load_manifest(data, verify_assets=True)
    torch.set_num_threads(4)
    cases = [(Body(**body), goal) for body in manifest["bodies"] if body["split"] == "validation"
             for goal in manifest["goals"]["validation"]]
    noise = torch.stack([torch.randn(150, HORIZON, 7,
                         generator=torch.Generator().manual_seed(goal["seed"]+600000))
                         for _, goal in cases], dim=1).to(device)
    results = []
    for seed in range(3):
        folder = Path(comparison) / f"baseline_seed{seed}"
        metadata = json.loads((folder / "training.json").read_text())
        if metadata["manifest_sha256"] != fingerprint(Path(data) / "manifest.json"):
            raise ValueError("Benchmark mismatch")
        if metadata["policy_sha256"] != fingerprint(folder / "policy.pt"):
            raise ValueError("Checkpoint mismatch")
        model = FlowPolicy().to(device)
        model.load_state_dict(torch.load(folder / "policy.pt",
                                        map_location=device, weights_only=True))
        model.eval()
        for mode in ["raw", "zero_last_joint"]:
            envs = [ReachEnv(body) for body, _ in cases]
            obs = np.stack([env.reset(goal["seed"], goal["xyz"]) for env, (_, goal) in zip(envs, cases)])
            active = np.ones(len(cases), dtype=bool)
            max_z = np.zeros((len(cases), 20))
            last_joint_drift = np.zeros(len(cases))
            initial_q = obs[:, :7].copy()
            errors = np.zeros(len(cases))
            count = np.zeros(len(cases))
            for step in range(150):
                tensor = torch.from_numpy(obs).to(device)
                z = ((tensor-model.obs_mean)/model.obs_std).abs().cpu().numpy()
                max_z[active] = np.maximum(max_z[active], z[active])
                action = model.sample(tensor, noise=noise[step]).cpu().numpy()[:, 0]
                if mode == "zero_last_joint":
                    # The current fixture's final joint rotates around the terminal
                    # link axis and does not change endpoint XYZ. Diagnostic only.
                    action[:, -1] = 0
                for idx in np.flatnonzero(active):
                    errors[idx] += np.mean((action[idx] - envs[idx].expert())**2)
                    count[idx] += 1
                    obs[idx], done, info = envs[idx].step(action[idx])
                    last_joint_drift[idx] = max(last_joint_drift[idx], abs(obs[idx, 6]-initial_q[idx, 6]))
                    if done:
                        results.append({"seed": seed, "checkpoint_sha256": metadata["policy_sha256"],
                            "mode": mode, "body_seed": cases[idx][0].seed,
                            "goal_id": cases[idx][1]["id"], "success": info["success"],
                            "error": info["error"], "steps": envs[idx].steps,
                            "max_abs_standardized_observation": max_z[idx].tolist(),
                            "max_last_joint_drift_rad": last_joint_drift[idx],
                            "visited_state_action_rmse_rad": float(np.sqrt(errors[idx]/count[idx]))})
                        active[idx] = False
                if not active.any():
                    break
            rows = [r for r in results if r["seed"] == seed and r["mode"] == mode]
            print(seed, mode, "success", sum(r["success"] for r in rows), "/", len(rows),
                  "median last joint drift", np.median([r["max_last_joint_drift_rad"] for r in rows]), flush=True)
    with Path(output).open("x") as file:
        json.dump({"comparison": str(comparison), "records": results,
                   "purpose": "Validation diagnostic; zero_last_joint is a fixture-specific intervention"}, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    audit(**vars(parser.parse_args()))
