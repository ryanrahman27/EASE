"""Action-fit and conditioning diagnostics using training data only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .baseline import FlowPolicy, training_arrays
from .benchmark import fingerprint, load_manifest


def audit(data, comparison, output, device="cpu"):
    torch.set_num_threads(4)
    load_manifest(data, verify_assets=True)
    observation, actions, mask = training_arrays(data)
    indices = np.random.default_rng(913).choice(len(observation), min(1024, len(observation)), replace=False)
    obs, target, valid = [tensor[indices].to(device) for tensor in (observation, actions, mask)]
    result = []
    for seed in range(3):
        folder = Path(comparison) / f"baseline_seed{seed}"
        metadata = json.loads((folder / "training.json").read_text())
        if metadata["manifest_sha256"] != fingerprint(Path(data) / "manifest.json"):
            raise ValueError("Benchmark mismatch")
        if metadata["policy_sha256"] != fingerprint(folder / "policy.pt"):
            raise ValueError("Checkpoint mismatch")
        model = FlowPolicy().to(device)
        model.load_state_dict(torch.load(folder / "policy.pt", map_location=device, weights_only=True))
        model.eval()
        torch.manual_seed(500)
        noise = torch.randn_like(target)
        tau = torch.rand(len(obs), device=device)
        x = noise.lerp(target, tau[:, None, None])
        swapped = obs.clone()
        swapped[:, -3:] = obs.flip(0)[:, -3:]
        with torch.no_grad():
            velocity = model(x, tau, obs)
            original_loss = float(((velocity - (target-noise)).square() * valid).sum() / valid.sum())
            swapped_loss = float(((model(x, tau, swapped) - (target-noise)).square() * valid).sum() / valid.sum())
            action = model.sample(obs, noise=noise)[:, 0]
            other = model.sample(swapped, noise=noise)[:, 0]
            # No dropout: train/eval parity should hold even with fused attention.
            model.train()
            parity_error = float((model(x, tau, obs)-velocity).abs().max())
            model.eval()
        desired = target[:, 0] * .04
        row = {"seed": seed, "checkpoint_sha256": metadata["policy_sha256"],
               "training_steps": metadata["steps"], "flow_loss": original_loss,
               "goal_swapped_flow_loss": swapped_loss, "train_eval_max_error": parity_error,
               "first_action_rmse_rad": float((action-desired).square().mean().sqrt()),
               "per_joint_rmse_rad": (action-desired).square().mean(0).sqrt().cpu().tolist(),
               "goal_swap_action_rms_rad": float((action-other).square().mean().sqrt()),
               "expert_action_rms_rad": float(desired.square().mean().sqrt()),
               "predicted_action_rms_rad": float(action.square().mean().sqrt())}
        result.append(row)
        print(json.dumps(row), flush=True)
    with Path(output).open("x") as file:
        json.dump({"source": str(comparison), "data": str(data), "states": len(obs), "records": result}, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    audit(**vars(parser.parse_args()))
