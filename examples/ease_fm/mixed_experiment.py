"""Shared-policy training on six/seven-DoF arms and held-out-arm history probes."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .benchmark import fingerprint
from .joint_policy import JointFlowPolicy
from .mixed_data import MixedData, load
from .mixed_sim import Arm, MixedReach, explore


def train(data, output, history=False, seed=0, steps=2000, device="cpu", auxiliary_weight=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    dataset = MixedData(data)
    joints, task, joint_mask, action, mask = [x.to(device) for x in dataset.tensors]
    torch.manual_seed(seed)
    model = JointFlowPolicy(history=history, auxiliary=auxiliary_weight is not None).to(device)
    responses = dataset.responses.to(device)
    if auxiliary_weight is not None:
        model.response_scale.copy_(responses.square().mean(0).sqrt().clamp_min(1e-4))
    model.joint_mean.copy_(joints[joint_mask].mean(0))
    model.joint_std.copy_(joints[joint_mask].std(0, unbiased=False).clamp_min(.01))
    model.task_mean.copy_(task.mean(0))
    model.task_std.copy_(task.std(0, unbiased=False).clamp_min(.01))
    if history:
        tokens = torch.from_numpy(np.concatenate([c.reshape(-1, 11) for c in dataset.contexts])).to(device)
        model.history_mean.copy_(tokens.mean(0))
        model.history_std.copy_(tokens.std(0, unbiased=False).clamp_min(.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    torch.manual_seed(seed+1000)
    rng = np.random.default_rng(seed+2000)
    losses, start = [], time.perf_counter()
    flow_losses, response_losses = [], []
    for step in range(steps):
        idx = torch.randint(len(joints), (32,), device=device)
        target, valid = action[idx], mask[idx]
        noise = torch.randn_like(target)
        tau = torch.rand(len(idx), device=device)
        latent = None
        if history:
            contexts, lengths = dataset.histories(idx.cpu().numpy(), rng)
            latent = model.encode(contexts.to(device), lengths.to(device), joint_mask[idx])
        prediction = model(noise.lerp(target, tau[:, None, None]), tau, joints[idx], task[idx], joint_mask[idx], latent)
        loss = ((prediction-(target-noise)).square()*valid).sum()/valid.sum()
        flow_losses.append(float(loss.detach()))
        if auxiliary_weight is not None:
            predicted_response = model.response(joints[idx], target[:, 0], joint_mask[idx], latent)
            response_loss = (predicted_response-responses[idx]/model.response_scale).square().mean()
            response_losses.append(float(response_loss.detach()))
            loss = loss + auxiliary_weight*response_loss
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 500 == 0:
            print("history" if history else "baseline", seed, "step", step, "loss", losses[-1], flush=True)
    torch.save(model.cpu().state_dict(), output / "policy.pt")
    source = output / "source"
    source.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(path, source / path.name)
    metadata = {"history": history, "seed": seed, "steps": steps, "batch_size": 32,
        "losses": losses, "flow_losses": flow_losses, "response_losses": response_losses,
        "auxiliary_weight": auxiliary_weight, "seconds": time.perf_counter()-start, "device": device,
        "torch": torch.__version__, "transitions": len(joints),
        "parameters": sum(p.numel() for p in model.parameters()),
        "manifest_sha256": fingerprint(Path(data)/"manifest.json"),
        "policy_sha256": fingerprint(output/"policy.pt"), "max_training_joints": dataset.max_joints,
        "source_hashes": {p.name: fingerprint(p) for p in source.glob("*.py")}}
    (output / "training.json").write_text(json.dumps(metadata, indent=2))
    print("saved", output, "seconds", round(metadata["seconds"], 1), flush=True)


def evaluate(data, checkpoint, output, device="cpu"):
    torch.set_num_threads(4)
    manifest = load(data)
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint.parent / "training.json").read_text())
    if metadata["manifest_sha256"] != fingerprint(Path(data)/"manifest.json"):
        raise ValueError("Dataset mismatch")
    if metadata["policy_sha256"] != fingerprint(checkpoint):
        raise ValueError("Checkpoint mismatch")
    history = metadata["history"]
    model = JointFlowPolicy(history=history, auxiliary=metadata.get("auxiliary_weight") is not None).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    arms = [Arm(**arm) for arm in manifest["arms"] if arm["split"] in {"validation", "validation_topology"}]
    cases = [(arm, goal) for arm in arms for goal in manifest["goals"]["validation"]]
    max_joints = max(arm.dof for arm, _ in cases)
    joint_mask = torch.tensor([[index < arm.dof for index in range(max_joints)] for arm, _ in cases], device=device)
    noise = torch.stack([torch.randn(150, 16, max_joints,
                         generator=torch.Generator().manual_seed(goal["seed"]+600000))
                         for _, goal in cases], dim=1).to(device)
    modes = [("empty", 0), ("correct8", 8), ("correct32", 32), ("wrong32", 32)] if history else [("none", 0)]
    records = []
    for mode, length in modes:
        contexts = np.zeros((len(cases), 32, max_joints, 11), np.float32)
        donors = []
        for idx, (arm, goal) in enumerate(cases):
            donor = None
            if length:
                context = explore(MixedReach(arm), goal["seed"]+500000, length)
                if mode == "wrong32":
                    donor = next(Arm(**item) for item in manifest["arms"]
                                 if item["split"] == "train" and len(item["lengths"]) == arm.dof)
                    context = explore(MixedReach(donor), goal["seed"]+500000, length)
                contexts[idx, :length, :arm.dof] = context
            donors.append(None if donor is None else donor.seed)
        with torch.no_grad():
            latent = model.encode(torch.from_numpy(contexts).to(device),
                                  torch.full((len(cases),), length, device=device), joint_mask) if history else None
        envs = [MixedReach(arm) for arm, _ in cases]
        observations = [env.reset(goal["seed"], goal["xyz"]) for env, (_, goal) in zip(envs, cases)]
        active = np.ones(len(cases), dtype=bool)
        for step in range(150):
            joints = np.zeros((len(cases), max_joints, 2), np.float32)
            for idx, observation in enumerate(observations):
                joints[idx, :len(observation["joints"])] = observation["joints"]
            task = np.stack([observation["task"] for observation in observations])
            actions = model.sample(torch.from_numpy(joints).to(device), torch.from_numpy(task).to(device),
                                   joint_mask, noise[step], latent).cpu().numpy()[:, 0]
            for idx in np.flatnonzero(active):
                arm, goal = cases[idx]
                observations[idx], done, info = envs[idx].step(actions[idx, :arm.dof])
                if done:
                    records.append({"body_seed": arm.seed, "family": arm.family, "dof": arm.dof,
                        "split": arm.split, "goal_id": goal["id"], "task_seed": goal["seed"],
                        "mode": mode, "calibration_actions": length, "calibration_seconds": length/25,
                        "donor_seed": donors[idx], "success": info["success"], "error": info["error"],
                        "collision": info["collision"], "steps": envs[idx].steps})
                    active[idx] = False
            if not active.any():
                break
        for family in ["six", "seven", "six_novel"]:
            subset = [r for r in records if r["mode"] == mode and r["family"] == family]
            print(mode, family, sum(r["success"] for r in subset), "/", len(subset), flush=True)
    with Path(output).open("x") as file:
        json.dump({"history": history, "training_seed": metadata["seed"],
            "manifest_sha256": metadata["manifest_sha256"], "checkpoint_sha256": metadata["policy_sha256"],
            "protocol": "Separate calibration episode then task reset; frozen context and weights; paired noise",
            "test_evaluated": False, "records": records}, file, indent=2)


def pilot(data, output, steps=2000, device="cpu"):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "protocol.json").write_text(json.dumps({"seed": 0, "steps": steps, "batch_size": 32,
        "primary": "correct32 versus wrong32 on held-out six-DoF arm", "test_evaluated": False,
        "scope": "Single-seed development pilot; not evidence for rapid adaptation", "device": device}, indent=2))
    for history in [False, True]:
        folder = root / ("history" if history else "baseline")
        train(data, folder, history=history, steps=steps, device=device)
        evaluate(data, folder / "policy.pt", folder / "validation.json", device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("pilot")
    run.add_argument("--data", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--steps", type=int, default=2000)
    run.add_argument("--device", default="cpu")
    args = vars(parser.parse_args())
    args.pop("command")
    pilot(**args)
