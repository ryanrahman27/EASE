"""E14: frozen E12 checkpoints on unseen validation arms; no weight updates."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .benchmark import fingerprint
from .joint_policy import JointFlowPolicy
from .mixed_data import load
from .mixed_sim import Arm, MixedReach, explore
from .full_control import commands

def evaluate(data, checkpoint, output, device="cpu"):
    torch.set_num_threads(4)
    manifest = load(data)
    if not manifest.get("synchronized_observations", False):
        raise ValueError("Synchronized observations required")
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint.parent / "training.json").read_text())
    if metadata["manifest_sha256"] != fingerprint(Path(data)/"manifest.json"):
        raise ValueError("Dataset mismatch")
    if metadata["policy_sha256"] != fingerprint(checkpoint):
        raise ValueError("Checkpoint mismatch")
    history = True
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
                context = explore(MixedReach(arm, synchronize=True), goal["seed"]+500000, length)
                if mode == "wrong32":
                    donor = next(Arm(**item) for item in manifest["arms"]
                                 if item["split"] == "train" and len(item["lengths"]) == arm.dof)
                    context = explore(MixedReach(donor, synchronize=True), goal["seed"]+500000, length)
                contexts[idx, :length, :arm.dof] = context
            donors.append(None if donor is None else donor.seed)
        with torch.no_grad():
            latent = model.encode(torch.from_numpy(contexts).to(device),
                                  torch.full((len(cases),), length, device=device), joint_mask) if history else None
        envs = [MixedReach(arm, synchronize=True) for arm, _ in cases]
        observations = [env.reset(goal["seed"], goal["xyz"]) for env, (_, goal) in zip(envs, cases)]
        active = np.ones(len(cases), dtype=bool)
        for step in range(150):
            joints = np.zeros((len(cases), max_joints, 2), np.float32)
            for idx, observation in enumerate(observations):
                joints[idx, :len(observation["joints"])] = observation["joints"]
            task = np.stack([observation["task"] for observation in observations])
            actions = commands(model, metadata["kind"], torch.from_numpy(joints).to(device), torch.from_numpy(task).to(device),
                                   joint_mask, noise[step], latent).cpu().numpy()
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


def run(data, run, output, device="cuda"):
    root=Path(output); root.mkdir(parents=True, exist_ok=False)
    run=Path(run)
    protocol={"experiment":"E14", "date":"2026-10-17", "scope":"Single-seed frozen validation; one new body per family, eight validation goals each",
        "models":["direct","flow"], "modes":["empty","correct8","correct32","wrong32"],
        "primary":"Correct32 versus same-DoF wrong32 on novel-axis six-joint body; report familiar-family results separately",
        "weights_updated":False, "test_evaluated":False, "synchronized_observations":True,
        "context":"Separate bounded-excitation episode then reset; frozen context during task",
        "calibration_seed":"goal seed + 500000", "flow_noise_seed":"goal seed + 600000; paired across conditions",
        "checkpoints":{k:fingerprint(run/("recovery_"+k)/"policy.pt") for k in ["direct","flow"]},
        "manifest_sha256":fingerprint(Path(data)/"manifest.json")}
    (root/"protocol.json").write_text(json.dumps(protocol,indent=2))
    source=root/"source"; source.mkdir()
    for p in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(p,source/p.name)
    rows=[]
    for kind in protocol["models"]:
        print("Evaluating",kind,flush=True)
        evaluate(data,run/("recovery_"+kind)/"policy.pt",root/(kind+".json"),device)
        records=json.loads((root/(kind+".json")).read_text())["records"]
        for mode in protocol["modes"]:
            for family in ["six","seven","six_novel"]:
                cases=[r for r in records if r["mode"]==mode and r["family"]==family]
                assert len(cases)==8
                rows.append({"model":kind,"mode":mode,"family":family,"successes":sum(r["success"] for r in cases),
                    "trials":len(cases),"mean_error_m":float(np.mean([r["error"] for r in cases])),"collisions":sum(r["collision"] for r in cases)})
    (root/"summary.json").write_text(json.dumps(rows,indent=2))
    lines=["# E14: Frozen unseen-arm validation", "", "One training seed; one validation body per family. Final test untouched.", "",
           "| Model | Context | New six-joint | New seven-joint | Novel six-joint axes |", "|---|---|---:|---:|---:|"]
    for kind in protocol["models"]:
        for mode in protocol["modes"]:
            selected=[r for r in rows if r["model"]==kind and r["mode"]==mode]
            lines.append("| "+kind+" | "+mode+" | "+" | ".join(str(r["successes"])+"/8" for r in selected)+" |")
    (root/"summary.md").write_text("\n".join(lines)+"\n")
    (root/"source_hashes.json").write_text(json.dumps({p.name:fingerprint(p) for p in source.glob("*.py")},indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--run",required=True); p.add_argument("--output",required=True); p.add_argument("--device",default="cuda")
    run(**vars(p.parse_args()))
