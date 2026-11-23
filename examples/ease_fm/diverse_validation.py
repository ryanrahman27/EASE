"""Expanded-family validation with 8/16/32-history interventions."""
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
    modes = [("empty", 0)] + [(prefix+str(n), n) for n in [8,16,32] for prefix in ["correct","wrong"]]
    records = []
    for mode, length in modes:
        contexts = np.zeros((len(cases), 32, max_joints, 11), np.float32)
        donors = []
        for idx, (arm, goal) in enumerate(cases):
            donor = None
            if length:
                context = explore(MixedReach(arm, synchronize=True), goal["seed"]+500000, length)
                if mode.startswith("wrong"):
                    donor = next(Arm(**item) for item in manifest["arms"]
                                 if item["split"] == "train" and len(item["lengths"]) == arm.dof and item["family"] != arm.family)
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
        for family in sorted({arm.family for arm in arms}):
            subset = [r for r in records if r["mode"] == mode and r["family"] == family]
            print(mode, family, sum(r["success"] for r in subset), "/", len(subset), flush=True)
    with Path(output).open("x") as file:
        json.dump({"history": history, "training_seed": metadata["seed"],
            "manifest_sha256": metadata["manifest_sha256"], "checkpoint_sha256": metadata["policy_sha256"],
            "protocol": "Separate calibration episode then task reset; frozen context and weights; paired noise",
            "test_evaluated": False, "records": records}, file, indent=2)


def run(data, run, output, device="cuda", diagnostic=False):
    root=Path(output); run=Path(run)
    gate=json.loads((run/"recovery_direct/gate.json").read_text())
    if not gate["passed"] and not diagnostic:
        raise ValueError("Training gate failed; inspect training control before validation. Explicit --diagnostic labels a confounded probe.")
    root.mkdir(parents=True,exist_ok=False)
    protocol={"scope":"Expanded-family frozen validation", "training_gate_passed":gate["passed"],
        "diagnostic_despite_failed_gate":diagnostic and not gate["passed"],"weights_updated":False,"final_test_evaluated":False,
        "modes":["empty","correct8","wrong8","correct16","wrong16","correct32","wrong32"],
        "wrong_donor":"First training body of same joint count and a DIFFERENT axis family",
        "primary":"Correct history versus both empty and same-DoF wrong-arrangement history, aggregated by held-out family and body",
        "models":["direct","flow"],"manifest_sha256":fingerprint(Path(data)/"manifest.json"),
        "checkpoints":{k:fingerprint(run/("recovery_"+k)/"policy.pt") for k in ["direct","flow"]}}
    (root/"protocol.json").write_text(json.dumps(protocol,indent=2))
    source=root/"source"; source.mkdir()
    for p in Path(__file__).parent.glob("*.py"): shutil.copyfile(p,source/p.name)
    summary=[]
    for kind in protocol["models"]:
        evaluate(data,run/("recovery_"+kind)/"policy.pt",root/(kind+".json"),device)
        records=json.loads((root/(kind+".json")).read_text())["records"]
        for mode in protocol["modes"]:
            for family in sorted({r["family"] for r in records}):
                subset=[r for r in records if r["mode"]==mode and r["family"]==family]
                summary.append({"model":kind,"mode":mode,"family":family,"split":subset[0]["split"],
                    "successes":sum(r["success"] for r in subset),"trials":len(subset),
                    "mean_error_m":float(np.mean([r["error"] for r in subset]))})
    (root/"summary.json").write_text(json.dumps(summary,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--run",required=True)
    p.add_argument("--output",required=True); p.add_argument("--device",default="cuda"); p.add_argument("--diagnostic",action="store_true")
    run(**vars(p.parse_args()))
