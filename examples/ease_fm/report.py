"""Summarize paired validation interventions with crossed-factor bootstrap intervals."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def paired_interval(left, right, samples=5000, seed=123):
    """Resample training seeds, bodies and goal IDs, preserving each paired contrast.

    These are exploratory percentile intervals, especially with only three bodies.
    Trials are not treated as independent samples of embodiment generalization.
    """
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("Expected matching seed x body x goal tensors")
    rng = np.random.default_rng(seed)
    delta = left.astype(float) - right.astype(float)
    bootstrap = np.empty(samples)
    for idx in range(samples):
        axes = [rng.integers(size, size=size) for size in delta.shape]
        bootstrap[idx] = delta[np.ix_(*axes)].mean()
    low, high = np.quantile(bootstrap, [.025, .975])
    return {"difference": float(delta.mean()), "low": float(low), "high": float(high)}


def summarize(directory):
    root = Path(directory)
    protocol = json.loads((root / "protocol.json").read_text())
    rows, training = [], []
    for seed in protocol["training_seeds"]:
        for method in protocol["methods"]:
            run = root / f"{method}_seed{seed}"
            evaluation = json.loads((run / "validation.json").read_text())
            metadata = json.loads((run / "training.json").read_text())
            if evaluation["split"] != "validation" or evaluation["manifest_sha256"] != protocol["manifest_sha256"]:
                raise ValueError("Evaluation does not match suite protocol")
            if metadata["steps"] != protocol["steps"] or metadata["seed"] != seed:
                raise ValueError("Training does not match suite protocol")
            training.append({key: metadata[key] for key in ["method", "seed", "parameters", "seconds", "training_transitions"]})
            for record in evaluation["records"]:
                rows.append({"training_seed": seed, "method": method, **record})
    with (root / "trials.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    conditions = sorted({(row["method"], row["mode"]) for row in rows})
    tensors, summary = {}, []
    reference_keys = None
    for method, mode in conditions:
        selected = [r for r in rows if r["method"] == method and r["mode"] == mode]
        keys = [(r["training_seed"], r["body_seed"], r["goal_id"]) for r in selected]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate paired trial")
        if reference_keys is None:
            reference_keys = set(keys)
        if set(keys) != reference_keys:
            raise ValueError("Unmatched conditions")
        seeds = sorted({k[0] for k in keys})
        bodies = sorted({k[1] for k in keys})
        goals = sorted({k[2] for k in keys})
        lookup = {k: r for k, r in zip(keys, selected)}
        success = np.array([[[lookup[(s, b, g)]["success"] for g in goals] for b in bodies] for s in seeds])
        tensors[f"{method}/{mode}"] = success
        summary.append({"condition": f"{method}/{mode}", "successes": int(success.sum()),
            "trials": int(success.size), "success_rate": float(success.mean()),
            "seed_success_rates": success.mean((1, 2)).tolist(),
            "body_success_rates": dict(zip(map(str, bodies), success.mean((0, 2)).tolist())),
            "mean_final_error_m": float(np.mean([r["error"] for r in selected])),
            "collision_rate": float(np.mean([r["collision"] for r in selected]))})
    contrasts = {}
    for contrast in protocol["primary_contrasts"] + protocol["secondary_contrasts"]:
        left, right = contrast.split(" - ")
        contrasts[contrast] = paired_interval(tensors[left], tensors[right])
    result = {"conditions": summary, "paired_success_contrasts": contrasts, "training": training,
        "interval_method": "5000 crossed bootstrap draws over training seeds, bodies and goal IDs; paired percentile 95%",
        "limitations": ["Only three validation bodies and three training seeds",
                        "Development experiment; not a confirmatory test",
                        "Fixed calibration and online context are separate interventions",
                        "Shared FM backbone, but history model has extra encoder parameters",
                        "Generic capsule arm; expert-screened common workspace; state sensing assumed"]}
    (root / "summary.json").write_text(json.dumps(result, indent=2))
    lines = ["EASE-FM v2 validation comparison", "", "Condition | successes/trials | success | mean final error (m)"]
    for row in summary:
        lines.append(f"{row['condition']} | {row['successes']}/{row['trials']} | "
                     f"{row['success_rate']:.1%} | {row['mean_final_error_m']:.4f}")
    lines += ["", "Paired success differences (percentage points; exploratory 95% intervals)"]
    for name, value in contrasts.items():
        lines.append(f"{name}: {100*value['difference']:+.1f} [{100*value['low']:+.1f}, {100*value['high']:+.1f}]")
    lines += ["", *result["limitations"]]
    (root / "summary.txt").write_text("\n".join(lines))
    print("\n".join(lines))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    summarize(parser.parse_args().directory)
