"""Report mixed-arm development pilots without implying statistical confirmation."""
import argparse
import json
from pathlib import Path

import numpy as np


def report(directory):
    root = Path(directory)
    baseline = json.loads((root / "baseline/validation.json").read_text())
    history = json.loads((root / "history/validation.json").read_text())
    if baseline["manifest_sha256"] != history["manifest_sha256"]:
        raise ValueError("Unmatched datasets")
    records = baseline["records"] + history["records"]
    rows = []
    lines = ["Mixed-arm development pilot (one training seed, one body per validation family)",
             "Family | context | successes/trials | final error (m)"]
    for family in ["six", "seven", "six_novel"]:
        reference = None
        for mode in ["none", "empty", "correct8", "correct32", "wrong32"]:
            trials = [record for record in records if record["family"] == family and record["mode"] == mode]
            keys = {(r["body_seed"], r["goal_id"], r["task_seed"]) for r in trials}
            if not trials or len(keys) != len(trials):
                raise ValueError("Missing or duplicate cases")
            if reference is None:
                reference = keys
            if keys != reference:
                raise ValueError("Unpaired cases")
            row = {"family": family, "mode": mode, "successes": sum(r["success"] for r in trials),
                   "trials": len(trials), "mean_final_error_m": float(np.mean([r["error"] for r in trials]))}
            rows.append(row)
            lines.append(f"{family} | {mode} | {row['successes']}/{row['trials']} | {row['mean_final_error_m']:.4f}")
    lines += ["", "One shared policy per method is trained on both six- and seven-DoF arms.",
              "six_novel has a joint-axis sequence excluded from training.",
              "Histories are separate-episode calibration; weights/context stay frozen during each task.",
              "Single-seed pilot: no conclusion about reliable or rapid adaptation. Final test body unused."]
    (root / "summary.txt").write_text("\n".join(lines))
    (root / "summary.json").write_text(json.dumps({"conditions": rows,
        "training_seed": baseline["training_seed"], "manifest_sha256": baseline["manifest_sha256"],
        "test_evaluated": False}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    report(parser.parse_args().directory)
