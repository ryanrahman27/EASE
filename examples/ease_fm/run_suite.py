"""Run a prespecified validation comparison without touching the final test split."""
import argparse
import json
from pathlib import Path

import torch

from .benchmark import fingerprint, load_manifest
from .experiment import evaluate, train


def run(data, output, steps=1500, device="cpu"):
    load_manifest(data, verify_assets=True)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    protocol = {"training_seeds": [0, 1, 2], "steps": steps, "batch_size": 64,
        "methods": ["baseline", "history"], "split": "validation", "device": device,
        "manifest_sha256": fingerprint(Path(data) / "manifest.json"),
        "selection": "Fixed final step, no checkpoint selection or early stopping",
        "primary_contrasts": ["history/correct - baseline/none", "history/correct - history/wrong"],
        "secondary_contrasts": ["history/correct - history/empty", "history/online - history/correct",
                                 "history/correct - history/shuffled"],
        "parameter_matching": "Shared FM backbone; history encoder adds parameters, counts reported",
        "test_evaluation": False, "torch": torch.__version__}
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2))
    for seed in protocol["training_seeds"]:
        for method in protocol["methods"]:
            directory = root / f"{method}_seed{seed}"
            train(data, directory, method=method, steps=steps, seed=seed, device=device)
            evaluate(data, directory / "policy.pt", directory / "validation.json", device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--device", default="cpu")
    run(**vars(parser.parse_args()))
