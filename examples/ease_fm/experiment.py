"""Matched training and paired history interventions on the v2 benchmark."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

from .baseline import FlowPolicy, HORIZON, training_arrays
from .benchmark import HISTORY_LENGTH, calibration, fingerprint, load_manifest, transition_token
from .history import HistoryBuffer, HistoryFlowPolicy, TOKEN_DIM
from .sim import Body, ReachEnv


class TrainingData:
    def __init__(self, directory):
        directory = Path(directory)
        self.manifest = load_manifest(directory, verify_assets=True)
        self.observation, self.action, self.mask = training_arrays(directory)
        self.row_episode, self.episode_body = [], []
        self.contexts, self.context_body, self.context_episode = [], [], []
        for episode in self.manifest["episodes"]:
            if episode["split"] != "train" or not episode["success"] or episode["collision"]:
                continue
            episode_id = len(self.episode_body)
            self.episode_body.append(episode["body_seed"])
            with np.load(directory / episode["file"]) as data:
                tokens = transition_token(data["observation"], data["action"], data["next_observation"])
            self.row_episode.extend([episode_id] * len(tokens))
            self.contexts.append(tokens)
            self.context_body.append(episode["body_seed"])
            self.context_episode.append(episode_id)
        for entry in self.manifest["calibrations"]:
            if entry["split"] != "train":
                raise ValueError("Held-out calibration in training manifest")
            self.contexts.append(np.load(directory / entry["file"]))
            self.context_body.append(entry["body_seed"])
            self.context_episode.append(-1)
        self.row_episode = np.array(self.row_episode)
        self.context_body = np.array(self.context_body)
        self.context_episode = np.array(self.context_episode)
        self.candidates = {}
        for episode_id, body_seed in enumerate(self.episode_body):
            # Half calibration and half separate task episodes when both exist.
            valid = (self.context_body == body_seed) & (self.context_episode != episode_id)
            self.candidates[episode_id] = [np.flatnonzero(valid & (self.context_episode == -1)),
                                           np.flatnonzero(valid & (self.context_episode != -1))]
            if any(len(pool) == 0 for pool in self.candidates[episode_id]):
                raise ValueError("Each training body needs calibration and at least two task episodes")

    def sample_history(self, indices, rng):
        tokens = np.zeros((len(indices), HISTORY_LENGTH, TOKEN_DIM), dtype=np.float32)
        lengths = np.zeros(len(indices), dtype=np.int64)
        chosen = []
        for row, idx in enumerate(indices):
            episode_id = self.row_episode[idx]
            pool = self.candidates[episode_id][rng.integers(2)]
            context_id = int(rng.choice(pool))
            context = self.contexts[context_id]
            end = int(rng.integers(1, len(context)+1))
            count = int(rng.integers(0, min(end, HISTORY_LENGTH)+1))
            if count:
                tokens[row, :count] = context[end-count:end]
            lengths[row] = count
            chosen.append(context_id)
        return torch.from_numpy(tokens), torch.from_numpy(lengths), chosen


def train(data, output, method="history", steps=1500, batch_size=64, seed=0, device="cpu"):
    if method not in {"baseline", "history"} or min(steps, batch_size) < 1:
        raise ValueError("Invalid training configuration")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    dataset = TrainingData(data)
    obs, action, mask = [x.to(device) for x in (dataset.observation, dataset.action, dataset.mask)]
    model = (HistoryFlowPolicy() if method == "history" else FlowPolicy()).to(device)
    model.obs_mean.copy_(obs.mean(0))
    model.obs_std.copy_(obs.std(0, unbiased=False).clamp_min(.01))
    if method == "history":
        all_context = torch.from_numpy(np.concatenate(dataset.contexts)).to(device)
        model.history_mean.copy_(all_context.mean(0))
        model.history_std.copy_(all_context.std(0, unbiased=False).clamp_min(.01))
    # Identical shared-backbone initialization and post-init training RNG across
    # methods; context sampling has a separate NumPy stream.
    torch.manual_seed(seed+1000)
    rng = np.random.default_rng(seed+2000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    start = time.perf_counter()
    for step in range(steps):
        idx = torch.randint(len(obs), (batch_size,), device=device)
        target, valid = action[idx], mask[idx]
        noise = torch.randn_like(target)
        tau = torch.rand(batch_size, device=device)
        latent = None
        if method == "history":
            tokens, lengths, _ = dataset.sample_history(idx.cpu().numpy(), rng)
            latent = model.encode(tokens.to(device), lengths.to(device))
        velocity = model(noise.lerp(target, tau[:, None, None]), tau, obs[idx], latent)
        loss = ((velocity - (target-noise)).square() * valid).sum() / valid.sum()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 250 == 0:
            print(method, "seed", seed, "step", step, "loss", losses[-1], flush=True)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    torch.save(model.cpu().state_dict(), output / "policy.pt")
    source = output / "source"
    source.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(path, source / path.name)
    metadata = {"method": method, "seed": seed, "steps": steps, "batch_size": batch_size,
        "data": str(Path(data).resolve()), "manifest_sha256": fingerprint(Path(data) / "manifest.json"),
        "policy_sha256": fingerprint(output / "policy.pt"), "losses": losses,
        "seconds": elapsed, "parameters": sum(p.numel() for p in model.parameters()),
        "training_transitions": len(obs), "device": device, "torch": torch.__version__,
        "python": platform.python_version(), "horizon": HORIZON, "integration_steps": 10,
        "history_length": HISTORY_LENGTH, "auxiliary_dynamics_weight": 0,
        "source_hashes": {p.name: fingerprint(p) for p in source.glob("*.py")}}
    (output / "training.json").write_text(json.dumps(metadata, indent=2))
    print("saved", output, "seconds", round(elapsed, 1), flush=True)


def evaluate(data, checkpoint, output, split="validation", modes=None, device="cpu", trials=None):
    """Frozen-weight, paired evaluation. Test is explicit and never the default."""
    if split not in {"validation", "test"}:
        raise ValueError("Choose validation or test")
    torch.set_num_threads(4)
    checkpoint = Path(checkpoint)
    metadata = json.loads((checkpoint.parent / "training.json").read_text())
    manifest = load_manifest(data, verify_assets=True)
    if metadata["manifest_sha256"] != fingerprint(Path(data) / "manifest.json"):
        raise ValueError("Checkpoint belongs to a different benchmark")
    if metadata["policy_sha256"] != fingerprint(checkpoint):
        raise ValueError("Checkpoint hash mismatch")
    history = metadata["method"] == "history"
    modes = modes or (["empty", "correct", "wrong", "shuffled", "online"] if history else ["none"])
    allowed = {"empty", "correct", "wrong", "shuffled", "online"} if history else {"none"}
    if not set(modes) <= allowed:
        raise ValueError("Invalid intervention for this model")
    model = (HistoryFlowPolicy() if history else FlowPolicy()).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected = {"validation"} if split == "validation" else {"interpolation", "extrapolation"}
    bodies = [Body(**b) for b in manifest["bodies"] if b["split"] in selected]
    donors = [Body(**b) for b in manifest["bodies"] if b["split"] == "train"]
    goals = manifest["goals"][split]
    if trials is not None:
        if trials < 1:
            raise ValueError("Trials must be positive")
        goals = goals[:trials]
    cases = [(body, goal) for body in bodies for goal in goals]
    correct, wrong, donor_ids = [], [], []
    for index, (body, goal) in enumerate(cases):
        calibration_seed = goal["seed"] + 500000
        correct.append(calibration(ReachEnv(body), calibration_seed))
        donor = donors[(index // len(goals)) % len(donors)]
        if donor.seed == body.seed:
            raise ValueError("Wrong-body donor equals target")
        wrong.append(calibration(ReachEnv(donor), calibration_seed))
        donor_ids.append(donor.seed)
    # Fixed per-case noise means early success in another rollout cannot change
    # a trial's random stream. Same samples across methods and interventions.
    noise = []
    for _, goal in cases:
        generator = torch.Generator().manual_seed(goal["seed"] + 600000)
        noise.append(torch.randn(150, HORIZON, 7, generator=generator))
    noise = torch.stack(noise, dim=1).to(device)
    records = []
    start = time.perf_counter()
    for mode in modes:
        envs = [ReachEnv(body) for body, _ in cases]
        obs = np.stack([env.reset(goal["seed"], goal["xyz"]) for env, (_, goal) in zip(envs, cases)])
        buffers = []
        for idx, (_, goal) in enumerate(cases):
            tokens = wrong[idx] if mode == "wrong" else correct[idx]
            if mode == "shuffled":
                tokens = np.random.default_rng(goal["seed"]+700000).permutation(tokens)
            buffers.append(HistoryBuffer(() if mode in {"empty", "none"} else tokens))
        active = np.ones(len(cases), dtype=bool)
        latent = None
        for step in range(150):
            with torch.no_grad():
                if history and (step == 0 or mode == "online"):
                    arrays, lengths = zip(*(buffer.array() for buffer in buffers))
                    latent = model.encode(torch.from_numpy(np.stack(arrays)).to(device),
                                          torch.tensor(lengths, device=device))
                actions = model.sample(torch.from_numpy(obs).to(device), latent=latent,
                                       noise=noise[step]).cpu().numpy()[:, 0]
            for idx in np.flatnonzero(active):
                new_obs, done, info = envs[idx].step(actions[idx])
                if mode == "online":
                    buffers[idx].append(obs[idx], info["executed_action"], new_obs)
                obs[idx] = new_obs
                if done:
                    body, goal = cases[idx]
                    records.append({"body_seed": body.seed, "body_split": body.split,
                        "goal_id": goal["id"], "task_seed": goal["seed"], "mode": mode,
                        "success": info["success"], "error": info["error"], "steps": envs[idx].steps,
                        "collision": envs[idx].episode_collision, "calibration_actions": HISTORY_LENGTH,
                        "donor_body_seed": donor_ids[idx] if mode == "wrong" else None})
                    active[idx] = False
            if not active.any():
                break
        rows = [r for r in records if r["mode"] == mode]
        print(metadata["method"], metadata["seed"], mode, "success",
              sum(r["success"] for r in rows), "/", len(rows), flush=True)
    result = {"method": metadata["method"], "training_seed": metadata["seed"], "split": split,
        "manifest_sha256": metadata["manifest_sha256"], "policy_sha256": metadata["policy_sha256"],
        "checkpoint": str(checkpoint.resolve()), "seconds": time.perf_counter()-start,
        "device": device, "records": records,
        "context_protocol": "32 separate-episode calibration actions, task reset, fixed context except online",
        "paired_noise": True, "weights_frozen": True}
    with Path(output).open("x") as file:
        json.dump(result, file, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--data", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--method", choices=["baseline", "history"], default="history")
    training.add_argument("--steps", type=int, default=1500)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--device", default="cpu")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--split", choices=["validation", "test"], default="validation")
    evaluation.add_argument("--modes", nargs="+")
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--trials", type=int)
    arguments = vars(parser.parse_args())
    command = arguments.pop("command")
    (train if command == "train" else evaluate)(**arguments)
