"""History-free conditional flow matching; development baseline only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .sim import ACTION_LIMIT, OBS_DIM, Body, ReachEnv

HORIZON = 16


class FlowPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(7, 128)
        self.condition = nn.Sequential(nn.Linear(OBS_DIM + 1, 128), nn.SiLU(), nn.Linear(128, 128))
        self.position = nn.Parameter(torch.randn(1, HORIZON, 128) * .02)
        layer = nn.TransformerEncoderLayer(128, 4, 256, dropout=0., batch_first=True)
        self.backbone = nn.TransformerEncoder(layer, 2)
        self.output = nn.Linear(128, 7)
        self.register_buffer("obs_mean", torch.zeros(OBS_DIM))
        self.register_buffer("obs_std", torch.ones(OBS_DIM))

    def forward(self, action, time, observation, latent=None):
        observation = (observation - self.obs_mean) / self.obs_std
        context = self.condition(torch.cat([observation, time[:, None]], dim=-1))
        if latent is not None:
            context = context + latent
        return self.output(self.backbone(self.input(action) + self.position + context[:, None]))

    @torch.no_grad()
    def sample(self, observation, steps=10, latent=None, noise=None):
        if steps < 1:
            raise ValueError("Integration steps must be positive")
        action = (torch.randn(len(observation), HORIZON, 7, device=observation.device)
                  if noise is None else noise.clone())
        for i in range(steps):
            time = torch.full((len(observation),), i / steps, device=observation.device)
            action += self(action, time, observation, latent) / steps
        return action.clamp(-1, 1) * ACTION_LIMIT


def training_arrays(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    observations, actions, masks = [], [], []
    for episode in manifest["episodes"]:
        if episode["split"] != "train" or not episode["success"] or episode["collision"]:
            continue
        with np.load(directory / episode["file"]) as data:
            for t, obs in enumerate(data["observation"]):
                chunk = data["action"][t:t+HORIZON]
                padded = np.zeros((HORIZON, 7), dtype=np.float32)
                mask = np.zeros((HORIZON, 7), dtype=np.float32)
                padded[:len(chunk)] = chunk / ACTION_LIMIT
                mask[:len(chunk)] = 1
                observations.append(obs)
                actions.append(padded)
                masks.append(mask)
    if not observations:
        raise ValueError("No successful collision-free training episodes")
    return tuple(torch.from_numpy(np.stack(x)) for x in (observations, actions, masks))


def train(data, output, steps=1000, batch_size=64, seed=0, device="cpu"):
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    obs, action, mask = [x.to(device) for x in training_arrays(data)]
    model = FlowPolicy().to(device)
    model.obs_mean.copy_(obs.mean(0))
    model.obs_std.copy_(obs.std(0, unbiased=False).clamp_min(.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = []
    for step in range(steps):
        idx = torch.randint(len(obs), (batch_size,), device=device)
        target, valid = action[idx], mask[idx]
        noise = torch.randn_like(target)
        time = torch.rand(batch_size, device=device)
        interpolated = noise.lerp(target, time[:, None, None])
        loss = ((model(interpolated, time, obs[idx]) - (target-noise)).square() * valid).sum() / valid.sum()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 100 == 0:
            print("step", step, "loss", losses[-1], flush=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.save(model.cpu().state_dict(), output / "policy.pt")
    (output / "training.json").write_text(json.dumps({"seed": seed, "steps": steps, "batch_size": batch_size,
        "data": str(Path(data).resolve()), "losses": losses, "torch": torch.__version__, "device": device,
        "training_transitions": len(obs), "horizon": HORIZON, "integration_steps": 10}, indent=2))
    (output / "data_manifest.json").write_text((Path(data) / "manifest.json").read_text())


def evaluate(data, checkpoint, output, trials=10, seed=500000):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    model = FlowPolicy()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    manifest = json.loads((Path(data) / "manifest.json").read_text())
    records = []
    for parameters in manifest["bodies"]:
        body = Body(**parameters)
        env = ReachEnv(body)
        for trial in range(trials):
            obs = env.reset(seed+trial)
            done, collision = False, False
            while not done:
                # Execute one action, then replan. Student gets only observation.
                action = model.sample(torch.from_numpy(obs)[None])[0, 0].numpy()
                obs, done, info = env.step(action)
                collision |= info["collision"]
            records.append({"body_seed": body.seed, "split": body.split, "trial_seed": seed+trial,
                            "success": info["success"], "error": info["error"], "steps": env.steps,
                            "collision": collision})
    with Path(output).open("x") as file:
        json.dump({"seed": seed, "checkpoint": str(checkpoint), "records": records}, file, indent=2)
    for split in ("train", "interpolation", "extrapolation"):
        rows = [r for r in records if r["split"] == split]
        if rows:
            print(split, "success", np.mean([r["success"] for r in rows]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--data", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--steps", type=int, default=1000)
    training.add_argument("--batch-size", type=int, default=64)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--device", default="cpu")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--data", required=True)
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--trials", type=int, default=10)
    evaluation.add_argument("--seed", type=int, default=500000)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if any(args.get(k, 1) < 1 for k in ("steps", "batch_size", "trials")):
        parser.error("Counts must be positive")
    (train if command == "train" else evaluate)(**args)
