"""E30: supervised joint-token Transformer for measured local motion models."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import mujoco
from .joint_policy import transformer, positions
from .benchmark import fingerprint
from .paired_data import register, calibration
from .mixed_sim import Arm, MixedReach


class MotionTransformer(nn.Module):
    """Predict J columns from measured normal-equation statistics; no inverse in forward."""
    def __init__(self, width=96):
        super().__init__()
        self.width = width
        self.input = nn.Linear(10, width)
        self.encoder = transformer(width, 3)
        self.output = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))

    def forward(self, gram, cross, mask):
        pair = mask[:, :, None] & mask[:, None, :]
        gram = gram.masked_fill(~pair, 0)
        scale = (gram.diagonal(dim1=-2, dim2=-1).sum(-1)/mask.sum(-1)).clamp_min(1e-8)
        features = torch.cat([gram/scale[:, None, None], cross/scale[:, None, None]], -1)
        features = features.masked_fill(~mask[:, :, None], 0)
        latent = self.input(features) + positions(7, self.width, gram.device)[None]
        latent = self.encoder(latent, src_key_padding_mask=~mask)
        return self.output(latent).masked_fill(~mask[:, :, None], 0)


def statistics(history):
    dq = history[:, :, 3].astype(np.float64)
    dx = history[:, 0, 8:11].astype(np.float64)
    return dq.T @ dq+1e-6*np.eye(dq.shape[1]), dq.T @ dx, (history[:, :, 2]*.04-dq).mean(0)


def fit(data, manifest, root, steps, device, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(12000+seed)
    profiles = []
    for entry in manifest['calibrations']:
        assert entry['split'] == 'train'
        h = np.load(Path(data)/entry['file'])
        dq = h[:, :, 3].astype(np.float32)
        padded = np.zeros((32, 7), np.float32); padded[:, :dq.shape[1]] = dq
        profiles.append(padded)
    profile = torch.tensor(np.stack(profiles), device=device)
    model = MotionTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    losses = []
    for step in range(steps):
        batch = 128
        # Random linear systems teach the identification operation, not held-out robot geometry.
        dofs = torch.randint(6, 8, (batch,), device=device)
        mask = torch.arange(7, device=device)[None] < dofs[:, None]
        dq = profile[torch.randint(len(profile), (batch,), device=device)].clone()
        mix = torch.randn((batch, 7, 7), device=device)/7**.5
        dq = dq @ mix
        if step % 2:
            dq = torch.randn((batch, 32, 7), device=device)*.008
            dq = dq @ mix
        dq *= torch.exp(torch.randn((batch, 1, 7), device=device)*.4)
        dq = dq.masked_fill(~mask[:, None], 0)
        raw = dq.transpose(1, 2) @ dq
        gram = raw + 1e-6*torch.eye(7, device=device)[None]
        teacher_jac = torch.randn((batch, 7, 3), device=device)*float(rng.uniform(.15, .8))
        cross = raw @ teacher_jac
        target = torch.linalg.solve(gram, cross).masked_fill(~mask[:, :, None], 0)
        prediction = model(gram, cross, mask)
        loss = ((prediction-target).square()*mask[:, :, None]).sum()/(mask.sum()*3)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite motion loss')
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        losses.append(loss.item())
        if (step+1) % 1000 == 0: print('motion', step+1, losses[-1], flush=True)
    model.eval(); torch.save(model.state_dict(), root/'policy.pt')
    (root/'training.json').write_text(json.dumps(dict(seed=seed, steps=steps, losses=losses,
        parameters=sum(p.numel() for p in model.parameters()), policy_sha256=fingerprint(root/'policy.pt'),
        profile_count=len(profiles), manifest_sha256=fingerprint(Path(data)/'manifest.json')), indent=2))
    return model


@torch.no_grad()
def evaluate(manifest, model, split, mode, device, probe=930000):
    arms = {a['seed']: Arm(**a) for a in manifest['arms']}
    pairs = {p['id']: p['bodies'] for p in manifest['pairs']}
    cases = [(replace(arms[b], home=tuple(g['home'])) if 'home' in g else arms[b], g)
             for g in manifest['goals'][split] for b in pairs[g['pair']]]
    records = []; cache = {}
    for offset in range(0, len(cases), 48):
        batch = cases[offset:offset+48]; size = len(batch)
        gram = np.zeros((size, 7, 7)); cross = np.zeros((size, 7, 3)); bias = np.zeros((size, 7))
        masks = np.zeros((size, 7), bool)
        envs = []; observations = []
        for i, (arm, goal) in enumerate(batch):
            n = arm.dof
            if mode == 'empty_neural_online':
                g, c, b = 1e-6*np.eye(n), np.zeros((n,3)), np.zeros(n)
            else:
                donor = replace(arms[next(b for b in pairs[goal['pair']] if b != arm.seed)],home=arm.home) if mode == 'twin_neural_online' else arm
                key = (donor.seed, tuple(donor.home))
                if key not in cache: cache[key] = statistics(calibration(donor, probe+goal['pair'])[0])
                g, c, b = cache[key]
            gram[i, :n, :n] = g; cross[i, :n] = c; bias[i, :n] = b; masks[i, :n] = True
            env = MixedReach(arm, synchronize=True); envs.append(env)
            observations.append(env.reset(0, goal['xyz'], spread=0))
        mask = torch.tensor(masks, device=device); active = np.ones(size, bool); predicted = None
        for step in range(150):
            if predicted is None or mode in ('neural_online','empty_neural_online','twin_neural_online'):
                predicted = model(torch.tensor(gram, device=device, dtype=torch.float32),
                    torch.tensor(cross, device=device, dtype=torch.float32), mask).cpu().numpy()
            for i in np.flatnonzero(active):
                arm, goal = batch[i]; n = arm.dof
                jac = predicted[i, :n].T
                if step and mode == 'hybrid_online': jac = np.linalg.solve(gram[i, :n, :n], cross[i, :n]).T
                before = observations[i]
                error = before['task'][3:]-before['task'][:3]
                action = np.clip(jac.T @ np.linalg.solve(jac @ jac.T+.015**2*np.eye(3), .5*error)+bias[i, :n], -.04, .04)
                after, done, info = envs[i].step(action)
                if mode != 'neural_frozen':
                    dq = (after['joints'][:, 0]-before['joints'][:, 0]).astype(float)
                    dx = (after['task'][:3]-before['task'][:3]).astype(float)
                    gram[i, :n, :n] = .98*gram[i, :n, :n]+np.outer(dq, dq)+2e-8*np.eye(n)
                    cross[i, :n] = .98*cross[i, :n]+np.outer(dq, dx)
                    bias[i, :n] = .8*bias[i, :n]+.2*(info['executed_action']-dq)
                observations[i] = after
                if done:
                    records.append(dict(split=split, mode=mode, probe_seed=probe, body_seed=arm.seed,
                        goal_id=goal['id'], dof=n, condition=goal.get('condition','train'), steps=envs[i].steps,
                        **{k: info[k] for k in ('success','error','collision')}))
                    active[i] = False
            if not active.any(): break
    return records


def run(data, output, steps=15000, device='cuda', seed=0):
    torch.set_num_threads(4)
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    manifest = register(data)
    protocol = dict(experiment='E30', seed=seed, steps=steps, batch=128,
        manifest_sha256=fingerprint(Path(data)/'manifest.json'), test_evaluated=False,
        architecture='Joint-token Transformer predicts 3D Jacobian columns from measured Gram/cross statistics. Shared damped Cartesian controller and measured tracking bias. No analytical inverse in neural Jacobian forward.',
        training='Random linear motion systems using only training-probe excitation profiles plus synthetic Gaussian excitation. Ridge-solution supervision; no held-out geometry, expert actions, or task labels.',
        modes=['neural_frozen','neural_online','hybrid_online'],
        hybrid_warning='Hybrid uses Transformer on the first task step only, then analytical identification; success is not evidence of neural-only transfer.',
        calibration_steps=32, probe_seeds=[930000], model_selection='Fixed final checkpoint, no validation tuning')
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2))
    source=root/'source'; source.mkdir()
    for p in Path(__file__).parent.glob('*.py'): shutil.copyfile(p,source/p.name)
    model = fit(data, manifest, root, steps, device, seed)
    with patch.object(mujoco, 'mj_jacSite', side_effect=RuntimeError('Privileged Jacobian forbidden')):
        for split in ('train','validation'):
            for mode in protocol['modes']:
                records=evaluate(manifest,model,split,mode,device)
                (root/f'{split}_{mode}.json').write_text(json.dumps(records,indent=2))
                print(split,mode,sum(r['success'] for r in records),'/',len(records),flush=True)
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    for key in ('data','output'): parser.add_argument('--'+key,required=True)
    parser.add_argument('--steps',type=int,default=15000); parser.add_argument('--device',default='cuda')
    parser.add_argument('--seed',type=int,default=0)
    run(**vars(parser.parse_args()))
