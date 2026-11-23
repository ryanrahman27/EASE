"""E23 fixed-budget pilot: privileged gate, then matched history/pose encoders."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .benchmark import fingerprint
from .paired_data import register, calibration
from .mixed_sim import Arm, MixedReach
from .diverse_oracle import OracleData, oracle_tokens
from .full_control import fit, commands
from .history_distill import make_student, frozen_equal


@torch.no_grad()
def evaluate(data, manifest, model, split, mode, device, calibration_seed=930000, intervention=None):
    arms = {a['seed']: Arm(**a) for a in manifest['arms']}
    pairs = {p['id']: p for p in manifest['pairs']}
    cases = [(replace(arms[body], home=tuple(goal['home'])) if 'home' in goal else arms[body], goal, pairs[goal['pair']]) for goal in manifest['goals'][split]
             for body in pairs[goal['pair']]['bodies']]
    records = []
    for offset in range(0, len(cases), 48):
        batch = cases[offset:offset+48]
        envs = [MixedReach(a, synchronize=True) for a, _, _ in batch]
        mask = torch.tensor([[j < a.dof for j in range(7)] for a, _, _ in batch], device=device)
        histories = np.zeros((len(batch), 32, 7, 11), np.float32)
        for i, (arm, _, pair) in enumerate(batch):
            if mode == 'oracle': histories[i] = oracle_tokens(arm, padded=True)
            elif mode != 'reactive':
                donor = replace(arms[next(b for b in pair['bodies'] if b != arm.seed)], home=arm.home) if mode == 'twin' else arm
                # Training evaluation uses a fresh calibration seed, outside the 16 training probes.
                history, _ = calibration(donor, calibration_seed + pair['id'])
                histories[i, :, :arm.dof] = history
        if intervention is not None:
            histories = intervention(histories, mode, mask.cpu().numpy())
        latent = model.encode(torch.tensor(histories, device=device),
                              torch.full((len(batch),), 0 if mode in ('empty', 'reactive') else 32, device=device), mask)
        obs = [env.reset(0, goal['xyz'], spread=0) for env, (_, goal, _) in zip(envs, batch)]
        active = np.ones(len(batch), bool); early = [None]*len(batch)
        for step in range(150):
            q = np.zeros((len(batch), 7, 2), np.float32)
            for i, o in enumerate(obs): q[i, :len(o['joints'])] = o['joints']
            action = commands(model, 'direct', torch.tensor(q, device=device),
                torch.tensor(np.stack([o['task'] for o in obs]), device=device), mask,
                torch.zeros((len(batch), 16, 7), device=device), latent).cpu().numpy()
            for i in np.flatnonzero(active):
                obs[i], done, info = envs[i].step(action[i, :envs[i].arm.dof])
                if step == 9: early[i] = info['error']
                if done:
                    arm, goal, pair = batch[i]
                    records.append({'body_seed': arm.seed, 'goal_id': goal['id'], 'pair': pair['id'],
                        'mode': mode, 'split': split, 'steps': envs[i].steps, 'error_at_10': early[i],
                        'condition': goal.get('condition', 'unspecified'),
                        **{k: info[k] for k in ('success', 'error', 'collision')}})
                    active[i] = False
            if not active.any(): break
    return records


def gate(records):
    bodies = sorted({r['body_seed'] for r in records})
    by_body = {b: [sum(r['success'] for r in records if r['body_seed'] == b),
                   sum(r['body_seed'] == b for r in records)] for b in bodies}
    overall = [sum(r['success'] for r in records), len(records)]
    return {'overall': overall, 'by_body': by_body,
            'passed': overall[0]/overall[1] >= .95 and all(n/d >= .9 for n, d in by_body.values())}


def run(data, output, device='cuda', teacher_steps=10000, student_steps=5000, seed=0):
    torch.set_num_threads(4)
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    manifest = register(data)
    protocol = {'experiment': 'E23' if seed == 0 else 'E24', 'seed': seed, 'scope': 'One recorded seed; aggregate all seeds externally',
        'teacher_steps': teacher_steps, 'student_steps': student_steps, 'batch': 32, 'lr': .0003,
        'manifest_sha256': fingerprint(Path(data)/'manifest.json'), 'test_evaluated': False,
        'gate': 'Teacher >=95% overall and >=90% every training body before student fitting',
        'matching': 'Same frozen teacher, fresh encoder seed, body/context sampling and normalization; only pose transformation differs',
        'history': 'Calibration prefixes 0..32 only, uniform body sampling; no task trajectories',
        'evaluation': 'Fresh paired calibration, frozen context, zero-spread task reset; correct/empty/twin',
        'early_error': 'Distance at step 10; null if episode already ended; do not silently pool nulls'}
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2))
    src = root/'source'; src.mkdir()
    for p in Path(__file__).parent.glob('*.py'): shutil.copyfile(p, src/p.name)
    dataset = OracleData(data)
    teacher = fit(dataset, root/'teacher', 'direct', teacher_steps, device, protocol['manifest_sha256'], seed=seed)
    records = evaluate(data, manifest, teacher, 'train', 'oracle', device)
    result = gate(records)
    (root/'teacher'/'train.json').write_text(json.dumps(records, indent=2))
    (root/'teacher'/'gate.json').write_text(json.dumps(result, indent=2))
    print('TEACHER GATE', result, flush=True)
    if not result['passed']:
        (root/'status.json').write_text(json.dumps({'status': 'teacher_gate_failed', 'students_trained': False}))
        return
    teacher.requires_grad_(False)
    arms = [Arm(**a) for a in manifest['arms'] if a['split'] == 'train']
    contexts = {a.seed: [np.load(Path(data)/c['file']) for c in manifest['calibrations'] if c['body_seed'] == a.seed] for a in arms}
    masks = torch.tensor([[j < a.dof for j in range(7)] for a in arms], device=device)
    with torch.no_grad():
        targets = teacher.encode(torch.tensor(np.stack([oracle_tokens(a, padded=True) for a in arms]), device=device),
                                 torch.full((len(arms),), 32, device=device), masks)
    summary = []
    for name in ('history', 'pose'):
        folder = root/name; folder.mkdir()
        model = make_student(teacher, [c for pool in contexts.values() for c in pool], device, seed, name == 'pose')
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.0003)
        rng = np.random.default_rng(3000+seed)
        losses = []
        for step in range(student_steps):
            ids = rng.integers(len(arms), size=32); lengths = rng.integers(0, 33, size=32)
            histories = np.zeros((32, 32, 7, 11), np.float32)
            for i, (idx, length) in enumerate(zip(ids, lengths)):
                arm = arms[idx]; pool = contexts[arm.seed]
                histories[i, :length, :arm.dof] = pool[rng.integers(len(pool))][:length]
            z = model.encode(torch.tensor(histories, device=device), torch.tensor(lengths, device=device), masks[ids])
            loss = ((z-targets[ids]).square()*masks[ids, :, None]).sum()/(masks[ids].sum()*model.width)
            assert torch.isfinite(loss)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
            losses.append(loss.item())
            if (step+1) % 1000 == 0: print(name, step+1, losses[-1], flush=True)
        model.eval(); assert frozen_equal(model, teacher)
        torch.save(model.state_dict(), folder/'policy.pt')
        (folder/'training.json').write_text(json.dumps({'losses': losses, 'policy_sha256': fingerprint(folder/'policy.pt'),
            'controller_bitwise_unchanged': True, 'context_input': name}, indent=2))
        for split in ('train', 'validation'):
            for mode in ('correct', 'empty', 'twin'):
                rows = evaluate(data, manifest, model, split, mode, device)
                (folder/f'{split}_{mode}.json').write_text(json.dumps(rows, indent=2))
                row = {'model': name, 'split': split, 'mode': mode, **gate(rows)}
                summary.append(row); print(row, flush=True)
                (root/'summary.json').write_text(json.dumps(summary, indent=2))
    (root/'status.json').write_text(json.dumps({'status': 'complete', 'test_evaluated': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('data', 'output'): parser.add_argument('--'+key, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--teacher-steps', type=int, default=10000)
    parser.add_argument('--student-steps', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=0)
    run(**vars(parser.parse_args()))
