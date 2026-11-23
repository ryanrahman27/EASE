"""E26 fixed shared recovery union and directly supervised controller comparison."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .benchmark import fingerprint
from .paired_data import register, calibration
from .mixed_sim import Arm, MixedReach
from .mixed_data import MixedData
from .paired_replication import ReactiveData
from .paired_experiment import evaluate, gate
from .full_control import fit, commands, recovery_chunk, add_recovery
from .joint_policy import JointFlowPolicy


class PrefixData(MixedData):
    def __init__(self, data):
        super().__init__(data)
        selected = np.flatnonzero(self.context_episode == -1)
        self.calibration_pools = {body: [self.contexts[i] for i in selected if self.context_body[i] == body]
                                  for body in self.episode_body}
        self.contexts = [c for pool in self.calibration_pools.values() for c in pool]

    def histories(self, indices, rng):
        histories = np.zeros((len(indices), 32, 7, 11), np.float32)
        lengths = rng.integers(0, 33, size=len(indices))
        for i, (idx, length) in enumerate(zip(indices, lengths)):
            pool = self.calibration_pools[self.episode_body[self.row_episode[idx]]]
            context = pool[rng.integers(len(pool))]
            histories[i, :length, :context.shape[1]] = context[:length]
        return torch.tensor(histories), torch.tensor(lengths)


def load_policy(folder, device):
    folder = Path(folder); meta = json.loads((folder/'training.json').read_text())
    assert fingerprint(folder/'policy.pt') == meta['policy_sha256']
    model = JointFlowPolicy(history=True).to(device)
    model.load_state_dict(torch.load(folder/'policy.pt', map_location=device, weights_only=True)); model.eval()
    return model, meta['policy_sha256']


@torch.no_grad()
def collect(manifest, model, name, folder, device):
    folder.mkdir()
    arms = {a['seed']: Arm(**a) for a in manifest['arms']}
    goals = {g['id']: g for g in manifest['goals']['train']}
    episodes = manifest['episodes']; recovery = []; queries = []
    for offset in range(0, len(episodes), 48):
        batch = episodes[offset:offset+48]
        envs = [MixedReach(arms[e['body_seed']], synchronize=True) for e in batch]
        mask = torch.tensor([[j < env.arm.dof for j in range(7)] for env in envs], device=device)
        contexts = np.zeros((len(batch), 32, 7, 11), np.float32)
        if name == 'history':
            for i, env in enumerate(envs): contexts[i, :, :env.arm.dof] = calibration(env.arm, 104000+env.arm.dof)[0]
        z = model.encode(torch.tensor(contexts, device=device), torch.full((len(batch),), 32 if name == 'history' else 0, device=device), mask)
        obs = [env.reset(0, goals[e['goal_id']]['xyz'], spread=0) for env, e in zip(envs, batch)]
        active = np.ones(len(batch), bool)
        for step in range(150):
            q = np.zeros((len(batch), 7, 2), np.float32)
            for i, o in enumerate(obs): q[i, :len(o['joints'])] = o['joints']
            task = np.stack([o['task'] for o in obs])
            action = commands(model, 'direct', torch.tensor(q, device=device), torch.tensor(task, device=device),
                mask, torch.zeros((len(batch), 16, 7), device=device), z).cpu().numpy()
            for i in np.flatnonzero(active):
                if step in (0, 1, 2, 4, 8, 16, 32, 64, 100) and not envs[i].collision:
                    target, valid, collision = recovery_chunk(envs[i], obs[i])
                    queries.append({'episode': int(offset+i), 'body': envs[i].arm.seed, 'step': step, 'branch_collision': bool(collision)})
                    if not collision:
                        recovery.append((q[i].copy(), task[i].copy(), mask[i].cpu().numpy(), target, valid, offset+i))
                obs[i], done, info = envs[i].step(action[i, :envs[i].arm.dof])
                if done: active[i] = False
            if not active.any(): break
        print('recovery', name, offset+len(batch), '/', len(episodes), 'accepted', len(recovery), flush=True)
    path = folder/'recovery.npz'
    np.savez_compressed(path, **{key: np.stack([r[i] for r in recovery]) for i, key in enumerate(('joints', 'task', 'mask', 'action', 'valid', 'episode'))})
    (folder/'queries.json').write_text(json.dumps({'queries': queries, 'accepted': len(recovery), 'sha256': fingerprint(path)}, indent=2))
    return path


def run(data, replication, output, device='cuda', steps=15000):
    torch.set_num_threads(4)
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    manifest = register(data)
    entries = json.loads((Path(replication)/'index.json').read_text())
    source = entries[0]
    protocol = {'experiment': 'E26', 'seed': 0, 'steps_per_model': steps, 'batch': 32, 'lr': .0003,
        'manifest_sha256': fingerprint(Path(data)/'manifest.json'), 'test_evaluated': False,
        'models': ['reactive', 'history'], 'data_conditions': ['original', 'shared_recovery'],
        'training': 'Four fresh directly supervised action policies; same initialization, uniform-row batch stream, capacity and budget per condition. No latent distillation.',
        'history': 'Training calibration prefixes only; current observations always available. Reactive context length zero.',
        'recovery': 'Fixed union from E24 seed0 reactive and E23 seed0 history policies on new training tasks only; expert 16-action branches at steps 0,1,2,4,8,16,32,64,100, reject contact branches.',
        'evaluation': 'All training goals plus four factorial validation conditions; failed gates retained as diagnostic; history correct/empty/twin with one fixed probe seed.',
        'reference': 'Frozen E24 privileged teachers evaluated on factorial validation; trained on previous dataset, not new teachers or upper bounds.',
        'scope': 'Single-seed bounded experiment; one recovery round; no validation tuning'}
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2))
    snapshots = root/'source'; snapshots.mkdir()
    for p in Path(__file__).parent.glob('*.py'): shutil.copyfile(p, snapshots/p.name)
    paths = []; source_hashes = {}
    for name, folder in [('reactive', source['reactive']), ('history', str(Path(source['run'])/'history'))]:
        model, digest = load_policy(folder, device); source_hashes[str(Path(folder)/'policy.pt')] = digest
        paths.append(collect(manifest, model, name, root/f'collector_{name}', device))
        assert fingerprint(Path(folder)/'policy.pt') == digest
    shared = {'source_checkpoints': source_hashes, 'files': {str(p): fingerprint(p) for p in paths}}
    (root/'shared_recovery.json').write_text(json.dumps(shared, indent=2))
    for entry in entries:
        model, digest = load_policy(Path(entry['run'])/'teacher', device)
        rows = evaluate(data, manifest, model, 'validation', 'oracle', device)
        (root/f'privileged_{entry["seed"]}.json').write_text(json.dumps({'checkpoint_sha256': digest, 'records': rows}, indent=2))
    summary = []
    for phase in ('original', 'shared_recovery'):
        for name, cls in [('reactive', ReactiveData), ('history', PrefixData)]:
            dataset = cls(data)
            if phase == 'shared_recovery': dataset = add_recovery(dataset, paths)
            folder = root/f'{phase}_{name}'
            model = fit(dataset, folder, 'direct', steps, device, protocol['manifest_sha256'],
                        fingerprint(root/'shared_recovery.json') if phase == 'shared_recovery' else None)
            for split in ('train', 'validation'):
                for mode in (['reactive'] if name == 'reactive' else ['correct', 'empty', 'twin']):
                    rows = evaluate(data, manifest, model, split, mode, device)
                    (folder/f'{split}_{mode}.json').write_text(json.dumps(rows, indent=2))
                    row = {'phase': phase, 'model': name, 'split': split, 'mode': mode, **gate(rows)}
                    if split == 'validation':
                        row['conditions'] = {c: gate([r for r in rows if r['condition'] == c]) for c in sorted({r['condition'] for r in rows})}
                    summary.append(row); print(row, flush=True)
                    (root/'summary.json').write_text(json.dumps(summary, indent=2))
    assert all(fingerprint(p) == h for p, h in shared['files'].items())
    (root/'status.json').write_text(json.dumps({'status': 'complete', 'test_evaluated': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('data', 'replication', 'output'): parser.add_argument('--'+key, required=True)
    parser.add_argument('--device', default='cuda'); parser.add_argument('--steps', type=int, default=15000)
    run(**vars(parser.parse_args()))
