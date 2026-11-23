"""E126: source-placement-grouped validation of an abstaining task-value gate."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from . import broad_task_value_gate as broad
from .broad_task_value_gate import Gate, PHASES, features, fingerprint, previous, ROT_SCALE, rotation_error

ROOT = Path('outputs/ease_abstaining_task_value_gate_v1')
DATA = Path('outputs/ease_broad_task_value_gate_v1')
COLLECT = DATA / 'collection'
OLD = broad.OLD
SEEDS = [126100, 126101, 126102]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2))


def setup():
    tasks, models, hashes = broad.setup()
    previous.ROOT = ROOT
    previous.core.ROOT = ROOT
    return tasks, models, hashes


def family(tasks, case):
    seen = set()
    while tasks[case].get('source_case', case) != case:
        assert case not in seen
        seen.add(case)
        case = tasks[case]['source_case']
    return case


def grouped_folds(tasks):
    sources = sorted({family(tasks, i) for i, t in enumerate(tasks) if not t['validation']})
    assert len(sources) == 8
    rng = np.random.default_rng(126000)
    assignments = {}
    for b, body in enumerate(sorted({tasks[i]['arm']['seed'] for i in sources})):
        members = [i for i in sources if tasks[i]['arm']['seed'] == body]
        assert len(members) == 2
        for j, source in enumerate(rng.permutation(members)):
            assignments[int(source)] = (b + 2*j) % 4
    folds = {i: assignments[family(tasks, i)] for i, t in enumerate(tasks) if not t['validation']}
    assert all(sum(v == f for v in folds.values()) == 12 for f in range(4))
    return folds


FOLD_DESCRIPTION = 'Four folds, each holding two complete source families from different bodies. Eight ancestral source families total: original placement, prior shifted placement, and all four added descendants stay together, across every phase and estimator seed. Twelve placements per fold. All four bodies remain represented in each training fold. Held placements excluded.'


def refold():
    # Correct ancestry grouping before any gate is fitted or outcomes are inspected.
    assert not list((ROOT / 'models').iterdir())
    tasks = read(ROOT / 'cases.json')
    save(ROOT / 'folds.json', grouped_folds(tasks))
    p = read(ROOT / 'protocol.json')
    p['folds'] = FOLD_DESCRIPTION
    p['grouping_note'] = 'Transitive source ancestry grouping verified before any E126 training; direct-parent-only draft replaced before use.'
    save(ROOT / 'protocol.json', p)
    shutil.copy2(__file__, ROOT / 'source' / Path(__file__).name)


def prepare():
    tasks, _, hashes = setup()
    ROOT.mkdir(exist_ok=False)
    for folder in ('trials', 'states', 'actions', 'source', 'models'):
        (ROOT / folder).mkdir()
    folds = grouped_folds(tasks)
    assert len(folds) == 48
    save(ROOT / 'cases.json', tasks)
    save(ROOT / 'folds.json', folds)
    grid = [dict(min_advantage=a, max_spread=s) for a in (.1, .25, .5, .75, .9) for s in (.05, .1, .25, .5, 2.)]
    grid.append(dict(min_advantage=2., max_spread=0.))
    inputs = [DATA / n for n in ('gate_dataset.npz', 'gate_rows.json', 'cases.json', 'records.json', 'summary.json', 'verification.json')]
    inputs += [COLLECT / 'records.json', OLD / 'records.json', OLD / 'summary.json']
    protocol = dict(experiment='E126', date='2026-11-15', inputs={str(p): fingerprint(p) for p in inputs}, checkpoints=hashes,
                    folds=FOLD_DESCRIPTION,
                    model='Three 139-64-32-2 tanh MLPs, seeds126100..126102; 1500 full-batch AdamW steps, lr.001 weight_decay.001, paired BCE. Fold-local normalization and training only. Final three models fit all clean fitting rows.',
                    rule='At phase entry, all three predicted off-minus-on advantages must exceed min_advantage and their range must not exceed max_spread. Otherwise continue E120. At most one up-to8-step off pulse, then E120 resumes. No physics selector. Agreement is not calibrated confidence.',
                    grid=grid, selection='Using out-of-fold complete-episode outcomes from existing matched continuations: require zero regressions versus E120; maximize gains, then minimize switches, then larger min_advantage, then smaller max_spread. All-abstain included. No held optimization. Pooled validation is used for selection, not an unbiased test estimate.',
                    validation='144 fitting placement/estimator episodes. Simulate first accepted clean phase using saved exact counterfactual outcomes. Post-collision phase entries omitted: success already irreversibly false, so they cannot change task success. Runtime has no collision flag input. All72 original episodes retested after selection is frozen.',
                    baseline='E120 unchanged. Original gains and previous gate successes audited separately. Existing development held placements, not a fresh final benchmark or unseen bodies.')
    save(ROOT / 'protocol.json', protocol)
    # Copy only the comparison episodes; refer to the immutable large collection by hash.
    rows = []
    for r in read(DATA / 'records.json'):
        r = dict(r)
        source = r['method']
        method = source.replace('candidate_', 'broadgate_')
        r['method'] = method
        a = f"{r['case']}_{source}_frozen_bias"
        b = f"{r['case']}_{method}_frozen_bias"
        trial = read(DATA / 'trials' / f'{a}.json')
        trial['method'] = method
        save(ROOT / 'trials' / f'{b}.json', trial)
        for ext in ('npz', 'xml'):
            shutil.copy2(DATA / 'states' / f'{a}.{ext}', ROOT / 'states' / f'{b}.{ext}')
        rows.append(r)
    assert len(rows) == 312
    save(ROOT / 'records.json', rows)
    for name in ('abstaining_task_value_gate.py', 'broad_task_value_gate.py', 'bounded_history_correction.py', 'training_estimator_core.py', 'local_probe.py', 'local_history.py'):
        shutil.copy2(Path(__file__).with_name(name), ROOT / 'source' / name)
    print('Prepared E126:48 fitting placements in four source-family folds', flush=True)


def fit(x, y, mask, directory):
    directory.mkdir(exist_ok=False)
    mean = x[mask].mean(0)
    scale = np.maximum(x[mask].std(0), 1e-6)
    np.savez_compressed(directory / 'normalization.npz', mean=mean, scale=scale, training_indices=np.flatnonzero(mask))
    xx = torch.tensor(np.clip((x[mask] - mean) / scale, -10, 10))
    yy = torch.tensor(y[mask])
    models = []
    losses = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = Gate()
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
        for _ in range(1500):
            loss = nn.functional.binary_cross_entropy_with_logits(model(xx), yy)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        torch.save(model.state_dict(), directory / f'{seed}.pt')
        models.append(model)
        losses.append(float(loss.detach()))
    save(directory / 'training.json', dict(seeds=SEEDS, rows=int(mask.sum()), losses=losses, steps=1500))
    return models, mean, scale


def probabilities(models, mean, scale, x):
    with torch.no_grad():
        xx = torch.tensor(np.clip((np.asarray(x, np.float32) - mean) / scale, -10, 10))
        return np.stack([m(xx).sigmoid().numpy() for m in models])


def accepts(prob, rule):
    advantages = prob[..., 1] - prob[..., 0]
    return (advantages.min(axis=0) > rule['min_advantage']) & (np.ptp(advantages, axis=0) <= rule['max_spread'])


def outcome_grid(prob, meta, tasks, grid):
    records = read(COLLECT / 'records.json')
    lookup = {(r['case'], r['seed'], r['off_phase']): r for r in records}
    index = {(r['case'], r['seed'], r['phase']): i for i, r in enumerate(meta)}
    scores = []
    for rule in grid:
        accepted = accepts(prob, rule)
        episodes = []
        for case, t in enumerate(tasks):
            if t['validation']:
                continue
            for seed in range(3):
                phase = next((p for p in PHASES if (case, seed, p) in index and accepted[index[case, seed, p]]), None)
                on = lookup[case, seed, None]['success']
                off = lookup[case, seed, phase]['success']
                episodes.append(dict(case=case, seed=seed, phase=phase, baseline_success=on, success=off))
        scores.append(dict(rule=rule, n=len(episodes), baseline_success=sum(r['baseline_success'] for r in episodes),
                           success=sum(r['success'] for r in episodes), gains=sum(r['success'] and not r['baseline_success'] for r in episodes),
                           losses=sum(r['baseline_success'] and not r['success'] for r in episodes), switches=sum(r['phase'] is not None for r in episodes), episodes=episodes))
    return scores


def train():
    torch.set_num_threads(4)
    tasks = read(ROOT / 'cases.json')
    meta = read(DATA / 'gate_rows.json')
    f = np.load(DATA / 'gate_dataset.npz')
    x, y = f['x'], f['y']
    folds = {int(k): v for k, v in read(ROOT / 'folds.json').items()}
    row_fold = np.array([folds[r['case']] for r in meta])
    assert all(not tasks[r['case']]['validation'] for r in meta)
    prob = np.empty((3, len(meta), 2), np.float32)
    audits = []
    for fold in range(4):
        mask = row_fold != fold
        models, mean, scale = fit(x, y, mask, ROOT / 'models' / f'fold_{fold}')
        prob[:, ~mask] = probabilities(models, mean, scale, x[~mask])
        groups = lambda rows: {family(tasks, meta[i]['case']) for i in np.flatnonzero(rows)}
        assert not groups(mask) & groups(~mask)
        audits.append(dict(fold=fold, train_sources=sorted(groups(mask)), validation_sources=sorted(groups(~mask)), train_rows=int(mask.sum()), validation_rows=int((~mask).sum())))
        print('trained fold', fold, flush=True)
    np.savez_compressed(ROOT / 'oof_predictions.npz', probabilities=prob, row_fold=row_fold)
    save(ROOT / 'fold_audit.json', audits)
    scores = outcome_grid(prob, meta, tasks, read(ROOT / 'protocol.json')['grid'])
    safe = [s for s in scores if s['losses'] == 0]
    selected = max(safe, key=lambda s: (s['gains'], -s['switches'], s['rule']['min_advantage'], -s['rule']['max_spread']))
    save(ROOT / 'validation_grid.json', scores)
    save(ROOT / 'selected_rule.json', selected)
    fit(x, y, np.ones(len(x), bool), ROOT / 'models' / 'final')
    freeze = {str(p.relative_to(ROOT)): fingerprint(p) for p in (ROOT / 'models').rglob('*') if p.is_file()}
    for name in ('selected_rule.json', 'protocol.json', 'folds.json', 'oof_predictions.npz', 'fold_audit.json', 'validation_grid.json'):
        freeze[name] = fingerprint(ROOT / name)
    save(ROOT / 'frozen_selection.json', freeze)
    print('FROZEN', json.dumps({k: v for k, v in selected.items() if k != 'episodes'}), flush=True)


def load_gate():
    models = []
    for seed in SEEDS:
        m = Gate()
        m.load_state_dict(torch.load(ROOT / 'models' / 'final' / f'{seed}.pt', weights_only=True))
        m.eval()
        models.append(m)
    f = np.load(ROOT / 'models' / 'final' / 'normalization.npz')
    return models, f['mean'], f['scale'], read(ROOT / 'selected_rule.json')['rule']


def evaluate(case, t, seed, models, gate, mean, scale, rule, save_result=True):
    method = f'candidate_{seed}'
    original_rollout, original_action = previous.core.probe_rollout, previous.bounded_action
    counts, decisions = {}, []
    chosen, off, step_index = None, False, 0
    def bounded(a, b):
        return a.copy() if off else original_action(a, b)
    def observer(env, home, c, **kwargs):
        original_step = env.step
        def step(action, aperture):
            nonlocal chosen, off, step_index
            phase = env.phase
            k = counts.get(phase, 0)
            counts[phase] = k + 1
            if phase in PHASES and k == 0 and chosen is None:
                obs, old = env.observe(), c.model
                try:
                    c.model = models[f'correction_{seed}']
                    j, bias = c.predict()
                finally:
                    c.model = old
                error = np.r_[env.goal - obs['position'], ROT_SCALE * rotation_error(np.eye(3), obs['rotation'])]
                proposal = np.clip(j.T @ np.linalg.solve(j @ j.T + .015**2 * np.eye(6), .35 * error) + bias, -.04, .04)
                x = features(c.n, c.gram, c.cross, c.bias, obs['q'], obs['position'], env.goal, phase, c.local, action, proposal)
                prob = probabilities(gate, mean, scale, x)
                selected = bool(accepts(prob, rule))
                decisions.append(dict(step=step_index, phase=phase, features=x.tolist(), probabilities=prob.tolist(), selected=selected))
                if selected:
                    chosen = phase
            off = phase == chosen and k < 8
            step_index += 1
            return original_step(action, aperture)
        env.step = step
        return original_rollout(env, home, c, **kwargs)
    with patch.object(previous.core, 'probe_rollout', observer), patch.object(previous, 'bounded_action', bounded):
        r, _ = previous.evaluate(case, t, method, models[f'reference_{seed}'], models[f'correction_{seed}'], 'bounded', save=save_result)
    if save_result:
        save(ROOT / 'actions' / f'{case}_{method}_gate.json', dict(chosen=chosen, decisions=decisions))
    return r, chosen


def run():
    tasks, models, hashes = setup()
    gate, mean, scale, rule = load_gate()
    frozen = read(ROOT / 'frozen_selection.json')
    for p, h in frozen.items():
        assert fingerprint(ROOT / p) == h
    rows = read(ROOT / 'records.json')
    done = {(r['case'], r['method']) for r in rows}
    for case, t in enumerate(tasks[:24]):
        for seed in range(3):
            if (case, f'candidate_{seed}') in done:
                continue
            r, chosen = evaluate(case, t, seed, models, gate, mean, scale, rule)
            old = read(ROOT / 'trials' / f'{case}_unbounded_{seed}_frozen_bias.json')
            assert r['calibration'] == old['calibration'] and r['initial_estimator'] == old['initial_estimator']
            if chosen is None:
                old['method'] = r['method']
                assert old == r
                np.testing.assert_array_equal(np.load(ROOT / 'states' / f'{case}_unbounded_{seed}_frozen_bias.npz')['states'], np.load(ROOT / 'states' / f'{case}_candidate_{seed}_frozen_bias.npz')['states'])
            rows.append({k: v for k, v in r.items() if k not in ('trace', 'telemetry', 'calibration', 'initial_estimator', 'final_estimator')} | dict(gate_phase=chosen))
            save(ROOT / 'records.json', rows)
        print('evaluated', case, flush=True)
    assert len(rows) == 384
    for p, h in hashes.items():
        assert fingerprint(p) == h
    for p, h in frozen.items():
        assert fingerprint(ROOT / p) == h


if __name__ == '__main__':
    import sys
    {'prepare': prepare, 'refold': refold, 'train': train, 'evaluate': run}[sys.argv[1]]()
