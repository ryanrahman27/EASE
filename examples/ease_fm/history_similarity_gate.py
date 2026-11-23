"""E127: fitting-family validation of history-supported interventions."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import abstaining_task_value_gate as prior
from .abstaining_task_value_gate import (DATA, COLLECT, OLD, SEEDS, PHASES, Gate, read, save, family,
    probabilities, accepts, previous, features, ROT_SCALE, rotation_error, fingerprint)

ROOT = Path('outputs/ease_history_similarity_gate_v1')
PRIOR = prior.ROOT
SUPPORT = None


def setup():
    tasks, models, hashes = prior.setup()
    previous.ROOT = previous.core.ROOT = ROOT
    return tasks, models, hashes


def group(x):
    return int(np.argmax(x[:9])), int(x[9:16].sum()), int(x[138])


class Support:
    def __init__(self, x, families, indices):
        self.x = np.asarray(x, np.float64)
        self.families = np.asarray(families)
        self.indices = np.asarray(indices)
        self.mean = self.x[:, 33:124].mean(0)
        self.scale = np.maximum(self.x[:, 33:124].std(0), 1e-6)
        self.z = (self.x[:, 33:124]-self.mean)/self.scale
        self.groups = [group(row) for row in self.x]
        self.loo = np.array([self.distance(row, self.families[i])[0] for i, row in enumerate(self.x)])

    def distance(self, x, exclude_family=None):
        key = group(x)
        ids = np.array([i for i, g in enumerate(self.groups) if g == key and (exclude_family is None or self.families[i] != exclude_family)], dtype=int)
        if not len(ids):
            return float('inf'), -1
        n = key[1]
        active = np.r_[np.arange(49).reshape(7, 7)[:n, :n].ravel(), 49+np.arange(42).reshape(7, 6)[:n].ravel()]
        z = (np.asarray(x, np.float64)[33:124]-self.mean)/self.scale
        distances = np.sqrt(np.mean((self.z[ids][:, active]-z[active])**2, axis=1))
        best = int(np.argmin(distances))
        return float(distances[best]), int(self.indices[ids[best]])

    def query(self, x, quantile):
        distance, nearest = self.distance(x)
        calibration = np.array([self.loo[i] for i, g in enumerate(self.groups) if g == group(x) and np.isfinite(self.loo[i])])
        threshold = float(np.quantile(calibration, quantile)) if len(calibration) else 0.
        supported = bool(np.isfinite(distance) and distance <= threshold and len(calibration))
        return dict(distance=distance if np.isfinite(distance) else None, threshold=threshold,
                    nearest_training_row=nearest, supported=supported)

    def snapshot(self, path):
        np.savez_compressed(path, training_indices=self.indices, families=self.families, mean=self.mean, scale=self.scale, leave_family_out_distance=self.loo)


def prepare():
    tasks, _, hashes = setup()
    ROOT.mkdir(exist_ok=False)
    for folder in ('trials', 'states', 'actions', 'source', 'support'):
        (ROOT/folder).mkdir()
    for name in ('cases.json', 'folds.json', 'oof_predictions.npz', 'fold_audit.json'):
        shutil.copy2(PRIOR/name, ROOT/name)
    shutil.copytree(PRIOR/'models', ROOT/'models')
    grid = [dict(min_advantage=a, max_spread=2., quantile=q) for a in (.1, .5, .9) for q in (.1, .25, .5, .75, .9, 1.)]
    grid.append(dict(min_advantage=2., max_spread=0., quantile=0.))
    inputs = {str(p): fingerprint(p) for p in [DATA/n for n in ('gate_dataset.npz', 'gate_rows.json', 'records.json', 'summary.json', 'verification.json')] + [COLLECT/'records.json', OLD/'records.json', OLD/'summary.json', PRIOR/'frozen_selection.json', PRIOR/'oof_predictions.npz']}
    save(ROOT/'protocol.json', dict(experiment='E127', date='2026-11-15', inputs=inputs, checkpoints=hashes, grid=grid,
        folds=prior.FOLD_DESCRIPTION,
        model='Reuse E126 four fold ensembles and final three models exactly; no neural retraining. Frozen seeds126100..126102. All inherited normalizations and OOF predictions reused by hash.',
        similarity='Nearest standardized RMS distance in measured Gram/cross history only (features33:124), active joint dimensions only. Same phase, joint count, and local-history flag. No positions, goals, actions, IDs, or outcome labels in metric. No clipping of standardized history distances.',
        calibration='Per-fold history mean/std from training rows only, std floor1e-6. For each training row compute nearest distance excluding its complete ancestral family. Per phase/DOF/local threshold is a fixed quantile of finite leave-family-out training distances. Empty support rejects. Final bank/calibration rebuilt from all fitting rows. Runtime has no family or case ID.',
        selection='18 predeclared minimum-advantage/history-quantile rules plus full abstention. Consensus of all three model advantages and history support required. Reconstruct first accepted clean phase using saved full-task counterfactual outcomes. Require zero E120 regressions; maximize gains, then fewer switches, larger advantage, smaller support quantile. No held optimization. Pooled fitting validation is selection data, not an unbiased test estimate.',
        inference='At most one up-to8-step correction-off pulse, then resume E120. Otherwise E120 throughout. No physics selector. Histories from all clean fitting outcomes enter support, not just successes. Similarity is not a calibrated success probability.',
        evaluation='All72 original episodes after freezing; fitting and reused held-development placements reported separately. Existing training bodies only. E120 unchanged. Compare prior gate gains and losses as well.'))
    rows = []
    for r in read(DATA/'records.json'):
        r = dict(r)
        source = r['method']
        r['method'] = source.replace('candidate_', 'broadgate_')
        a, b = f"{r['case']}_{source}_frozen_bias", f"{r['case']}_{r['method']}_frozen_bias"
        trial = read(DATA/'trials'/f'{a}.json')
        trial['method'] = r['method']
        save(ROOT/'trials'/f'{b}.json', trial)
        for ext in ('npz', 'xml'):
            shutil.copy2(DATA/'states'/f'{a}.{ext}', ROOT/'states'/f'{b}.{ext}')
        rows.append(r)
    assert len(rows) == 312
    save(ROOT/'records.json', rows)
    for name in ('history_similarity_gate.py', 'abstaining_task_value_gate.py', 'broad_task_value_gate.py', 'bounded_history_correction.py', 'training_estimator_core.py', 'matched_correction.py', 'local_probe.py', 'local_history.py'):
        shutil.copy2(Path(__file__).with_name(name), ROOT/'source'/name)
    print('Prepared E127 with fixed fitting-only similarity protocol', flush=True)


def data():
    tasks, meta = read(ROOT/'cases.json'), read(DATA/'gate_rows.json')
    x = np.load(DATA/'gate_dataset.npz')['x']
    families = np.array([family(tasks, r['case']) for r in meta])
    fold = np.load(ROOT/'oof_predictions.npz')['row_fold']
    assert all(not tasks[r['case']]['validation'] for r in meta)
    return tasks, meta, x, families, fold


def score_rules(prob, supported, meta, tasks, grid):
    collection = {(r['case'], r['seed'], r['off_phase']): r for r in read(COLLECT/'records.json')}
    index = {(r['case'], r['seed'], r['phase']): i for i, r in enumerate(meta)}
    scores = []
    for k, rule in enumerate(grid):
        accepted = accepts(prob, rule) & supported[k]
        episodes = []
        for case, t in enumerate(tasks):
            if t['validation']:
                continue
            for seed in range(3):
                phase = next((p for p in PHASES if (case, seed, p) in index and accepted[index[case, seed, p]]), None)
                on, off = collection[case, seed, None]['success'], collection[case, seed, phase]['success']
                episodes.append(dict(case=case, seed=seed, phase=phase, baseline_success=on, success=off))
        scores.append(dict(rule=rule, n=len(episodes), baseline_success=sum(r['baseline_success'] for r in episodes), success=sum(r['success'] for r in episodes), gains=sum(r['success'] and not r['baseline_success'] for r in episodes), losses=sum(r['baseline_success'] and not r['success'] for r in episodes), switches=sum(r['phase'] is not None for r in episodes), episodes=episodes))
    return scores


def select(scores):
    return max([s for s in scores if s['losses'] == 0], key=lambda s: (s['gains'], -s['switches'], s['rule']['min_advantage'], -s['rule']['quantile']))


def train():
    tasks, meta, x, families, folds = data()
    grid = read(ROOT/'protocol.json')['grid']
    supported = np.zeros((len(grid), len(x)), bool)
    distances = np.zeros(len(x))
    thresholds = np.zeros((len(grid), len(x)))
    nearest = np.zeros(len(x), int)
    for fold in range(4):
        train_ids, val_ids = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        assert not set(families[train_ids]) & set(families[val_ids])
        bank = Support(x[train_ids], families[train_ids], train_ids)
        bank.snapshot(ROOT/'support'/f'fold_{fold}.npz')
        for i in val_ids:
            for k, rule in enumerate(grid):
                q = bank.query(x[i], rule['quantile'])
                supported[k, i], thresholds[k, i] = q['supported'], q['threshold']
                distances[i], nearest[i] = q['distance'] if q['distance'] is not None else np.inf, q['nearest_training_row']
        print('validated support fold', fold, flush=True)
    np.savez_compressed(ROOT/'oof_support.npz', supported=supported, distances=distances, thresholds=thresholds, nearest=nearest)
    scores = score_rules(np.load(ROOT/'oof_predictions.npz')['probabilities'], supported, meta, tasks, grid)
    save(ROOT/'validation_grid.json', scores)
    selected = select(scores)
    save(ROOT/'selected_rule.json', selected)
    Support(x, families, np.arange(len(x))).snapshot(ROOT/'support'/'final.npz')
    frozen = {str(p.relative_to(ROOT)): fingerprint(p) for folder in ('models', 'support') for p in (ROOT/folder).rglob('*') if p.is_file()}
    for name in ('selected_rule.json', 'protocol.json', 'folds.json', 'oof_predictions.npz', 'fold_audit.json', 'oof_support.npz', 'validation_grid.json'):
        frozen[name] = fingerprint(ROOT/name)
    save(ROOT/'frozen_selection.json', frozen)
    print('FROZEN', json.dumps({k: v for k, v in selected.items() if k != 'episodes'}), flush=True)


def load_gate():
    global SUPPORT
    _, _, x, families, _ = data()
    SUPPORT = Support(x, families, np.arange(len(x)))
    models = []
    for seed in SEEDS:
        m = Gate()
        m.load_state_dict(torch.load(ROOT/'models'/'final'/f'{seed}.pt', weights_only=True))
        m.eval()
        models.append(m)
    n = np.load(ROOT/'models'/'final'/'normalization.npz')
    return models, n['mean'], n['scale'], read(ROOT/'selected_rule.json')['rule']

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
                support = SUPPORT.query(x, rule['quantile'])
                selected = bool(accepts(prob, rule)) and support['supported']
                decisions.append(dict(step=step_index, phase=phase, features=x.tolist(), probabilities=prob.tolist(), support=support, selected=selected))
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
    {'prepare': prepare, 'train': train, 'evaluate': run}[sys.argv[1]]()
