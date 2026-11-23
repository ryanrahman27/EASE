"""E127 report, independent support audit, and saved-video provenance."""
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from . import history_similarity_gate as hs
from .history_similarity_gate import (ROOT, DATA, COLLECT, OLD, PRIOR, SEEDS, PHASES, Gate, read, save,
    setup, load_gate, evaluate, probabilities, accepts, family, fingerprint, Support, data, score_rules, select)


def build():
    s = Path(__file__).with_name('abstaining_gate_videos.py').read_text()
    s = s.replace('abstaining_task_value_gate import', 'history_similarity_gate import').replace('E126', 'E127').replace('Abstaining gate / E120 fallback', 'History similarity / E120 fallback')
    Path(__file__).with_name('history_similarity_videos.py').write_text(s)


def support_audit():
    tasks, meta, x, families, folds = data()
    grid = read(ROOT/'protocol.json')['grid']
    saved = np.load(ROOT/'oof_support.npz')
    supported = np.zeros_like(saved['supported'])
    for fold in range(4):
        ids, val = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        assert not set(families[ids]) & set(families[val])
        bank = Support(x[ids], families[ids], ids)
        stored = np.load(ROOT/'support'/f'fold_{fold}.npz')
        for key, value in dict(training_indices=ids, families=families[ids], mean=bank.mean, scale=bank.scale, leave_family_out_distance=bank.loo).items():
            np.testing.assert_array_equal(stored[key], value)
        for i in val:
            for k, rule in enumerate(grid):
                q = bank.query(x[i], rule['quantile'])
                supported[k, i] = q['supported']
                assert q['nearest_training_row'] == saved['nearest'][i]
                assert q['nearest_training_row'] in ids or q['nearest_training_row'] == -1
                assert q['threshold'] == saved['thresholds'][k, i]
                assert (q['distance'] if q['distance'] is not None else np.inf) == saved['distances'][i]
        # Only history and matching context affect support, not goals or action proposals.
        probe = x[val[0]].copy()
        expected = bank.query(probe, .5)
        probe[16:33] += 10
        probe[124:138] += 10
        assert bank.query(probe, .5) == expected
    np.testing.assert_array_equal(supported, saved['supported'])
    bank = Support(x, families, np.arange(len(x)))
    stored = np.load(ROOT/'support'/'final.npz')
    for key, value in dict(training_indices=np.arange(len(x)), families=families, mean=bank.mean, scale=bank.scale, leave_family_out_distance=bank.loo).items():
        np.testing.assert_array_equal(stored[key], value)
    for p in (ROOT/'models').rglob('*'):
        if p.is_file():
            assert fingerprint(p) == fingerprint(PRIOR/p.relative_to(ROOT))
    return score_rules(np.load(ROOT/'oof_predictions.npz')['probabilities'], supported, meta, tasks, grid)


def write_report(summary):
    grid, selected = read(ROOT/'validation_grid.json'), read(ROOT/'selected_rule.json')
    a = summary['aggregate']
    candidate = next(r for r in a if r['method']=='candidate' and r['split']==r['dof']=='all')
    lines = ['# E127: History similarity as an abstention signal', '',
             f"Candidate: {candidate['success']}/72 successes, {candidate['collisions']} collision episodes. E120 remains the immutable baseline and fallback.", '',
             '## Fixed protocol', '']
    p = read(ROOT/'protocol.json')
    for key in ('folds', 'model', 'similarity', 'calibration', 'selection', 'inference', 'evaluation'):
        lines += [f"**{key}:** {p[key]}", '']
    lines += ['Reuse1108 clean fitting pairs and1440 saved collection episodes. No new neural fitting or training rollouts. Fitting-bank membership is independent of outcome labels; nearest-neighbor history distance is not a success probability. All placements from an ancestral family stay together. Final fitting evaluations are in-sample and can match their own history-bank entries; they do not measure generalization.', '',
              'Validation outcomes reconstruct the first accepted clean phase using matched complete-episode continuations. Phase starts after an earlier collision are excluded because success is already irreversibly false; runtime receives no collision flag. The same zero-regression validation criterion as E126 is used. Full abstention is feasible by construction. Selected rules are not guaranteed to preserve held success.', '',
              '## Fitting-family validation', '', '| Min advantage | History quantile | Success /144 | Gains | Losses | Switches |', '|---:|---:|---:|---:|---:|---:|']
    for r in grid:
        lines.append(f"| {r['rule']['min_advantage']} | {r['rule']['quantile']} | {r['success']} | {r['gains']} | {r['losses']} | {r['switches']} |")
    # Describe all outcome-changing fitting-validation choices of the least
    # restrictive reused ensemble, without tuning the frozen support rule.
    meta = read(DATA/'gate_rows.json')
    index = {(r['case'], r['seed'], r['phase']): i for i, r in enumerate(meta)}
    assert fingerprint(PRIOR/'validation_grid.json') == read(PRIOR/'frozen_selection.json')['validation_grid.json']
    previous_grid = read(PRIOR/'validation_grid.json')
    previous_rule = next(r for r in previous_grid if r['rule']==dict(min_advantage=.1, max_spread=2.))
    support = np.load(ROOT/'oof_support.npz')
    columns = [next(k for k,r in enumerate(grid) if r['rule']['min_advantage']==.1 and r['rule']['quantile']==q) for q in (.1,.75,1.)]
    diagnostics = []
    for r in previous_rule['episodes']:
        if r['success']==r['baseline_success']:
            continue
        i = index[r['case'],r['seed'],r['phase']]
        diagnostics.append(r | dict(distance=float(support['distances'][i]), thresholds={str(q):float(support['thresholds'][k,i]) for q,k in zip((.1,.75,1.),columns)}))
    save(ROOT/'support_diagnostics.json', diagnostics)
    lines += ['', 'History support on every outcome-changing choice of the reused ensemble without the history filter (fitting validation only):', '', '| Case / seed | Phase | Outcome change | Distance | 10% cutoff | 75% cutoff | Maximum cutoff |', '|---|---|---|---:|---:|---:|---:|']
    for r in diagnostics:
        t=r['thresholds']
        lines.append(f"| {r['case']}/{r['seed']} | {r['phase']} | {'gain' if r['success'] else 'loss'} | {r['distance']:.6f} | {t['0.1']:.6f} | {t['0.75']:.6f} | {t['1.0']:.6f} |")
    lines += ['', 'In this fixed grid, tighter history support rejects harmful choices but also the only successful rescue. No supported rule gains an episode. Rules through the75% cutoff lose none;90% and maximum-cutoff rules lose one. The fewer-switches tie-break selects full abstention. This does not establish that all history metrics fail, only that this standardized Gram/cross nearest-neighbor rule is not useful under the tested selection criterion.', '']
    lines += ['', f"Selected `{selected['rule']}`: {selected['success']}/144 versus E120 {selected['baseline_success']}/144, {selected['gains']} gains, {selected['losses']} losses, {selected['switches']} switches. Frozen before evaluation.", '',
              '## Complete original-episode retest', '', 'reference=E118; unbounded=E120 (bounded correction without a gate); previousgate=E124; broadgate=E125; candidate=E127. E126 full abstention exactly matched E120 and is represented by the E120 row.', '',
              '| Method | Split | DOF | Success | Collision episodes |', '|---|---|---|---:|---:|']
    for r in a:
        lines.append(f"| {r['method']} | {r['split']} | {r['dof']} | {r['success']}/{r['n']} | {r['collisions']} |")
    lines += ['', f"Retained {summary['gains_retained']}/6 original E120 gains; recovered {summary['regressions_recovered']}/4 regressions. No E120 success lost: {summary['no_e120_losses']}. Strict preservation of all prior successful episodes and recovery of four regressions: {summary['strict_retention_passed']}.", '',
              '## Every paired outcome change', '', '```json', json.dumps(summary['paired_changes'], indent=2), '```', '',
              '## Interpretation', '',
              'Read held changes separately from fitting rescues. History support is a diagnostic abstention heuristic, not a calibrated safety guarantee. No held data selected the cutoff, and these are reused development placements on training bodies. No unseen-body claim is made. E120 remains baseline; a higher aggregate alone is insufficient for promotion. All negative outcomes and videos are retained.', '']
    (ROOT/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (method, label) in enumerate((('unbounded','E120 / E126'), ('previousgate','E124 gate'), ('broadgate','E125 broader gate'), ('candidate','E127 history support'))):
        rr = [next(r for r in a if r['method']==method and r['split']==sp and r['dof']=='all') for sp in ('fitting','held')]
        xx = np.arange(2)+(i-1.5)*.20
        ax.bar(xx, [100*r['success']/r['n'] for r in rr], .18, label=label)
        for x, r in zip(xx, rr):
            ax.text(x, 100*r['success']/r['n']+2, f"{r['success']}/{r['n']}", ha='center')
    ax.set_xticks([0,1], ['Original fitting placements','Held development placements'])
    ax.set_ylim(0,125)
    ax.set_ylabel('Task success (%)')
    ax.legend(ncol=4, loc='upper center')
    fig.tight_layout()
    fig.savefig(ROOT/'comparison.png', dpi=160)
    plt.close(fig)
    path = Path(__file__).with_name('TECHNICAL_REPORT.md')
    text = path.read_text(encoding='utf-8')
    marker = '## E127: history-similarity abstention'
    assert marker not in text
    text = text.replace('Revision 114 · 15 November 2026', 'Revision 115 · 15 November 2026', 1)
    text += f"\n{marker}\n\nReused E126 ensemble weights; fitting-family-only history banks and leave-family-out distance calibration. Selected rule{selected['rule']} gives{selected['gains']} gains/{selected['losses']} losses over144 fitting validation episodes. Final candidate{candidate['success']}/72; retained{summary['gains_retained']}/6 original E120 gains and recovered{summary['regressions_recovered']}/4 regressions. Strict retention{summary['strict_retention_passed']}. No held optimization or unseen bodies. E120 unchanged. Full report: `outputs/ease_history_similarity_gate_v1/report.md`.\n"
    path.write_text(text, encoding='utf-8')
    shutil.copy2(__file__, ROOT/'source'/Path(__file__).name)
    print(json.dumps(summary, indent=2), flush=True)


def finish():
    v, s, p = read(ROOT/'verification.json'), read(ROOT/'summary.json'), read(ROOT/'protocol.json')
    m = read(ROOT/'videos'/'manifest.json')
    for name, h in {**p['inputs'], **p['checkpoints']}.items():
        assert fingerprint(name) == h
    for name, h in read(ROOT/'frozen_selection.json').items():
        assert fingerprint(ROOT/name) == h
    for path in (ROOT/'source').glob('*.py'):
        assert fingerprint(path) == fingerprint(Path(__file__).with_name(path.name))
    for r in m['videos']:
        assert fingerprint(ROOT/'videos'/r['file']) == r['sha256']
    save(ROOT/'status.json', dict(status='complete', new_training_rollouts=0, neural_weights_retrained=False, fitting_pairs=1108, grouped_folds=4, new_evaluation_episodes=72,
        no_e120_losses=s['no_e120_losses'], strict_retention_passed=s['strict_retention_passed'], videos=len(m['videos']), held_training_rows=0, unseen_evaluated=False))
    path = ROOT/'report.md'
    path.write_text(path.read_text()+f"\n## Verification and media\n\nAll fold history banks, leave-family-out distances, OOF support decisions and selected rule independently reconstructed. Fold normalizations and predictions reproduced;1108 labels verified; no held training rows. {v['decisions_checked']} runtime decisions and{v['action_steps_checked']} actions checked; {v['fitting_continuations_exact']} fitting continuations exact, {v['e120_continuations_exact']} no-switch E120 continuations exact, two held episodes exactly replayed. Frozen E126 models and E120 inputs unchanged. Saved{len(m['videos'])} trace-verified success/failure videos.\n")
    path = ROOT/'videos'/'index.html'
    path.write_text(path.read_text().replace('</h1>', '</h1><p>E127: fitting-family-calibrated history similarity. E120 is the fallback. reference=E118; unbounded=E120; previousgate=E124; broadgate=E125; candidate=E127. See the report for fitting versus held outcomes.</p>', 1))
    hashes = read(ROOT/'artifact_hashes.json')
    for name in ('report.md','status.json','videos/index.html','verification.json'):
        hashes[name] = fingerprint(ROOT/name)
    assert set(hashes) == {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
    save(ROOT/'artifact_hashes.json', hashes)
    print(json.dumps(read(ROOT/'status.json'), indent=2), flush=True)


def diagnose():
    tasks, meta, x, families, folds = data()
    index = {(r['case'], r['seed'], r['phase']): i for i,r in enumerate(meta)}
    rows = read(ROOT/'support_diagnostics.json')
    for r in rows:
        i=index[r['case'],r['seed'],r['phase']]
        r.update(fold=int(folds[i]), support_group=list(hs.group(x[i])))
    gain=next(r for r in rows if r['success'])
    comparable=[r for r in rows if not r['success'] and r['fold']==gain['fold'] and r['support_group']==gain['support_group']]
    assert comparable and all(r['distance'] < gain['distance'] for r in comparable)
    save(ROOT/'support_diagnostics.json', rows)
    text='\nThe useful descent intervention and a harmful descent intervention share the same validation fold and support group. The useful example has distance3.685777 versus2.871023 for the harmful one. Therefore a single upper-distance cutoff in this metric cannot accept the useful example while rejecting that harmful example. This observed ordering points to the representation/metric, rather than merely a too-tight cutoff. It is a fitting-validation diagnostic, not a new held-tuned rule.\n'
    path=ROOT/'report.md'
    path.write_text(path.read_text()+text)
    path=Path(__file__).with_name('TECHNICAL_REPORT.md')
    path.write_text(path.read_text(encoding='utf-8')+text, encoding='utf-8')
    shutil.copy2(__file__, ROOT/'source'/Path(__file__).name)
    print(text, flush=True)

def report():
    tasks, rows = read(ROOT / 'cases.json'), read(ROOT / 'records.json')
    assert len(rows) == 384
    lookup = {(r['case'], r['method']): r for r in rows}
    methods = ('reference', 'unbounded', 'previousgate', 'broadgate', 'candidate')
    aggregate, changes = [], []
    for method in methods:
        for split in ('all', 'fitting', 'held'):
            for dof in ('all', 6, 7):
                rr = [r for r in rows if r['method'].startswith(method+'_') and (split == 'all' or tasks[r['case']]['validation'] == (split == 'held')) and (dof == 'all' or r['dof'] == dof)]
                aggregate.append(dict(method=method, split=split, dof=str(dof), n=len(rr), success=sum(r['success'] for r in rr), collisions=sum(r['collision'] for r in rr)))
    for baseline in methods[:-1]:
        for seed in range(3):
            for case in range(24):
                a, b = lookup[case, f'{baseline}_{seed}'], lookup[case, f'candidate_{seed}']
                if a['success'] != b['success']:
                    changes.append(dict(baseline=baseline, case=case, seed=seed, gain=b['success'], split='held' if tasks[case]['validation'] else 'fitting', dof=b['dof']))
    targets = [dict(case=c['case'], seed=p['seed'], previous_gain=c['gain'], success=lookup[c['case'], f"candidate_{p['seed']}"]['success']) for p in read(OLD / 'summary.json')['paired'] for c in p['changes']]
    retained = sum(t['success'] for t in targets if t['previous_gain'])
    recovered = sum(t['success'] for t in targets if not t['previous_gain'])
    no_e120_losses = not any(c['baseline'] == 'unbounded' and not c['gain'] for c in changes)
    strict = retained == 6 and recovered == 4 and not any(not c['gain'] for c in changes)
    summary = dict(aggregate=aggregate, paired_changes=changes, targeted_pairs=targets, gains_retained=retained, regressions_recovered=recovered,
                   no_e120_losses=no_e120_losses, strict_retention_passed=strict, selected_rule=read(ROOT / 'selected_rule.json')['rule'])
    save(ROOT / 'summary.json', summary)
    write_report(summary)

def verify():
    torch.set_num_threads(4)
    tasks, models, hashes = setup()
    f = np.load(DATA / 'gate_dataset.npz')
    x, y = f['x'], f['y']
    meta = read(DATA / 'gate_rows.json')
    folds = {int(k): v for k, v in read(ROOT / 'folds.json').items()}
    row_fold = np.array([folds[r['case']] for r in meta])
    oof = np.load(ROOT / 'oof_predictions.npz')
    np.testing.assert_array_equal(row_fold, oof['row_fold'])
    assert all(not tasks[r['case']]['validation'] for r in meta)
    for fold in range(4):
        mask = row_fold != fold
        a = {family(tasks, meta[i]['case']) for i in np.flatnonzero(mask)}
        b = {family(tasks, meta[i]['case']) for i in np.flatnonzero(~mask)}
        assert not a & b
        n = np.load(ROOT / 'models' / f'fold_{fold}' / 'normalization.npz')
        np.testing.assert_array_equal(n['training_indices'], np.flatnonzero(mask))
        np.testing.assert_array_equal(n['mean'], x[mask].mean(0))
        np.testing.assert_array_equal(n['scale'], np.maximum(x[mask].std(0), 1e-6))
        mm = []
        for seed in SEEDS:
            m = Gate()
            m.load_state_dict(torch.load(ROOT / 'models' / f'fold_{fold}' / f'{seed}.pt', weights_only=True))
            m.eval()
            mm.append(m)
        np.testing.assert_array_equal(probabilities(mm, n['mean'], n['scale'], x[~mask]), oof['probabilities'][:, ~mask])
    collection = {(r['case'], r['seed'], r['off_phase']): r for r in read(COLLECT / 'records.json')}
    for i, r in enumerate(meta):
        np.testing.assert_array_equal(y[i], [collection[r['case'], r['seed'], phase]['success'] for phase in (None, r['phase'])])
    scores = support_audit()
    assert scores == read(ROOT / 'validation_grid.json')
    selected = select(scores)
    assert selected == read(ROOT / 'selected_rule.json')
    gate, mean, scale, rule = load_gate()
    np.testing.assert_array_equal(mean, x.mean(0))
    np.testing.assert_array_equal(scale, np.maximum(x.std(0), 1e-6))
    n = np.load(ROOT / 'models' / 'final' / 'normalization.npz')
    np.testing.assert_array_equal(n['training_indices'], np.arange(len(x)))
    rows = read(ROOT / 'records.json')
    checked = decisions = exact_fit = exact_e120 = switches = 0
    for r in rows:
        if not r['method'].startswith('candidate_'):
            continue
        case, seed = r['case'], int(r['method'].split('_')[-1])
        stem = f"{case}_{r['method']}_frozen_bias"
        d = read(ROOT / 'actions' / f"{case}_{r['method']}_gate.json")
        assert sum(t['selected'] for t in d['decisions']) <= 1
        switches += d['chosen'] is not None
        for t in d['decisions']:
            prob = probabilities(gate, mean, scale, t['features'])
            np.testing.assert_array_equal(prob, np.array(t['probabilities'], np.float32))
            support = hs.SUPPORT.query(np.asarray(t['features'], np.float32), rule['quantile'])
            assert support == t['support']
            assert (bool(accepts(prob, rule)) and support['supported']) == t['selected']
            decisions += 1
        counts = {}
        for action in read(ROOT / 'actions' / f'{stem}.json'):
            phase = action['phase']
            k = counts.get(phase, 0)
            counts[phase] = k+1
            if not action['active']:
                continue
            a, b = np.array(action['base_action']), np.array(action['correction_action'])
            expected = a if phase == d['chosen'] and k < 8 else np.clip(a + np.clip(.25*(b-a), -.001, .001), -.04, .04)
            np.testing.assert_array_equal(expected, action['nominal_action'])
            checked += 1
        new = read(ROOT / 'trials' / f'{stem}.json')
        if not tasks[case]['validation']:
            method = f'on_{seed}' if d['chosen'] is None else f"off_{d['chosen']}_{seed}"
            old = read(COLLECT / 'trials' / f'{case}_{method}_frozen_bias.json')
            old['method'] = r['method']
            assert old == new
            np.testing.assert_array_equal(np.load(COLLECT / 'states' / f'{case}_{method}_frozen_bias.npz')['states'], np.load(ROOT / 'states' / f'{stem}.npz')['states'])
            exact_fit += 1
        if d['chosen'] is None:
            old = read(ROOT / 'trials' / f'{case}_unbounded_{seed}_frozen_bias.json')
            old['method'] = r['method']
            assert old == new
            np.testing.assert_array_equal(np.load(ROOT / 'states' / f'{case}_unbounded_{seed}_frozen_bias.npz')['states'], np.load(ROOT / 'states' / f'{stem}.npz')['states'])
            exact_e120 += 1
    replays = []
    for outcome in (True, False):
        r = next(r for r in rows if r['method'].startswith('candidate_') and tasks[r['case']]['validation'] and r['success'] == outcome)
        case, seed = r['case'], int(r['method'].split('_')[-1])
        stem = f"{case}_{r['method']}_frozen_bias"
        old = read(ROOT / 'trials' / f'{stem}.json')
        states = np.load(ROOT / 'states' / f'{stem}.npz')['states'].copy()
        new, _ = evaluate(case, tasks[case], seed, models, gate, mean, scale, rule)
        assert old == new
        np.testing.assert_array_equal(states, np.load(ROOT / 'states' / f'{stem}.npz')['states'])
        replays.append(stem)
    protocol = read(ROOT / 'protocol.json')
    for path, h in {**hashes, **protocol['inputs']}.items():
        assert fingerprint(path) == h
    for path, h in read(ROOT / 'frozen_selection.json').items():
        assert fingerprint(ROOT / path) == h
    save(ROOT / 'verification.json', dict(history_support_reconstructed=True, unchanged_E126_models=True, ancestral_families_disjoint=True, fold_normalization_training_only=True, held_training_rows=0, fitting_rows=len(meta),
         oof_predictions_exact=True, validation_selection_reproduced=True, labels_verified=True, decisions_checked=decisions, action_steps_checked=checked,
         fitting_continuations_exact=exact_fit, e120_continuations_exact=exact_e120, switches=switches, held_exact_replays=replays, frozen_weights_and_selection_unchanged=True))
    print('Verified', checked, 'actions;', exact_e120, 'exact E120 continuations', flush=True)


if __name__ == '__main__':
    import sys
    {'build': build, 'report': report, 'verify': verify, 'diagnose': diagnose, 'finish': finish}[sys.argv[1]]()
