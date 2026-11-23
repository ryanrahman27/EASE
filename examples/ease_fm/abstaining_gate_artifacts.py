"""E126 grouped-validation audit, evaluation report, and diagnostic media."""
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from .abstaining_task_value_gate import (ROOT, DATA, COLLECT, OLD, SEEDS, PHASES, Gate, read, save,
    setup, load_gate, evaluate, probabilities, accepts, family, outcome_grid, fingerprint)


def build():
    s = Path(__file__).with_name('broad_task_value_gate_videos.py').read_text()
    s = s.replace('broad_task_value_gate import', 'abstaining_task_value_gate import').replace('E125', 'E126').replace('len(rows)==312', 'len(rows)==384')
    s = s.replace('Learned task-value gate / Retention', 'Abstaining gate / E120 fallback')
    s = s.replace('<h2>Placement coverage</h2><p>Exact held placements remain excluded from fitting; the new fitting data cover a broader spatial region.</p><img width="1000" src="../placement_coverage.png">', '')
    s = s.replace('paired first gain/loss overall and on unchanged held development placements', 'paired first gain/loss overall and on shifted held development placements')
    # Include every changed outcome versus E120, E124, and E125, not only the original targets.
    needle = '    selection=list(dict.fromkeys(selection));manifest='
    extra = "    for r in rows:\n        if not r['method'].startswith('candidate_'):continue\n        seed=int(r['method'].split('_')[-1])\n        for family in ('unbounded','previousgate','broadgate'):\n            baseline=f'{family}_{seed}'\n            if lookup[r['case'],baseline,'frozen_bias']['success'] != r['success']:\n                selection.extend([(r['case'],baseline,'frozen_bias'),(r['case'],r['method'],'frozen_bias')])\n"
    assert needle in s
    s = s.replace(needle, extra + needle).replace('Outcome-selected diagnostic examples.', 'Includes every changed outcome versus E120/E124/E125. Outcome-selected diagnostic examples.')
    Path(__file__).with_name('abstaining_gate_videos.py').write_text(s)


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
    selected, grid = read(ROOT / 'selected_rule.json'), read(ROOT / 'validation_grid.json')
    stringent = next(r for r in grid if r['rule'] == dict(min_advantage=.9, max_spread=.05))
    meta = read(DATA / 'gate_rows.json')
    index = {(r['case'], r['seed'], r['phase']): i for i, r in enumerate(meta)}
    oof = np.load(ROOT / 'oof_predictions.npz')
    diagnostic = []
    for r in stringent['episodes']:
        if r['success'] == r['baseline_success']:
            continue
        i = index[r['case'], r['seed'], r['phase']]
        prob = oof['probabilities'][:, i]
        diagnostic.append(r | dict(fold=int(oof['row_fold'][i]), model_probabilities=prob.tolist(), advantages=(prob[:, 1]-prob[:, 0]).tolist()))
    save(ROOT / 'high_agreement_diagnostics.json', diagnostic)
    lines = ['# E126: Fitting-placement validation of an abstaining gate', '',
             'E120 remains the immutable baseline and fallback. E126 selects full abstention: every nontrivial rule in the predeclared grid lost two E120 fitting-validation successes while gaining one. The selected fallback is a safety/retention result, not evidence of learned improvement.', '',
             '## Protocol and split isolation', '', read(ROOT / 'protocol.json')['folds'], '',
             'Reuse E125\'s1108 verified clean fitting pairs from1440 saved trajectories; no new training rollouts. Three MLPs (139-64-32-2, tanh; seeds126100..126102) per fold,1500 full-batch AdamW updates, paired final-success BCE. Each fold computes normalization from its training rows only. All phases and three frozen motion-estimator seeds of a placement family remain in the same fold. Four folds yield144 out-of-fold complete-episode outcomes from the exact saved on/off continuations.', '',
             'At a phase entry, require the minimum of three predicted off-minus-on advantages to exceed the threshold, and their range to remain within the spread limit. Otherwise E120 continues. At most one up-to8-step correction-off pulse, followed by E120 for the rest of the episode. No physics selector, collision flag, case/robot ID, or future outcome enters the gate. Ensemble agreement is not calibrated confidence.', '',
             'Twenty-five fixed threshold/spread combinations and full abstention were declared before fitting. Select zero-regression rules by most gains, then fewest switches, then larger advantage threshold and smaller spread. Pooled fitting-validation outcomes select this rule: their score is not an independent test estimate. Three final models are refit on all fitting rows after selection; the rule and checkpoints are hashed before retesting the original72 episodes. No held labels select the rule, threshold, seeds, or checkpoints.', '',
             'The source-family grouping follows transitive ancestry, including earlier shifted versions. A direct-parent draft was corrected before any E126 training. Validation outcome reconstruction omits starts that already collided, because those episodes cannot recover collision-free success; runtime receives no collision flag. Full retests include all phases.', '',
             'All bodies are existing training bodies. Held placements are reused development placements, not an untouched final benchmark. This experiment preserves E125\'s broader fitting support.', '',
             '## Fitting-fold rule selection', '', '| Minimum advantage | Maximum spread | Success /144 | Gains | Losses | Switches |', '|---:|---:|---:|---:|---:|---:|']
    for r in grid:
        lines.append(f"| {r['rule']['min_advantage']} | {r['rule']['max_spread']} | {r['success']} | {r['gains']} | {r['losses']} | {r['switches']} |")
    lines += ['', 'Even the rule requiring every advantage >0.90 and spread <=0.05 makes these outcome-changing decisions on excluded fitting families:', '', '| Case / estimator seed | Fold | Phase | Three predicted advantages | Observed change |', '|---|---:|---|---|---|']
    for r in diagnostic:
        margins=', '.join(f'{a:.4f}' for a in r['advantages'])
        lines.append(f"| {r['case']}/{r['seed']} | {r['fold']} | {r['phase']} | {margins} | {'gain' if r['success'] else 'loss'} |")
    lines += ['', f"Selected rule: `{selected['rule']}`. Validation successes {selected['success']}/144 versus E120 {selected['baseline_success']}/144; {selected['gains']} gains, {selected['losses']} losses, {selected['switches']} switches.", '',
              '## Complete-episode retest', '', 'reference=E118; unbounded=E120 (bounded correction, no gate); previousgate=E124; broadgate=E125; candidate=E126.', '',
              '| Method | Split | DOF | Success | Collision episodes |', '|---|---|---|---:|---:|']
    for r in aggregate:
        lines.append(f"| {r['method']} | {r['split']} | {r['dof']} | {r['success']}/{r['n']} | {r['collisions']} |")
    lines += ['', f'Retained {retained}/6 original E120 gains; recovered {recovered}/4 regressions. No E120 success lost: {no_e120_losses}. Preserve all prior gate successes and recover all regressions: {strict}.', '',
              'Full abstention deliberately gives up the nine fitting rescues of E125 while restoring its two held losses. Matching E120 does not satisfy the stronger objective of preserving the learned gate\'s rescues as well. E120 is not replaced.', '',
              '## Interpretation and next test', '',
              'Strong agreement did not reject the harmful decisions on held fitting families. Thus three-model agreement and a larger sigmoid margin are insufficient uncertainty controls here. The next candidate should test explicit support in measured-history features and abstain outside that support, chosen with the same ancestry-grouped fitting validation. Do not choose an exception for particular held cases or adjust thresholds using their outcomes.', '',
              '## Every paired change', '', '```json', json.dumps(changes, indent=2), '```', '']
    (ROOT / 'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (method, label) in enumerate((('unbounded', 'E120'), ('previousgate', 'E124 gate'), ('broadgate', 'E125 broader gate'), ('candidate', 'E126 abstaining'))):
        rr = [next(r for r in aggregate if r['method'] == method and r['split'] == split and r['dof'] == 'all') for split in ('fitting', 'held')]
        x = np.arange(2) + (i - 1.5)*.20
        ax.bar(x, [100*r['success']/r['n'] for r in rr], .18, label=label)
        for xx, r in zip(x, rr):
            ax.text(xx, 100*r['success']/r['n']+2, f"{r['success']}/{r['n']}", ha='center')
    ax.set_xticks([0, 1], ['Original fitting placements', 'Held development placements'])
    ax.set_ylim(0, 125)
    ax.set_ylabel('Task success (%)')
    ax.legend(ncol=4, loc='upper center')
    fig.tight_layout()
    fig.savefig(ROOT / 'comparison.png', dpi=160)
    plt.close(fig)
    path = Path(__file__).with_name('TECHNICAL_REPORT.md')
    text = path.read_text(encoding='utf-8')
    marker = '## E126: ancestry-grouped validation and abstention'
    assert marker not in text
    text = text.replace('Revision 113 · 14 November 2026', 'Revision 114 · 15 November 2026', 1)
    text += f'\n{marker}\n\nFour ancestral-placement folds, three gate models per fold,1108 existing clean fitting pairs. Every nontrivial fixed rule gains one but loses two E120 validation successes; full abstention selected before held retest. E126 matches E12052/72 (37/48 fitting;15/24 held), retains{retained}/6 original gains and recovers{recovered}/4 regressions. All nine E125 fitting rescues are relinquished. E120 remains baseline; no useful learned intervention rule validated. No held optimization or unseen bodies. Full report: `outputs/ease_abstaining_task_value_gate_v1/report.md`.\n'
    path.write_text(text, encoding='utf-8')
    shutil.copy2(__file__, ROOT / 'source' / Path(__file__).name)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('aggregate', 'paired_changes', 'targeted_pairs')}, indent=2), flush=True)


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
    scores = outcome_grid(oof['probabilities'], meta, tasks, read(ROOT / 'protocol.json')['grid'])
    assert scores == read(ROOT / 'validation_grid.json')
    selected = max([s for s in scores if s['losses'] == 0], key=lambda s: (s['gains'], -s['switches'], s['rule']['min_advantage'], -s['rule']['max_spread']))
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
            assert bool(accepts(prob, rule)) == t['selected']
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
    save(ROOT / 'verification.json', dict(ancestral_families_disjoint=True, fold_normalization_training_only=True, held_training_rows=0, fitting_rows=len(meta),
         oof_predictions_exact=True, validation_selection_reproduced=True, labels_verified=True, decisions_checked=decisions, action_steps_checked=checked,
         fitting_continuations_exact=exact_fit, e120_continuations_exact=exact_e120, switches=switches, held_exact_replays=replays, frozen_weights_and_selection_unchanged=True))
    print('Verified', checked, 'actions;', exact_e120, 'exact E120 continuations', flush=True)


def finish():
    v, s = read(ROOT / 'verification.json'), read(ROOT / 'summary.json')
    manifest = read(ROOT / 'videos' / 'manifest.json')
    protocol = read(ROOT / 'protocol.json')
    for path, h in {**protocol['inputs'], **protocol['checkpoints']}.items():
        assert fingerprint(path) == h
    for path, h in read(ROOT / 'frozen_selection.json').items():
        assert fingerprint(ROOT / path) == h
    for p in (ROOT / 'source').glob('*.py'):
        assert fingerprint(p) == fingerprint(Path(__file__).with_name(p.name))
    for video in manifest['videos']:
        assert fingerprint(ROOT / 'videos' / video['file']) == video['sha256']
    save(ROOT / 'status.json', dict(status='complete', new_training_rollouts=0, reused_collection_episodes=1440, fitting_pairs=1108, grouped_folds=4, new_evaluation_episodes=72,
         no_e120_losses=s['no_e120_losses'], strict_retention_passed=s['strict_retention_passed'], videos=len(manifest['videos']), held_training_rows=0, unseen_evaluated=False))
    report_path = ROOT / 'report.md'
    report_path.write_text(report_path.read_text()+f"\n## Verification and saved media\n\nAll four fold predictions and training-only normalizations reproduced; ancestral source families disjoint;1108 fitting labels checked; zero held training rows. {v['decisions_checked']} gate decisions and{v['action_steps_checked']} action steps verified. {v['fitting_continuations_exact']} exact fitting continuations and{v['e120_continuations_exact']} exact E120 continuations; two held episodes exactly replayed. Selection/checkpoints and E120 inputs unchanged. Saved{len(manifest['videos'])} trace-verified success/failure videos. Data collection is referenced by immutable hashes instead of duplicating the large E125 archive.\n")
    p = ROOT / 'videos' / 'index.html'
    p.write_text(p.read_text().replace('</h1>', '</h1><p>E126: fitting-family validation selected full abstention. Candidate executes E120 exactly. reference=E118; unbounded=E120; previousgate=E124; broadgate=E125. This preserves E120 gains but relinquishes the learned gates\' fitting rescues.</p>', 1))
    hashes = read(ROOT / 'artifact_hashes.json')
    for name in ('report.md', 'status.json', 'videos/index.html', 'verification.json'):
        hashes[name] = fingerprint(ROOT / name)
    assert set(hashes) == {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'}
    save(ROOT / 'artifact_hashes.json', hashes)
    print(json.dumps(read(ROOT / 'status.json'), indent=2), flush=True)


if __name__ == '__main__':
    import sys
    {'build': build, 'report': report, 'verify': verify, 'finish': finish}[sys.argv[1]]()
