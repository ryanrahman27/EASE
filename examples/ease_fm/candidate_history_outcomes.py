"""Persist the paired outcome transitions, without further fitting or selection."""
import json
import shutil
from pathlib import Path
from .candidate_history_repair import ROOT


def run():
    summary = json.loads((ROOT / 'summary.json').read_text())
    changes = []
    for pair in summary['paired']:
        seed = pair['seed']
        for change in pair['changes']:
            item = dict(seed=seed, **change)
            for method in ('reference', 'candidate'):
                trial = json.loads((ROOT / 'trials' / f"{change['case']}_{method}_{seed}_frozen_bias.json").read_text())
                collision = next((dict(step=i, phase=t['phase'], contacts=t['bad']) for i, t in enumerate(trial['trace']) if t['bad']), None)
                item[method] = dict(success=trial['success'], collision=trial['collision'], first_collision=collision)
            changes.append(item)
    (ROOT / 'paired_outcomes.json').write_text(json.dumps(changes, indent=2))
    lines = ['\n## Outcome interpretation', '',
             'The corrective run recovers four previous failures but loses seven previous successes. On fitting placements it recovers one failure and loses six successes (38/48 to 33/48). On held placements it recovers three failures and loses one success (12/24 to 14/24). These are 24 placements repeated across three learned checkpoints, not 72 independent tasks. All bodies are training bodies.', '',
             'The result does not support promoting this correction. Matching successful actions on recorded histories is insufficient to guarantee retention in closed-loop execution. The exact source of each regression remains to be isolated; lower aggregate prediction error alone is not evidence of better task control. The next diagnostic should compare the first action/history divergence on these lost-success pairs, then test a bounded learned residual against the frozen reference.', '',
             '| Seed | Case | Split | Change | Previous first collision | Corrected first collision |',
             '|---|---|---|---|---|---|']
    for item in changes:
        phase = lambda method: (item[method]['first_collision'] or {}).get('phase', 'none')
        lines.append(f"| {item['seed']} | {item['case']} | {item['group']} | {'recovered' if item['gain'] else 'lost'} | {phase('reference')} | {phase('candidate')} |")
    path = ROOT / 'report.md'
    path.write_text(path.read_text() + '\n'.join(lines) + '\n')
    shutil.copy2(__file__, ROOT / 'source' / Path(__file__).name)
    print(json.dumps(changes, indent=2))


if __name__ == '__main__':
    run()
