"""Describe E125 fitting outcomes without changing gate training or selection."""
import json
import shutil
from pathlib import Path

from .broad_task_value_gate import ROOT, COLLECT


def run():
    rows = json.loads((ROOT / 'gate_rows.json').read_text())
    records = json.loads((COLLECT / 'records.json').read_text())
    tasks = json.loads((ROOT / 'cases.json').read_text())
    groups = []
    for source in ('original', 'added'):
        for dof in (6, 7):
            selected = [r for r in rows if (r['case'] >= 24) == (source == 'added')
                        and len(tasks[r['case']]['home']) == dof]
            on = [r for r in records if r['off_phase'] is None
                  and (r['case'] >= 24) == (source == 'added') and r['dof'] == dof]
            groups.append(dict(source=source, dof=dof, clean_pairs=len(selected),
                               on_episodes=len(on), on_success=sum(r['success'] for r in on),
                               off_helps=sum(r['labels'][1] > r['labels'][0] for r in selected),
                               off_hurts=sum(r['labels'][1] < r['labels'][0] for r in selected),
                               unchanged=sum(r['labels'][1] == r['labels'][0] for r in selected)))
    assert all(not tasks[r['case']]['validation'] for r in rows)
    result = dict(groups=groups, note='Descriptive fitting outcomes only; phase pairs are dependent. No held labels or changes to training.')
    (ROOT / 'label_audit.json').write_text(json.dumps(result, indent=2))
    shutil.copy2(__file__, ROOT / 'source' / Path(__file__).name)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
