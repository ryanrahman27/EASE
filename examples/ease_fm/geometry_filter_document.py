"""Save E72 interpretation in the cumulative research record."""
import json
from pathlib import Path
import shutil
from .geometry_filter import ROOT


def run():
    result=json.loads((ROOT/'summary.json').read_text());b=result['pooled']['baseline'];g=result['pooled']['geometry']
    decision=('The prespecified development gate passes; freeze the filter for fresh-placement validation, retaining all casewise regressions.' if result['passes'] else 'The prespecified gate fails; retain the original controller and preserve this negative geometry-filter result.')
    lead=(f"**E72 completed:** A shared filter with privileged geometry changes learned success from {b['success']}/96 to {g['success']}/96 and collision cases from {b['collision']} to {g['collision']}. There are {result['gains']} recoveries and {result['losses']} regressions. {decision} This tests controller assistance with extra robot/scene information, not improved history-only adaptation. All 224 v2 trials are retained; 112 baseline replays are exact and 102 tests pass. See Section 4.62.")
    next_step=('Next validate this frozen filter on E71 placements, and subsequently compare against a shared action-scaling control to distinguish geometry-conditioned decisions from simply slowing motion.' if result['passes'] else 'Before any further filter redesign, inspect collision_diagnostics.json to distinguish sampled-geometry misses, fallback behavior and task-progress loss. Any stronger predictive controller would require separate accounting for dynamics/geometry access and an identical analytical baseline. Do not tune the current margin or scales on these failures.')
    path=ROOT/'report.md';path.write_text(path.read_text()+'\n### Interpretation and next decision\n\n'+decision+' '+next_step+'\n\nThe broader research status remains proof of working simulated cross-kinematic control with limited manipulation robustness. Neither this diagnostic nor the prior placement results establishes learned superiority over analytical identification, broad new-embodiment manipulation, or hardware performance. Publication readiness remains unresolved.\n')
    folder=Path(__file__).parent;p=folder/'TECHNICAL_REPORT.md';text=p.read_text();assert '## 4.62 E72:' not in text
    text=text.replace('Revision 57 · 27 October 2026','Revision 58 · 28 October 2026',1)
    text=text.replace('## 1. Research question and claim criteria\n','## 1. Research question and claim criteria\n\n'+lead+'\n',1)
    line=next(x for x in text.splitlines(True) if x.startswith('| E71 |'))
    text=text.replace(line,line+'| E72 | Shared privileged-geometry action filter | Complete: original budgets, paired frozen estimators; extra information disclosed | `ease_geometry_filter_v2` |\n',1)
    section=path.read_text().replace('# E72: shared geometry-based action filtering','## 4.62 E72: shared geometry-based action filtering',1)
    text=text.replace('## 5. Limitations and next decisions',section+'\n\n## 5. Limitations and next decisions',1)
    text=text.replace('## 5. Limitations and next decisions\n\n','## 5. Limitations and next decisions\n\n'+decision+' '+next_step+' E72 provides exact robot/scene geometry to the filter; do not describe its results as learned history-only adaptation.\n\n',1)
    text=text.replace('## Revision history\n','## Revision history\n\n- Revision 58, 28 October 2026: Completed E72 geometry-filter v2, 224 paired trials with unchanged physics, original phase budgets and frozen estimators. Preserved interrupted v1 after exact no-op action correction. Audited 112 exact baseline replays, source/input weights, filter work and unchanged physical margins; 102 tests pass. Inspected comparison figure. No training, native tuning or final-test evaluation; explicitly disclosed privileged geometry access.\n',1)
    p.write_text(text)
    for name in ('README.md','MIXED_DESIGN.md'):
        p=folder/name;first,rest=p.read_text().split('\n',1);p.write_text(first+'\n\n'+lead+'\n'+rest)
    for name in ('geometry_filter_report.py','geometry_filter_document.py'):shutil.copy2(folder/name,ROOT/'source'/name)


if __name__=='__main__':run()
