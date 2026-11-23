"""Record E71 in the cumulative technical report."""
import json
from pathlib import Path
import shutil
from .phase_fresh import ROOT


def run():
    r=json.loads((ROOT/'summary.json').read_text());b=r['pooled']['baseline'];e=r['pooled']['extended'];audit=json.loads((ROOT/'audit.json').read_text())
    decision=('The aggregate replication rule passes; keep extended timing experimental pending native retention and casewise checks.' if r['passes'] else 'The replication rule fails; retain the original phase budgets. Do not promote extended timing.')
    lead=(f"**E71 completed:** Fresh-placement validation yields baseline {b['success']}/{b['n']} and extended {e['success']}/{e['n']} learned successes, with {r['paired_gains']} recoveries and {r['paired_losses']} regressions. Collisions change from {b['collisions']} to {e['collisions']}. {audit['accepted']}/{audit['candidates']} placements pass structural screening on the same six bodies; this is not new-embodiment transfer. {decision} All {audit['trials']} trials and screening rejections are retained; 99 tests pass. See Section 4.61.")
    next_step=('Next check frozen extended timing against original timing on native retention cases, keeping every estimator and recording casewise six-joint regressions and costs. Do not optimize the budget on those native outcomes.' if r['passes'] else 'Stop tuning waiting times on these placements. The next distinct experiment should change the path before contact using a shared clearance-aware controller, with identical geometry access for learned and analytical estimators and explicit disclosure of that extra information.')
    rp=ROOT/'report.md';rp.write_text(rp.read_text()+'\n### Decision and next step\n\n'+decision+' '+next_step+' The reserved final test remains untouched.\n')
    folder=Path(__file__).parent;p=folder/'TECHNICAL_REPORT.md';text=p.read_text();assert '## 4.61 E71:' not in text
    text=text.replace('Revision 56 · 27 October 2026','Revision 57 · 27 October 2026',1)
    text=text.replace('## 1. Research question and claim criteria\n','## 1. Research question and claim criteria\n\n'+lead+'\n',1)
    line=next(x for x in text.splitlines(True) if x.startswith('| E70 |'))
    text=text.replace(line,line+'| E71 | Frozen extended timing on fresh procedural placements | Complete: prospective placement retention; all screening outcomes retained | `ease_phase_fresh_v1` |\n',1)
    section=rp.read_text().replace('# E71: fresh-placement validation of extended phase budgets','## 4.61 E71: fresh-placement validation of extended phase budgets',1)
    text=text.replace('## 5. Limitations and next decisions',section+'\n\n## 5. Limitations and next decisions',1)
    text=text.replace('## 5. Limitations and next decisions\n\n','## 5. Limitations and next decisions\n\n'+decision+' '+next_step+' E71 changes placements, homes and probe seeds on existing bodies; do not describe it as new-embodiment validation.\n\n',1)
    text=text.replace('## Revision history\n','## Revision history\n\n- Revision 57, 27 October 2026: Completed E71 prospective placement validation with frozen E70 timing and checkpoints. Retained all screening rejections, paired outcomes, joint-count/family breakdowns and costs. Verified source/model/input hashes, trial coverage and trace lengths; inspected figure; 99 tests pass. No training, native tuning or final-test evaluation.\n',1)
    p.write_text(text)
    for name in ('README.md','MIXED_DESIGN.md'):
        p=folder/name;first,rest=p.read_text().split('\n',1);p.write_text(first+'\n\n'+lead+'\n'+rest)
    for name in ('phase_fresh_report.py','phase_fresh_document.py'):shutil.copy2(folder/name,ROOT/'source'/name)


if __name__=='__main__':run()
