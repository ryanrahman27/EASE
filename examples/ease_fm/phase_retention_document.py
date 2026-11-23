"""Append E70 evidence and its retention decision to the research record."""
import json
from pathlib import Path
import shutil
from .phase_retention import ROOT


def run():
    result=json.loads((ROOT/'summary.json').read_text());p=result['pooled'];passed=[m for m,d in result['decisions'].items() if d['passes']]
    decision=('Retain these as development candidates for broader validation: '+', '.join(passed)+'. Keep the baseline until that validation passes.' if passed else 'Neither intervention passes retention. Keep the original sequencer and frozen E35 reference.')
    lead=(f"**E70 completed:** Learned procedural success is baseline {p['baseline']['success']}/96, extended budget {p['extended']['success']}/96 and gated transitions {p['gated']['success']}/96. Collision counts are {p['baseline']['collisions']}, {p['extended']['collisions']} and {p['gated']['collisions']}; gated control aborts {p['gated']['aborts']} trials. {decision} All 336 trials are retained; 112 baseline and 112 gate-prefix checks pass, with 99 tests passing. See Section 4.60.")
    next_step=('First freeze the extended-budget candidate and validate it on fresh procedural placements, including casewise six-joint regressions and cost. ' if passed else '')
    interpretation=('\n### Interpretation and next decision\n\n'+decision+'\n\n'+next_step+
        'Stopping a trial at a failed transition changes exposure to later contacts, so lower collision counts cannot be presented as successful collision avoidance. Extended budgets change interaction duration and can alter contact outcomes even without a gate. These trials test timing and stopping; they do not test geometry-aware motion.\n\n'
        'A separate subsequent intervention is a shared path-clearance controller on procedural cases, with an explicit accounting of any geometric information it receives. It should change motion before contact, preserve task success and six-joint retention, and be compared with the original estimator/controller and an analytical estimator under the same geometry access. Avoid another threshold or waiting-time sweep on these observed cases. Native transfer and the reserved final test remain deferred.\n')
    report=ROOT/'report.md';report.write_text(report.read_text()+interpretation)
    folder=Path(__file__).parent;path=folder/'TECHNICAL_REPORT.md';text=path.read_text();assert '## 4.60 E70:' not in text
    text=text.replace('Revision 55 · 27 October 2026','Revision 56 · 27 October 2026',1)
    text=text.replace('## 1. Research question and claim criteria\n','## 1. Research question and claim criteria\n\n'+lead+'\n',1)
    line=next(x for x in text.splitlines(True) if x.startswith('| E69 |'))
    text=text.replace(line,line+'| E70 | Bounded phase budgets and transition retention | Complete: baseline/extended/gated matched comparison | `ease_phase_retention_v1` |\n',1)
    section=report.read_text().replace('# E70: bounded phase-transition retention','## 4.60 E70: bounded phase-transition retention',1)
    text=text.replace('## 5. Limitations and next decisions',section+'\n\n## 5. Limitations and next decisions',1)
    text=text.replace('## 5. Limitations and next decisions\n\n','## 5. Limitations and next decisions\n\n'+decision+' E70 isolates timing and transition stopping, not collision-clearance motion. Next investigate a shared geometry-aware path intervention with matched information access and explicit six-joint/task-success retention. Do not interpret fewer collisions from aborting as recovered tasks.\n\n',1)
    text=text.replace('## Revision history\n','## Revision history\n\n- Revision 56, 27 October 2026: Completed E70, 336 frozen-estimator trials comparing original timing, extended phase budgets and matched gated transitions. Recorded paired gains/losses, per-checkpoint six-joint retention, aborts, collisions and costs. Verified 112 exact baseline replays, 112 exact gated prefixes, unchanged hashes and 99 passing tests; inspected comparison figure. No training, native or reserved final-test evaluation.\n',1)
    path.write_text(text)
    for name in ('README.md','MIXED_DESIGN.md'):
        pth=folder/name;first,rest=pth.read_text().split('\n',1);pth.write_text(first+'\n\n'+lead+'\n'+rest)
    for name in ('phase_retention_report.py','phase_retention_document.py'):shutil.copy2(folder/name,ROOT/'source'/name)


if __name__=='__main__':run()
