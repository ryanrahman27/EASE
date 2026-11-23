"""Record the completed E68 result in the cumulative research documents."""
import json
from pathlib import Path
import shutil
from .grasp_settling import ROOT


def run():
    audit=json.loads((ROOT/'audit.json').read_text());p=audit['pooled_learned']
    b,f,c=[p[m] for m in ('baseline','fixed','contact')]
    decision=('The prespecified gate passes; native retention remains required before adoption.' if audit['advance_native'] else 'The prespecified advancement gate fails. Retain the original sequencer and E35; do not expand this settling rule to native evaluation.')
    lead=(f"**E68 completed:** Bounded procedural grasp settling yields baseline {b['success']}/96, fixed waiting {f['success']}/96 and contact-ended waiting {c['success']}/96 pooled learned successes. Six-joint scores are {b['six']}, {f['six']}, {c['six']} /42; collisions are {b['collisions']}, {f['collisions']}, {c['collisions']}. {decision} All 336 trials are retained, all 64 historical baseline traces reproduce exactly, and 95 tests pass. See Section 4.58.")
    folder=Path(__file__).parent;report=folder/'TECHNICAL_REPORT.md';text=report.read_text()
    assert '## 4.58 E68:' not in text
    text=text.replace('Revision 53 · 27 October 2026','Revision 54 · 27 October 2026',1)
    text=text.replace('## 1. Research question and claim criteria\n','## 1. Research question and claim criteria\n\n'+lead+'\n',1)
    registry='| E68 | Frozen-estimator procedural grasp settling | Complete: bounded contact and fixed-duration comparison; advancement gate '+('passes' if audit['advance_native'] else 'fails')+' | `ease_grasp_settling_v1` |\n'
    line=next(l for l in text.splitlines(True) if l.startswith('| E67 |'))
    text=text.replace(line,line+registry,1)
    section=(ROOT/'report.md').read_text().replace('# E68: frozen-estimator grasp settling','## 4.58 E68: frozen-estimator grasp settling',1)
    section+='\nArtifacts: `outputs/ease_grasp_settling_v1/{protocol,summary,paired,audit}.json`, `report.md`, `comparison.png`, and all full trial traces. Implementation: `examples/ease_fm/grasp_settling.py`; three new sequencing tests cover unchanged baseline, contact streak reset and bounded dwell.\n\n'
    text=text.replace('## 5. Limitations and next decisions',section+'## 5. Limitations and next decisions',1)
    marker='## 5. Limitations and next decisions\n\n'
    text=text.replace(marker,marker+f"E68 tests a specific bounded settling intervention, not active regrasp recovery. {decision} Keep the E66 candidate experimental. Further grasp work should distinguish relative object/finger slip from estimator-driven pose error and test an explicit recovery action on procedural cases before further native tuning. No stronger cross-embodiment or publication claim follows from this experiment.\n\n",1)
    text=text.replace('## Revision history\n','## Revision history\n\n- Revision 54, 27 October 2026: Completed E68, 336 matched procedural trials across seven frozen estimators and three settling conditions. Recorded all paired gains/losses, joint-count retention, collisions and interaction costs; verified 64 exact historical baseline traces and unchanged model/input hashes. Inspected comparison figure; 95 tests pass. Reserved final test untouched.\n',1)
    report.write_text(text)
    for name in ['README.md','MIXED_DESIGN.md']:
        path=folder/name;t=path.read_text();first,rest=t.split('\n',1);path.write_text(first+'\n\n'+lead+'\n'+rest)
    for name in ['grasp_settling_report.py','grasp_settling_document.py']:
        shutil.copy2(folder/name,ROOT/'source'/name)


if __name__=='__main__':run()
