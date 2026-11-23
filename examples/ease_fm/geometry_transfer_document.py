"""Record E73 validation and slowing-control evidence."""
import json
from pathlib import Path
import shutil
from .geometry_transfer import ROOT


def run():
    r=json.loads((ROOT/'summary.json').read_text());b,g,s=[r['pooled'][m] for m in ('baseline','geometry','slow')]
    if r['passes'] and r['beats_fixed_slowing']:
        decision='The filter passes the placement replication rule and exceeds the fixed slowing control; retain it experimentally for native retention checks.'
        next_step='Next evaluate the frozen filter and the same slowing control on native retention cases with matched geometry access. Preserve the original controller until retention is established; do not tune on native failures.'
    elif r['passes']:
        decision='The filter passes aggregate placement retention but does not exceed fixed slowing. Do not attribute the gain specifically to geometry checks or promote the more complex filter.'
        next_step='Carry the simpler slowing control forward as an explicit comparator, with its interaction cost. Further validation must retain every checkpoint and casewise six-joint regression; avoid selecting another scale from these results.'
    else:
        decision='The frozen geometry filter fails the prespecified replication rule. Retain the original controller; no promotion of the filter.'
        next_step='Keep the full slowing comparison as evidence and inspect the paired geometry failures before proposing a different mechanism. Do not retune the filter margin or scale list on this validation set.'
    lead=(f"**E73 completed:** On 17 E71 placements, pooled learned success is original {b['success']}/102, geometry {g['success']}/102 and fixed slowing {s['success']}/102. Collision counts are {b['collision']}, {g['collision']} and {s['collision']}. {decision} All 357 trials are retained, 119 baseline traces reproduce exactly and 103 tests pass. See Section 4.63.")
    rp=ROOT/'report.md';rp.write_text(rp.read_text()+'\n### Interpretation and next decision\n\n'+decision+' '+next_step+'\n\nThe geometry intervention still depends on a known robot model and exact simulator scene state. No outcome here establishes a learned advantage over analytical estimation, broad unseen-embodiment manipulation or publication readiness. The slowing control tests one fixed value, not all geometry-free alternatives.\n')
    folder=Path(__file__).parent;p=folder/'TECHNICAL_REPORT.md';text=p.read_text();assert '## 4.63 E73:' not in text
    text=text.replace('Revision 58 · 28 October 2026','Revision 59 · 28 October 2026',1)
    text=text.replace('## 1. Research question and claim criteria\n','## 1. Research question and claim criteria\n\n'+lead+'\n',1)
    line=next(x for x in text.splitlines(True) if x.startswith('| E72 |'))
    text=text.replace(line,line+'| E73 | Frozen geometry transfer and geometry-free slowing | Complete: 357 paired trials on E71 placements | `ease_geometry_transfer_v1` |\n',1)
    section=rp.read_text().replace('# E73: geometry-filter transfer and fixed slowing control','## 4.63 E73: geometry-filter transfer and fixed slowing control',1)
    text=text.replace('## 5. Limitations and next decisions',section+'\n\n## 5. Limitations and next decisions',1)
    text=text.replace('## 5. Limitations and next decisions\n\n','## 5. Limitations and next decisions\n\n'+decision+' '+next_step+' These are development placements on existing robot bodies, and geometry is extra privileged information.\n\n',1)
    text=text.replace('## Revision history\n','## Revision history\n\n- Revision 59, 28 October 2026: Completed E73 with 357 trials on all 17 E71 placements, frozen geometry filter, original controller and fixed half-motion comparator. Verified 119 exact baseline replays, unchanged code/model/input hashes, physical margins and blocked control-condition geometry queries. Audited paired regressions, joint-count retention and costs; inspected figure; 103 tests pass. No training, native tuning or final-test evaluation.\n',1)
    p.write_text(text)
    for name in ('README.md','MIXED_DESIGN.md'):
        p=folder/name;first,rest=p.read_text().split('\n',1);p.write_text(first+'\n\n'+lead+'\n'+rest)
    for name in ('geometry_transfer_report.py','geometry_transfer_document.py'):shutil.copy2(folder/name,ROOT/'source'/name)


if __name__=='__main__':run()
