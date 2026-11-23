import json
from pathlib import Path
from .retreat_contact_audit import ROOT
from .benchmark import fingerprint

def run():
    spec=json.loads((ROOT/'protocol.json').read_text())
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    videos=json.loads((ROOT/'videos/manifest.json').read_text())['videos']
    for v in videos:assert fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    v=json.loads((ROOT/'verification.json').read_text())
    lead=f"**E106 completed:** Six exact E105 diagnostic replays save {v['contact_samples']} contact samples. The fresh-body regression contacts the table 3.150s into retreat, after the ramp ends, despite a preceding static filter decision marked feasible at its 2mm clearance cap. The successful baseline reaches a lower tool height, so tool height alone does not explain the failure. No causal mechanism isolated and no controller change. Six original verified videos retained. See Section 4.96."
    base=Path(__file__).parent;p=base/'TECHNICAL_REPORT.md';text=p.read_text(encoding='utf-8');assert '## 4.96 E106' not in text
    text=text.replace('Revision 91 · 6 November 2026','Revision 92 · 13 November 2026',1)
    text=text.replace('## 1. Research question and claim criteria\n\n','## 1. Research question and claim criteria\n\n'+lead+'\n\n',1)
    lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('| E105 |'));lines.insert(i+1,'| E106 | Opposite retreat contact diagnosis | Six exact replays; static feasibility fails to guarantee dynamic clearance; no policy change | `ease_retreat_contact_audit_v1` |');text='\n'.join(lines)+'\n'
    report=(ROOT/'report.md').read_text().replace('# E106:','## 4.96 E106:',1)
    report='\n'.join('### E106 '+l[3:] if l.startswith('## ') and not l.startswith('## 4.96') else l for l in report.splitlines())
    text=text.replace('## 5. Limitations and next decisions\n',report+'\n\n## 5. Limitations and next decisions\n\n'+lead+'\n',1)
    text=text.replace('- Revision 91,','- Revision 92, 13 November 2026: E106 observes six unchanged E105 replays at2ms, diagnoses static-filter/dynamic-contact mismatch, preserves all failures and six verified original videos. No control change.\n- Revision 91,',1);p.write_text(text,encoding='utf-8')
    for name in ('README.md','MIXED_DESIGN.md'):
        p=base/name;first,rest=p.read_text(encoding='utf-8').split('\n',1);p.write_text(first+'\n\n'+lead+'\n'+rest,encoding='utf-8')
    (ROOT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2));print(lead)

if __name__=='__main__':run()
