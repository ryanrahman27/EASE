import json
import shutil
from pathlib import Path
from .physics_prediction_audit import ROOT
from .benchmark import fingerprint

def run():
    spec=json.loads((ROOT/'protocol.json').read_text());summary=json.loads((ROOT/'summary.json').read_text());verify=json.loads((ROOT/'verification.json').read_text())
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    videos=json.loads((ROOT/'videos/manifest.json').read_text())['videos']
    for v in videos:assert v['exact_replay'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    primary=[r for r in summary['table'] if r['group']=='all' and r['clean_only'] and r['horizon_ms']==40]
    score='; '.join(f"{r['method']}: TP {r['TP']}, FP {r['FP']}, FN {r['FN']}, TN {r['TN']}" for r in primary)
    lead=f"**E107 completed:** Unchanged training-body baseline audit: 36 exact task replays and {verify['exact_40ms_states']} exact 40ms predicted states/contact unions. Clean-start 40ms counts: {score}. Reports held-placement split, longer held-target horizons, latency and all errors. Exact physical-model/state access is privileged; this verifies prediction, not collision avoidance or learned-dynamics transfer. No controller change or final test. See Section 4.97."
    base=Path(__file__).parent;p=base/'TECHNICAL_REPORT.md';text=p.read_text(encoding='utf-8');assert '## 4.97 E107' not in text
    text=text.replace('Revision 92 · 13 November 2026','Revision 93 · 13 November 2026',1);text=text.replace('## 1. Research question and claim criteria\n\n','## 1. Research question and claim criteria\n\n'+lead+'\n\n',1)
    lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('| E106 |'));lines.insert(i+1,f'| E107 | Training-body short physics prediction | {score}; no control intervention | `ease_physics_prediction_v1` |');text='\n'.join(lines)+'\n'
    report=(ROOT/'report.md').read_text().replace('# E107:','## 4.97 E107:',1);report='\n'.join('### E107 '+l[3:] if l.startswith('## ') and not l.startswith('## 4.97') else l for l in report.splitlines())
    text=text.replace('## 5. Limitations and next decisions\n',report+'\n\n## 5. Limitations and next decisions\n\n'+lead+'\n',1)
    text=text.replace('- Revision 92,',f'- Revision 93, 13 November 2026: E107 completes 36 unchanged training-body replays. {score}. Saves all prediction traces, latency, error cases and verified reused baseline videos. No controller intervention.\n- Revision 92,',1);p.write_text(text,encoding='utf-8')
    for name in ('README.md','MIXED_DESIGN.md'):
        p=base/name;first,rest=p.read_text(encoding='utf-8').split('\n',1);p.write_text(first+'\n\n'+lead+'\n'+rest,encoding='utf-8')
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(base/p.name)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    (ROOT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2));print(lead)

if __name__=='__main__':run()
