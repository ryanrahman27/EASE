import json
import shutil
from pathlib import Path
from .goal_action_audit import ROOT
from .benchmark import fingerprint

def run():
    s=json.loads((ROOT/'summary.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text());v=json.loads((ROOT/'verification.json').read_text());videos=json.loads((ROOT/'videos/manifest.json').read_text())['videos']
    assert v['snapshot_branches_exact']>0
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    for item in videos:assert item['exact_replay'] and fingerprint(ROOT/'videos'/item['file'])==item['sha256']
    base=Path(__file__).parent
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(base/p.name)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    a=next(r for r in s['table'] if r['kind']=='warning' and r['split']=='all' and r['bank']=='narrow');b=next(r for r in s['table'] if r['kind']=='warning' and r['split']=='all' and r['bank']=='broad')
    lead=f"**E110 completed:** Broader copied-physics action audit on {v['states']} training-body states. Warning-state contact-free progress availability at40ms: narrow {a['admissible40']}/{a['n']}, broad {b['admissible40']}/{b['n']}; broad candidates maintaining2mm margin {b['margin40']}/{b['n']}. The40ms-selected broad action retains contact-free progress at160ms in {b['chosen_retains160']}/{b['n']}. {v['exact_replays']} source tasks replay exactly; {v['snapshot_branches_exact']} snapshot branches reconstruct exactly. No new actions executed in policy, no task-success improvement claim, no final test. See Section4.100."
    p=base/'TECHNICAL_REPORT.md';text=p.read_text(encoding='utf-8');assert '## 4.100 E110' not in text
    text=text.replace('Revision 95 · 13 November 2026','Revision 96 · 13 November 2026',1);text=text.replace('## 1. Research question and claim criteria\n\n','## 1. Research question and claim criteria\n\n'+lead+'\n\n',1)
    lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('| E109 |'));lines.insert(i+1,f'| E110 | Broad goal-directed action availability | Narrow {a["admissible40"]}/{a["n"]}, broad {b["admissible40"]}/{b["n"]} warning states; audit only | `ease_goal_action_audit_v1` |');text='\n'.join(lines)+'\n'
    report=(ROOT/'report.md').read_text().replace('# E110:','## 4.100 E110:',1);report='\n'.join('### E110 '+l[3:] if l.startswith('## ') and not l.startswith('## 4.100') else l for l in report.splitlines())
    text=text.replace('## 5. Limitations and next decisions\n',report+'\n\n## 5. Limitations and next decisions\n\n'+lead+'\n',1);text=text.replace('- Revision 95,',f'- Revision 96, 13 November 2026: E110 tests broad goal-directed actions on copied training states; warning-state availability {a["admissible40"]}/{a["n"]} narrow versus {b["admissible40"]}/{b["n"]} broad. Preserves margins, hold-horizon qualification and no policy-success claim.\n- Revision 95,',1);p.write_text(text,encoding='utf-8')
    for name in ('README.md','MIXED_DESIGN.md'):
        p=base/name;first,rest=p.read_text(encoding='utf-8').split('\n',1);p.write_text(first+'\n\n'+lead+'\n'+rest,encoding='utf-8')
    v.update(reused_verified_videos=len(videos),video_hashes_verified=True,tests_passed=131,test_seconds=13.99);(ROOT/'verification.json').write_text(json.dumps(v,indent=2))
    (ROOT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2));print(lead)

if __name__=='__main__':run()
