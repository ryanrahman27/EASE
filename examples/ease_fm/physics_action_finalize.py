import json
import shutil
from pathlib import Path
from .physics_action import ROOT
from .benchmark import fingerprint

def run():
    summary=json.loads((ROOT/'summary.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text());verify=json.loads((ROOT/'verification.json').read_text());manifest=json.loads((ROOT/'videos/manifest.json').read_text())
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    for v in manifest['videos']:
        assert v['exact_replay'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
        assert fingerprint(ROOT/'trials'/f"{v['case']}_{v['seed']}_None_{v['mode']}.json")==v['reference_sha256']
    base=Path(__file__).parent
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(base/p.name)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    scores=', '.join(f"{r['mode']} {r['success']}/36 ({r['collisions']} collisions)" for r in summary['table'] if r['group']=='all')
    lead=f"**E108 completed:** Fixed 40ms exact-physics safer-action selection tested in 72 paired training-body tasks: {scores}. Training/held-placement gate {'PASS' if summary['passes'] else 'FAIL'}. Frozen weights and unchanged task budgets; no route planner. All36 baselines,72 pre-retreat prefixes and {verify['exact_selected_predictions']} selected next-state/contact predictions verify exactly. Saves {len(manifest['videos'])} new verified full-task videos. Privileged physics access; no unseen-body or final-test evaluation. See Section 4.98."
    p=base/'TECHNICAL_REPORT.md';text=p.read_text(encoding='utf-8');assert '## 4.98 E108' not in text
    text=text.replace('Revision 93 · 13 November 2026','Revision 94 · 13 November 2026',1)
    text=text.replace('## 1. Research question and claim criteria\n\n','## 1. Research question and claim criteria\n\n'+lead+'\n\n',1)
    lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('| E107 |'));lines.insert(i+1,f'| E108 | Physics-based safer action selection | {scores}; gate {summary["passes"]} | `ease_physics_action_v1` |');text='\n'.join(lines)+'\n'
    report=(ROOT/'report.md').read_text().replace('# E108:','## 4.98 E108:',1);report='\n'.join('### E108 '+l[3:] if l.startswith('## ') and not l.startswith('## 4.98') else l for l in report.splitlines())
    text=text.replace('## 5. Limitations and next decisions\n',report+'\n\n## 5. Limitations and next decisions\n\n'+lead+'\n',1)
    text=text.replace('- Revision 93,',f'- Revision 94, 13 November 2026: E108 completes72 paired training-body tasks. {scores}; gate {summary["passes"]}. Saves all alternatives and {len(manifest["videos"])} verified videos; no transfer test.\n- Revision 93,',1);p.write_text(text,encoding='utf-8')
    for name in ('README.md','MIXED_DESIGN.md'):
        p=base/name;first,rest=p.read_text(encoding='utf-8').split('\n',1);p.write_text(first+'\n\n'+lead+'\n'+rest,encoding='utf-8')
    verify.update(verified_videos=len(manifest['videos']),video_hashes_verified=True);(ROOT/'verification.json').write_text(json.dumps(verify,indent=2))
    (ROOT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2));print(lead)

if __name__=='__main__':run()
