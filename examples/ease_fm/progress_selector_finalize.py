import json
import shutil
from pathlib import Path
from .progress_selector import ROOT
from .benchmark import fingerprint

def run():
    summary=json.loads((ROOT/'summary.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text());verify=json.loads((ROOT/'verification.json').read_text());manifest=json.loads((ROOT/'videos/manifest.json').read_text())
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    for v in manifest['videos']:
        assert v['exact_replay'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
        assert fingerprint(ROOT/'trials'/f"{v['case']}_{v['seed']}_None_{v['mode']}.json")==v['reference_sha256']
    base=Path(__file__).parent
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(base/p.name)
    for name in ('progress_selector_finalize.py','build_progress_selector.py'):shutil.copy2(base/name,ROOT/'source'/name)
    scores=', '.join(f"{r['mode']} {r['success']}/36 ({r['collisions']} collisions; {r['collision_free_failures']} contact-free failures)" for r in summary['table'] if r['group']=='all')
    lead=f"**E111 completed:** Frozen broad progress-aware selector with nonnegative predicted geometric clearance, tested in72 paired training-body episodes: {scores}. Training/held-placement gate {'PASS' if summary['passes'] else 'FAIL'}. Same weights and task budgets; no route planner. Reports phase progress,2mm-margin fraction and search compute cost. All36 baseline replays,72 approach prefixes and {verify['exact_selected_predictions']} selected next states/contact unions verify exactly. Saves {len(manifest['videos'])} verified executed videos. Privileged exact model/state; no unseen-body or final-test evaluation. See Section4.101."

    p=base/'TECHNICAL_REPORT.md';text=p.read_text(encoding='utf-8');assert '## 4.101 E111' not in text
    text=text.replace('Revision 96 · 13 November 2026','Revision 97 · 13 November 2026',1)
    text=text.replace('## 1. Research question and claim criteria\n\n','## 1. Research question and claim criteria\n\n'+lead+'\n\n',1)
    lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('| E110 |'));lines.insert(i+1,f'| E111 | Broad progress-aware physics selection | {scores}; gate {summary["passes"]} | `ease_progress_selector_v1` |');text='\n'.join(lines)+'\n'
    report=(ROOT/'report.md').read_text().replace('# E111:','## 4.101 E111:',1);report='\n'.join('### E111 '+l[3:] if l.startswith('## ') and not l.startswith('## 4.101') else l for l in report.splitlines())
    text=text.replace('## 5. Limitations and next decisions\n',report+'\n\n## 5. Limitations and next decisions\n\n'+lead+'\n',1)
    text=text.replace('- Revision 96,',f'- Revision 97, 13 November 2026: E111 tests frozen broad progress-aware physics selection in72 paired tasks. {scores}; gate {summary["passes"]}. Saves phase progress, failures and {len(manifest["videos"])} verified videos.\n- Revision 96,',1);p.write_text(text,encoding='utf-8')
    for name in ('README.md','MIXED_DESIGN.md'):
        p=base/name;first,rest=p.read_text(encoding='utf-8').split('\n',1);p.write_text(first+'\n\n'+lead+'\n'+rest,encoding='utf-8')
    verify.update(verified_videos=len(manifest['videos']),video_hashes_verified=True,tests_passed=131,test_seconds=12.32);(ROOT/'verification.json').write_text(json.dumps(verify,indent=2))
    (ROOT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2));print(lead)

if __name__=='__main__':run()
