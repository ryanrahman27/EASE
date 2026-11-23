"""Archive E113 and link the independently produced progress reel."""
import json
import shutil
from pathlib import Path
from .release_selector import ROOT
from .benchmark import fingerprint

def run():
    s=json.loads((ROOT/'summary.json').read_text());p=json.loads((ROOT/'protocol.json').read_text());v=json.loads((ROOT/'verification.json').read_text());m=json.loads((ROOT/'videos/manifest.json').read_text())
    for path,h in {**p['inputs'],**p['checkpoints']}.items():assert fingerprint(path)==h
    for x in m['videos']:
        assert x['exact_replay'] and fingerprint(ROOT/'videos'/x['file'])==x['sha256']
        assert fingerprint(ROOT/'trials'/f"{x['case']}_{x['seed']}_None_{x['mode']}.json")==x['reference_sha256']
    base=Path(__file__).parent
    for name in ('release_selector.py','unseen_progress_selector.py','unseen_progress_rollout.py','progress_action_bank.py'):assert fingerprint(base/name)==fingerprint(ROOT/'source'/name)
    for name in ('release_selector_report.py','release_selector_videos.py','release_selector_finalize.py','build_release_videos.py','progress_demo.py'):shutil.copy2(base/name,ROOT/'source'/name)
    scores=', '.join(f"{('original selector' if r['mode']=='baseline' else 'release extension')} {r['success']}/24 ({r['collisions']} collisions)" for r in s['table'] if r['group']=='all')
    lead=f"**E113 completed:** Release-only extension on all24 six-joint E112 pairs: {scores}. Gate {'PASS' if s['passes'] else 'FAIL'}; {s['release_changed_actions']} release substitutions. All24 original-selector replays and24 pre-release prefixes exact. Reused development cohort; frozen40ms bank/progress/nonpenetration criteria, no tuning. {len(m['videos'])} verified videos. Separate60-second concept/progress reel covers results through E113. See Section4.103."
    v.update(verified_videos=len(m['videos']),video_hashes_verified=True);(ROOT/'verification.json').write_text(json.dumps(v,indent=2))
    report=(ROOT/'report.md').read_text()+f"\nFinal verification: {len(m['videos'])} full-result video replays and hashes match.\n\nProgress demo: ../ease_progress_demo_v1/progress_demo.mp4 (60 seconds; results through E113; illustrative graphics plus verified footage; final card includes this release experiment).\n"
    (ROOT/'report.md').write_text(report)
    target=base/'TECHNICAL_REPORT.md';t=target.read_text(encoding='utf-8');assert '## 4.103 E113' not in t
    t=t.replace('Revision 98 · 13 November 2026','Revision 99 · 14 November 2026',1).replace('## 1. Research question and claim criteria\n\n','## 1. Research question and claim criteria\n\n'+lead+'\n\n',1)
    lines=t.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('| E112 |'));lines.insert(i+1,f'| E113 | Six-joint release extension | {scores}; gate {s["passes"]} | `ease_release_selector_v1` |');t='\n'.join(lines)+'\n'
    section=report.replace('# E113:','## 4.103 E113:',1);section='\n'.join('### E113 '+l[3:] if l.startswith('## ') and not l.startswith('## 4.103') else l for l in section.splitlines())
    t=t.replace('## 5. Limitations and next decisions\n',section+'\n\n## 5. Limitations and next decisions\n\n'+lead+'\n',1).replace('- Revision 98,',f'- Revision 99, 14 November 2026: E113 release extension: {scores}; gate {s["passes"]}. Added60-second progress demo through E113.\n- Revision 98,',1);target.write_text(t,encoding='utf-8')
    for name in ('README.md','MIXED_DESIGN.md'):
        target=base/name;first,rest=target.read_text(encoding='utf-8').split('\n',1);target.write_text(first+'\n\n'+lead+'\n'+rest,encoding='utf-8')
    demo=Path('outputs/ease_progress_demo_v1');shutil.copy2(base/'progress_demo.py',demo/'progress_demo.py')
    for root in (ROOT,demo):(root/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(root)):fingerprint(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2))
    print(lead)

if __name__=='__main__':run()
