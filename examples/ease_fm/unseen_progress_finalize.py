"""Verify E112 artifacts and append the full result to the research record."""
import json
import shutil
from pathlib import Path
from .unseen_progress import ROOT
from .benchmark import fingerprint

def run():
    summary=json.loads((ROOT/'summary.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text());verify=json.loads((ROOT/'verification.json').read_text());manifest=json.loads((ROOT/'videos/manifest.json').read_text())
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    for v in manifest['videos']:
        assert v['exact_replay'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
        assert fingerprint(ROOT/'trials'/f"{v['case']}_{v['seed']}_None_{v['mode']}.json")==v['reference_sha256']
    base=Path(__file__).parent
    for name in ('unseen_progress.py','unseen_progress_selector.py','unseen_progress_rollout.py','progress_selector.py','progress_action_bank.py'):
        assert fingerprint(ROOT/'source'/name)==fingerprint(base/name)
    for name in ('unseen_progress_finalize.py','build_unseen_progress.py','build_unseen_progress_videos.py','unseen_progress_verify.py'):shutil.copy2(base/name,ROOT/'source'/name)
    scores=', '.join(f"{r['mode']} {r['success']}/{r['n']} ({r['collisions']} collisions)" for r in summary['table'] if r['group']=='all')
    gains=sum(c['change']=='gain' for c in summary['changes']);losses=sum(c['change']=='loss' for c in summary['changes'])
    lead=f"**E112 completed:** Frozen E111 selector on eight prospectively perturbed bodies from four previously studied novel-axis families: {scores}; {gains} gains and {losses} losses. {summary['accepted_candidates']}/16 placements pass prespecified endpoint screening. Transfer gate {'PASS' if summary['passes'] else 'FAIL'}. No route planner, weight update, budget extension or selector tuning. Exact model/state remain privileged; not history-only adaptation or untouched axis-family evidence. Saved {len(manifest['videos'])} verified success/failure videos. See Section4.102."
    verify.update(adapter_verification=json.loads((ROOT/'adapter_verification.json').read_text()),verified_videos=len(manifest['videos']),video_hashes_verified=True)
    (ROOT/'verification.json').write_text(json.dumps(verify,indent=2))
    report=(ROOT/'report.md').read_text();report+='\nFinal artifact verification: '+str(len(manifest['videos']))+' full-result video replays and SHA256 hashes verified.\n';(ROOT/'report.md').write_text(report)
    p=base/'TECHNICAL_REPORT.md';text=p.read_text(encoding='utf-8');assert '## 4.102 E112' not in text
    text=text.replace('Revision 97 · 13 November 2026','Revision 98 · 13 November 2026',1)
    text=text.replace('## 1. Research question and claim criteria\n\n','## 1. Research question and claim criteria\n\n'+lead+'\n\n',1)
    lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('| E111 |'));lines.insert(i+1,f'| E112 | Frozen progress selector on prospective unseen bodies | {scores}; {gains} gains, {losses} losses; gate {summary["passes"]} | `ease_unseen_progress_v1` |');text='\n'.join(lines)+'\n'
    section=report.replace('# E112:','## 4.102 E112:',1);section='\n'.join('### E112 '+l[3:] if l.startswith('## ') and not l.startswith('## 4.102') else l for l in section.splitlines())
    text=text.replace('## 5. Limitations and next decisions\n',section+'\n\n## 5. Limitations and next decisions\n\n'+lead+'\n',1)
    text=text.replace('- Revision 97,',f'- Revision 98, 13 November 2026: E112 prospective unseen-body replication: {scores}; {gains} gains, {losses} losses; gate {summary["passes"]}.\n- Revision 97,',1);p.write_text(text,encoding='utf-8')
    for name in ('README.md','MIXED_DESIGN.md'):
        p=base/name;first,rest=p.read_text(encoding='utf-8').split('\n',1);p.write_text(first+'\n\n'+lead+'\n'+rest,encoding='utf-8')
    (ROOT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2));print(lead)

if __name__=='__main__':run()
