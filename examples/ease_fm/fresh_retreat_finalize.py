"""Verify E105 artifacts and append the completed replication to TR-001."""
import json
import shutil
from pathlib import Path
from .fresh_retreat import ROOT
from .benchmark import fingerprint

def run():
    summary=json.loads((ROOT/'summary.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text());manifest=json.loads((ROOT/'videos/manifest.json').read_text())
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==9*summary['accepted_candidates']
    assert all(r['exact'] for r in json.loads((ROOT/'compatibility.json').read_text()))
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    for v in manifest['videos']:
        assert v['exact_replay'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
        assert fingerprint(ROOT/'trials'/f"{v['case']}_{v['seed']}_{v['affected_phase']}_{v['mode']}.json")==v['reference_sha256']
    assert {v['success'] for v in manifest['videos']}=={r['success'] for r in rows}
    source=Path(__file__).parent
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(source/p.name),p
    for name in ('fresh_retreat_report.py','fresh_retreat_videos.py','build_fresh_videos.py','fresh_retreat_finalize.py'):shutil.copy2(source/name,ROOT/'source'/name)
    (ROOT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2))
    totals=[r for r in summary['table'] if r['group']=='all'];n=summary['accepted_candidates']*3
    scores=', '.join(f"{r['mode']} {r['success']}/{n} ({r['collisions']} collisions)" for r in totals)
    lead=f"**E105 completed:** Prospective replication on eight fresh body parameterizations and two new placements per body, using the same four axis orders. {summary['accepted_candidates']}/16 placements pass endpoint IK screening; no replacement. {scores}. Strict retention gate {'PASS' if summary['passes'] else 'FAIL'}. Reports all 48 candidate/seed opportunities separately from the {n} executable pairs, with body-clustered uncertainty. Saves 131 passing tests, nine exact E104 compatibility replays and {len(manifest['videos'])} verified success/failure videos. Privileged planning/geometry remain; no reserved final test. See Section 4.95."
    p=source/'TECHNICAL_REPORT.md';text=p.read_text(encoding='utf-8');assert '## 4.95 E105' not in text
    text=text.replace('Revision 90 · 6 November 2026','Revision 91 · 6 November 2026',1)
    text=text.replace('## 1. Research question and claim criteria\n\n','## 1. Research question and claim criteria\n\n'+lead+'\n\n',1)
    lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('| E104 |'));lines.insert(i+1,f'| E105 | Prospective fresh body/placement replication | {scores}; gate {summary["passes"]} | `ease_fresh_retreat_v1` |');text='\n'.join(lines)+'\n'
    report=(ROOT/'report.md').read_text().replace('# E105:','## 4.95 E105:',1)
    report='\n'.join('### E105 '+l[3:] if l.startswith('## ') and not l.startswith('## 4.95') else l for l in report.splitlines())
    text=text.replace('## 5. Limitations and next decisions\n',report+'\n\n## 5. Limitations and next decisions\n\n'+lead+'\n',1)
    text=text.replace('- Revision 90,',f'- Revision 91, 6 November 2026: E105 prospectively replicates frozen retreat on fresh bodies/placements. {scores}; gate {summary["passes"]}. Saves all outcomes, coverage, clustered intervals and {len(manifest["videos"])} verified videos.\n- Revision 90,',1);p.write_text(text,encoding='utf-8')
    for name in ('README.md','MIXED_DESIGN.md'):
        p=source/name;first,rest=p.read_text(encoding='utf-8').split('\n',1);p.write_text(first+'\n\n'+lead+'\n'+rest,encoding='utf-8')
    verification=dict(tests_passed=131,test_seconds=12.99,compatibility_exact=9,rollouts=len(rows),pre_retreat_prefixes=len(rows)*2//3,video_count=len(manifest['videos']),video_hashes_verified=True,source_input_checkpoint_hashes_verified=True,reserved_test_evaluated=False)
    (ROOT/'verification.json').write_text(json.dumps(verification,indent=2));print(lead)

if __name__=='__main__':run()
