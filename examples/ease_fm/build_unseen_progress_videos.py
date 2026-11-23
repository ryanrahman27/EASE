"""Reuse verified renderer with E112 cohort and prespecified selection."""
from pathlib import Path
base=Path(__file__).parent
s=(base/'progress_selector_videos.py').read_text()
s=s.replace('from .progress_selector import ROOT,evaluate','from .unseen_progress import ROOT,models\nfrom .unseen_progress_selector import evaluate as raw_evaluate\n\ndef evaluate(case,t,seed,mode,parents,students,record_step=None,affected_phase=None):\n    r=raw_evaluate(case,t,seed,mode,parents,students,record_step,affected_phase=affected_phase)\n    r.update(body=t["body"],candidate=t["candidate"],validation=True)\n    return r\n')
start=s.index('    tasks=json.loads(');end=s.index('    rows=json.loads(',start)
s=s[:start]+"    tasks=json.loads((ROOT/'cases.json').read_text());AXES.update(json.loads((ROOT/'axis_catalog.json').read_text()));parents,students,hashes=models()\n"+s[end:]
start=s.index('    changed=sorted(');end=s.index('    for outcome in (True,False):',start)
s=s[:start]+'''    for collision in (True,False):
        failed=sorted([r for r in rows if r['mode']=='physics' and not r['success'] and r['collision']==collision],key=lambda r:(r['case'],r['seed']))
        if failed:
            for m in ('baseline','physics'):
                key=(failed[0]['case'],failed[0]['seed'],None,m)
                if key not in selection:selection.append(key)
'''+s[end:]
s=s.replace('Prespecified case0 seed0 both modes; every paired task gain/loss, earliest changed-action pair, first collision-free physics failure, missing outcome category. Supplemental first changed pair per earlier phase, selected after partial outcomes to show the intervention. Outcome-selected illustrations; aggregate evaluation unchanged.','Prespecified case0 seed0 both modes; every paired task gain/loss; first remaining selector collision and first contact-free selector failure both modes; missing outcome category. Outcome-selected illustrations; aggregate evaluation unchanged.')
s=s.replace('E111 training case','E112 unseen-body case').replace('Progress-aware physics selector on training bodies','Frozen progress-aware selector on prospective unseen bodies')
(base/'unseen_progress_videos.py').write_text(s)
