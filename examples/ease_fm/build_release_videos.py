from pathlib import Path
b=Path(__file__).parent
s=(b/'unseen_progress_videos.py').read_text().replace('from .unseen_progress import ROOT,models','from .release_selector import ROOT,models').replace('from .unseen_progress_selector import evaluate as raw_evaluate','from .release_selector import evaluate as raw_evaluate')
s=s.replace('    for outcome in (True,False):', '''    warnings=sorted([r for r in rows if r['mode']=='physics' and r['release_warnings']],key=lambda r:(r['case'],r['seed']))
    if warnings:
        for m in ('baseline','physics'):
            key=(warnings[0]['case'],warnings[0]['seed'],None,m)
            if key not in selection:selection.append(key)
    for outcome in (True,False):''')
s=s.replace('E112 unseen-body case','E113 six-joint case').replace('Frozen progress-aware selector on prospective unseen bodies','Release extension: six-joint development episodes').replace('Fixed weights; privileged geometry', 'baseline = original selector; physics = release extension. Fixed weights; privileged geometry').replace('missing outcome category. Outcome-selected','first release-warning pair; missing outcome category. Outcome-selected')
(b/'release_selector_videos.py').write_text(s)
