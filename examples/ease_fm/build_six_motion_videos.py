from pathlib import Path
b=Path(__file__).parent;s=(b/'adaptation_ablation_videos.py').read_text()
s=s.replace('from .adaptation_ablation import ROOT,MODES','from .six_motion_core import ROOT\nMODES=("baseline","probe","lift")').replace('E114','E116').replace('==192','==80')
s=s.replace("['mode']","['condition']").replace("'mode':","'condition':")
start=s.index("    for method in ('analytical'");end=s.index('    selection=list(dict.fromkeys(selection))',start)
s=s[:start]+'''    for method in ('analytical','learned_0','learned_1','learned_2'):
        for outcome in (True,False):
            candidates=[r for r in rows if r['method']==method and r['condition']=='baseline' and r['success']==outcome]
            if candidates:selection.append((min(r['case'] for r in candidates),method,'baseline'))
    for condition in ('probe','lift'):
        for gain in (True,False):
            candidates=[r for r in rows if r['condition']==condition and r['success']==gain and lookup[r['case'],r['method'],'baseline']['success']!=gain]
            if candidates:
                r=min(candidates,key=lambda r:(r['case'],r['method']))
                for mode in MODES:selection.append((r['case'],r['method'],mode))
                selection.append((r['case'],'analytical','baseline'))
'''+s[end:]
s=s.replace('First success/failure per estimator checkpoint and update mode where available; all modes for first seed0 online gain/loss against each frozen control.', 'First baseline success/failure per estimator; first recovery/regression per substitution, with baseline, both substitutions and analytical reference.')
s=s.replace('Learned adaptation without a selector','Six-joint motion estimator diagnosis')
(b/'six_motion_videos.py').write_text(s)
