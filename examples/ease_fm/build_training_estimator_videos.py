from pathlib import Path
b=Path(__file__).parent;s=(b/'adaptation_ablation_videos.py').read_text().replace('from .adaptation_ablation import ROOT,MODES','from .training_estimator_core import ROOT\nMODES=("frozen_bias",)').replace('E114','E117').replace('==192','==84')
start=s.index("    for method in ('analytical'");end=s.index('    selection=list(dict.fromkeys(selection))',start)
s=s[:start]+'''    for method in ('analytical',*[f'{family}_{seed}' for family in ('learned','candidate') for seed in range(3)]):
        for outcome in (True,False):
            candidates=[r for r in rows if r['method']==method and r['success']==outcome]
            if candidates:selection.append((min(r['case'] for r in candidates),method,'frozen_bias'))
    for seed in range(3):
        for gain in (True,False):
            candidates=[r for r in rows if r['method']==f'candidate_{seed}' and r['success']==gain and lookup[r['case'],f'learned_{seed}','frozen_bias']['success']!=gain]
            if candidates:
                case=min(r['case'] for r in candidates)
                for method in (f'learned_{seed}',f'candidate_{seed}'):selection.append((case,method,'frozen_bias'))
'''+s[end:]
s=s.replace('First success/failure per estimator checkpoint and update mode where available; all modes for first seed0 online gain/loss against each frozen control.', 'First success/failure per parent/candidate seed and analytical reference; paired first gain/loss per candidate seed.')
s=s.replace('Learned adaptation without a selector','Training-body estimator repair')
(b/'training_estimator_videos.py').write_text(s)
