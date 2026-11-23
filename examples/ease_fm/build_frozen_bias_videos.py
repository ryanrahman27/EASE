"""Adapt the verified state replay renderer for the E115 comparison."""
from pathlib import Path
b=Path(__file__).parent;s=(b/'adaptation_ablation_videos.py').read_text().replace('from .adaptation_ablation import ROOT,MODES','from .frozen_bias_core import ROOT,MODES').replace('E114','E115').replace('==192','==256')
start=s.index("    for method in ('analytical'");end=s.index('    selection=list(dict.fromkeys(selection))',start)
s=s[:start]+'''    for method in ('analytical','learned_0','learned_1','learned_2'):
        for outcome in (True,False):
            candidates=[r for r in rows if r['method']==method and r['mode']=='frozen_bias' and r['success']==outcome]
            if candidates:selection.append((min(r['case'] for r in candidates),method,'frozen_bias'))
    for dof in (6,7):
        for gain in (True,False):
            candidates=[c for c in range(16) if lookup[c,'learned_0','frozen_bias']['dof']==dof and lookup[c,'learned_0','frozen_bias']['success']==gain and lookup[c,'learned_0','online']['success']!=gain]
            if candidates:
                for mode in ('frozen_bias','online'):selection.append((min(candidates),'learned_0',mode))
'''+s[end:]
s=s.replace('First success/failure per estimator checkpoint and update mode where available; all modes for first seed0 online gain/loss against each frozen control.', 'First motion-only success/failure per checkpoint; paired motion-only/online examples for first seed0 gain and loss per DOF.')
s=s.replace('Learned adaptation without a selector','Motion-only adaptation / Frozen bias')
(b/'frozen_bias_videos.py').write_text(s)
