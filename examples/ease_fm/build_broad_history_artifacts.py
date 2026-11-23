from pathlib import Path
b=Path(__file__).parent;s=(b/'adaptation_ablation_videos.py').read_text().replace('from .adaptation_ablation import ROOT,MODES','from .broad_history_repair import ROOT\nMODES=("frozen_bias",)').replace('E114','E118').replace('==192','==204')
start=s.index("    for method in ('analytical'");end=s.index('    selection=list(dict.fromkeys(selection))',start)
s=s[:start]+'''    tasks=json.loads((ROOT/'cases.json').read_text())
    for method in ('analytical',*[f'{family}_{seed}' for family in ('reference','candidate') for seed in range(3)]):
        for outcome in (True,False):
            candidates=[r for r in rows if r['method']==method and r['success']==outcome]
            if candidates:selection.append((min(r['case'] for r in candidates),method,'frozen_bias'))
    for seed in range(3):
        for group in ('all','fresh_held'):
            for gain in (True,False):
                candidates=[r for r in rows if r['method']==f'candidate_{seed}' and r['success']==gain and lookup[r['case'],f'reference_{seed}','frozen_bias']['success']!=gain and (group=='all' or tasks[r['case']]['group']==group)]
                if candidates:
                    case=min(r['case'] for r in candidates)
                    for method in (f'reference_{seed}',f'candidate_{seed}'):selection.append((case,method,'frozen_bias'))
'''+s[end:]
s=s.replace('First success/failure per estimator checkpoint and update mode where available; all modes for first seed0 online gain/loss against each frozen control.', 'First reference/candidate success/failure per seed; paired first gain/loss overall and on fresh held placements per seed.')
s=s.replace('Learned adaptation without a selector','Broader training histories / Phase retention')
(b/'broad_history_videos.py').write_text(s)
s=(b/'training_estimator_phase_audit.py').read_text().replace('from .training_estimator_repair import ROOT,setup,predict,analytical_targets,action_tensor','from .broad_history_repair import ROOT,setup,predict,analytical_targets,action_tensor').replace('_,models,_=setup()','_,_,models,_=setup()')
s=s.replace("f=np.load(ROOT/'contexts.npz');", "archive=np.load(ROOT/'contexts.npz');f={k:archive[k] for k in archive.files};")
(b/'broad_history_phase_audit.py').write_text(s)
