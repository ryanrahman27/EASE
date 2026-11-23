from pathlib import Path
s=Path(__file__).with_name('gentle_transition_videos.py').read_text().replace('from .gentle_transition import ROOT,MODES,evaluate,prior','from .single_phase import ROOT,MOTION,evaluate,prior')
a=s.index("    rows=json.loads");b=s.index('    for case,seed,mode in selection:',a)
s=s[:a]+'''    rows=json.loads((ROOT/'records.json').read_text());selection=[(2,0,None,'baseline')]+[(2,0,p,m) for p in MOTION for m in ('budget','gentle')]
    lookup=lambda c,s,p,m:next(r for r in rows if r['case']==c and r['seed']==s and r['affected_phase']==p and r['mode']==m)
    for phase in MOTION:
        for control in ('baseline','budget'):
            refphase=None if control=='baseline' else phase
            losses=[r for r in rows if r['mode']=='gentle' and r['affected_phase']==phase and not r['success'] and lookup(r['case'],r['seed'],refphase,control)['success']]
            if losses:
                r=min(losses,key=lambda r:(r['case'],r['seed']))
                for p,m in ((refphase,control),(phase,'gentle')):
                    key=(r['case'],r['seed'],p,m)
                    if key not in selection:selection.append(key)
    for outcome in (True,False):
        if not any(lookup(*key)['success']==outcome for key in selection):
            r=next(r for r in sorted(rows,key=lambda r:(r['case'],r['seed'],r['affected_phase'] or '',r['mode'])) if r['success']==outcome);selection.append((r['case'],r['seed'],r['affected_phase'],r['mode']))
    manifest=dict(selection='Prespecified training case2 seed0 baseline and all eight phase controls. Append first gentle loss versus each reference in each phase, plus missing outcome category. Outcome-selected illustrations.',playback='Complete executed task,32 calibration steps omitted. Every fourth40ms step at6.25fps plus final frame. Fixed E99 weights; privileged geometry filter, no route planner.',videos=[])
''' +s[b:]
s=s.replace('for case,seed,mode in selection:','for case,seed,phase,mode in selection:').replace("f'{case}_{seed}_{mode}.json'", "f'{case}_{seed}_{phase}_{mode}.json'").replace("f'case{case}_seed{seed}_{mode}_{label}.mp4'", "f'case{case}_seed{seed}_{phase}_{mode}_{label}.mp4'")
s=s.replace("| {mode} | {label.upper()}","| {phase}/{mode} | {label.upper()}").replace('parents,students,capture)','parents,students,capture,affected_phase=phase)').replace('seed=seed,mode=mode,','seed=seed,affected_phase=phase,mode=mode,').replace('E103','E104').replace('Gentle phase-transition training screen','Isolated phase-transition training screen')
Path(__file__).with_name('single_phase_videos.py').write_text(s)
