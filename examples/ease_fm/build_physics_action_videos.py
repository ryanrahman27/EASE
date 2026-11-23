from pathlib import Path

p=Path(__file__).with_name('single_phase_videos.py');s=p.read_text()
s=s.replace('from .single_phase import ROOT,MOTION,evaluate,prior','from .physics_action import ROOT,evaluate\nfrom .single_phase import prior')
start=s.index('    rows=json.loads');end=s.index('    for case,seed,phase,mode in selection:',start)
s=s[:start]+'''    rows=json.loads((ROOT/'records.json').read_text());selection=[(0,0,None,m) for m in ('baseline','physics')]
    lookup=lambda c,s,p,m:next(r for r in rows if r['case']==c and r['seed']==s and r['mode']==m)
    for r in rows:
        if r['mode']=='physics' and r['success']!=lookup(r['case'],r['seed'],None,'baseline')['success']:
            for m in ('baseline','physics'):
                key=(r['case'],r['seed'],None,m)
                if key not in selection:selection.append(key)
    changed=sorted([r for r in rows if r['mode']=='physics' and r['changed_actions']],key=lambda r:(r['case'],r['seed']))
    if changed:
        for m in ('baseline','physics'):
            key=(changed[0]['case'],changed[0]['seed'],None,m)
            if key not in selection:selection.append(key)
    for outcome in (True,False):
        if any(r['success']==outcome for r in rows) and not any(lookup(*key)['success']==outcome for key in selection):
            r=next(r for r in sorted(rows,key=lambda r:(r['case'],r['seed'],r['mode'])) if r['success']==outcome)
            selection.append((r['case'],r['seed'],None,r['mode']))
    manifest=dict(selection='Prespecified case0 seed0 both modes; every paired task gain/loss, earliest changed-action pair, missing outcome category. Outcome-selected illustrations.',playback='Complete executed task except32 calibration steps. Every fourth40ms step at6.25fps plus final frame. Fixed weights; privileged geometry and exact physics lookahead in physics mode; no route planner.',videos=[])
'''+s[end:]
s=s.replace('E104 training','E108 training').replace('EXECUTED | privileged geometry | fixed neural weights','EXECUTED | privileged physics/geometry | fixed weights')
s=s.replace('Isolated phase-transition training screen','Physics-based action selection on training bodies')
Path(__file__).with_name('physics_action_videos.py').write_text(s)
