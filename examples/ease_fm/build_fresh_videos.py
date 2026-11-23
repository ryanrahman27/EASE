"""Reuse the verified renderer, with E105-specific selection and evaluator."""
from pathlib import Path
p=Path(__file__).with_name('single_phase_transfer_videos.py')
s=p.read_text();start=s.index('from .single_phase import');end=s.index('from .mixed_sim import')
s=s[:start]+'''from .fresh_retreat import ROOT,retained,evaluate as fresh_evaluate
MOTION=('retreat',)
def evaluate(case,t,seed,mode,parents,students,record_step=None,affected_phase=None):
    return fresh_evaluate(case,t,seed,mode,parents,students,record_step)
'''+s[end:]
s=s.replace("(prior.previous.PRIOR/'cases.json')","(ROOT/'cases.json')").replace("(prior.previous.PRIOR/'axis_catalog.json')","(ROOT/'axis_catalog.json')").replace('prior.previous.models()','retained.previous.models()').replace('prior.e99.load_students()','retained.load_students()')
start=s.index("    rows=json.loads");end=s.index('    for outcome in (True,False):',start)
s=s[:start]+'''    rows=json.loads((ROOT/'records.json').read_text());selection=[(0,0,None,'baseline'),(0,0,'retreat','budget'),(0,0,'retreat','gentle')]
    lookup=lambda c,s,p,m:next(r for r in rows if r['case']==c and r['seed']==s and r['affected_phase']==p and r['mode']==m)
    for r in rows:
        if r['mode']!='gentle':continue
        if any(r['success']!=lookup(r['case'],r['seed'],p,m)['success'] for p,m in ((None,'baseline'),('retreat','budget'))):
            for p,m in ((None,'baseline'),('retreat','budget'),('retreat','gentle')):
                key=(r['case'],r['seed'],p,m)
                if key not in selection:selection.append(key)
'''+s[end:]
s=s.replace("if not any(lookup(*key)['success']==outcome for key in selection):","if any(r['success']==outcome for r in rows) and not any(lookup(*key)['success']==outcome for key in selection):")
start=s.index('    manifest=dict(');end=s.index('    for case,seed,phase,mode in selection:',start)
s=s[:start]+'''    manifest=dict(selection='First accepted case seed0 all modes; every gain or loss versus either control with all modes; append missing outcome category. Outcome-selected illustrations.',playback='Full executed task except32 calibration steps. Every fourth40ms step at6.25fps plus final frame. Frozen neural weights; privileged route planner and geometry filter.',videos=[])
'''+s[end:]
s=s.replace('E104 development','E105 fresh-body').replace('Isolated phase-transition development transfer','Prospective body and placement replication')
p.with_name('fresh_retreat_videos.py').write_text(s)
