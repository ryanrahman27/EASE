"""Reuse full calibration/history/budget verification; write E177 reporting."""
from pathlib import Path

def run():
    base=Path(__file__).parent;s=(base/'frozen_estimator_report.py').read_text().replace('from .frozen_estimator_comparison import','from .unseen_estimator import')
    s=s.replace("selected={0,26,43,62,65,67}","selected=set()")
    s=s.replace("AXES.update(read('outputs/ease_unseen_progress_v1/axis_catalog.json'))", "AXES.update(read(ROOT/'axis_catalog.json'));seen_bodies=set()\n    for i,t in enumerate(tasks):\n        if t['arm']['seed'] not in seen_bodies:selected.add(i);seen_bodies.add(t['arm']['seed'])")
    s=s.replace("ref=read(REFERENCE/'clearance'/f'{case}_e120_ensemble_frozen_bias.json')", "other='analytical' if method=='e120_ensemble' else 'e120_ensemble';ref=read(ROOT/other/'clearance'/f'{case}_{other}_frozen_bias.json')")
    s=s.replace("    sources.append((26,Path('outputs/ease_object_preserving_v1/guarded_handoff'),'e120_ensemble','privileged_E174_historical'))\n",'')
    start=s.index("    lines=['# E176:");end=s.index('    import matplotlib',start)
    s=s[:start]+'''    from .unseen_estimator_writeup import write_report
    write_report(summary,metrics,case_rows,paired,pairs)
''' +s[end:]
    s=s.replace(".replace('link_clearance_diagnostic import ROOT','frozen_estimator_comparison import ROOT')", ".replace('link_clearance_diagnostic import ROOT','unseen_estimator import ROOT')").replace(".replace('E168','E176')", ".replace('E168','E177')")
    s=s.replace('frozen_estimator_videos.py','unseen_estimator_videos.py').replace('E176 | frozen controller, matched budgets | privileged fitting cases','E177 | frozen controller | prospective unseen bodies')
    s=s.replace('E120 versus analytical estimation with the same privileged E175 controller and budget caps. Fitting cases only. Historical E174 failure is labeled separately.','Prospective new six-joint bodies: E120 versus analytical, identical privileged controller and budget rules. Successes and failures retained; previously used axis families.')
    start=s.index('\ndef finish():');s=s[:start]+'''\nif __name__=='__main__':run()
'''
    (base/'unseen_estimator_report.py').write_text(s)

if __name__=='__main__':run()
