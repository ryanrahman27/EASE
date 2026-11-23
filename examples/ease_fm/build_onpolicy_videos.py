"""Build E101 gallery renderer with explicit original/starting references."""
from pathlib import Path
s=Path(__file__).with_name('task_retention_videos.py').read_text()
s=s.replace('from .task_retention_training import ROOT,evaluate,previous,MotionTransformer,load_students','from .onpolicy_history import ROOT,evaluate,previous,MotionTransformer,load_students,e99')
s=s.replace('students,new_hashes=load_students();hashes.update(new_hashes)',"students,new_hashes=load_students();hashes.update(new_hashes)\n    starts,start_hashes=e99.load_students();hashes.update(start_hashes)\n    students.update({('start',seed):starts[('retained',seed)] for seed in range(3)})")
s=s.replace("rows=json.loads((ROOT/'records.json').read_text())", "rows=json.loads((ROOT/'records.json').read_text())\n    rows += [dict(r,mode='start') for r in json.loads((e99.ROOT/'records.json').read_text()) if r['mode']=='retained']")
s=s.replace("('parent','data_only','retained')", "('parent','replay','onpolicy')")
a=s.index("    for condition in ('data_only','retained'):");b=s.index('    for case,method,mode in selection:',a)
s=s[:a]+'''    for condition in ('replay','onpolicy'):
        for reference in ('parent','start'):
            losses=[r for r in rows if r['mode']==condition and not r['success'] and outcome((r['case'],r['method'],reference))]
            if losses:
                r=min(losses,key=lambda r:(r['case'],r['method']))
                for mode in (reference,condition):
                    key=(r['case'],r['method'],mode)
                    if key not in selection:selection.append(key)
    manifest=dict(selection='Prespecified case2 seed0 and cases10/15 seed2 in parent/replay/onpolicy. Append first regression against parent and starting student for each condition with its reference. Outcome-selected illustrations, not representative samples.',playback='Executed full task; calibration omitted. Every fourth40ms step at6.25fps plus final frame. No task cuts. Original parent/planner/probe prefix, weights change only afterward. Privileged geometry and route planning.',videos=[])
''' + s[b:]
s=s.replace("else 'data_only'", "else 'replay'").replace('E99','E101').replace('Task data and action-retention neural continuation','Student histories versus matched replay')
Path(__file__).with_name('onpolicy_history_videos.py').write_text(s)
