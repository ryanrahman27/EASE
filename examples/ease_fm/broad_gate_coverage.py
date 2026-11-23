"""Audit the fixed E125 placement split before collecting new outcomes."""
import json,shutil
from pathlib import Path
import numpy as np
from .broad_task_value_gate import ROOT,OLD,fingerprint

def run():
    tasks=json.loads((ROOT/'cases.json').read_text());original=json.loads((OLD/'cases.json').read_text());assert tasks[:24]==original
    assert len(tasks)==56 and sum(not t['validation'] for t in tasks)==48
    for t in tasks[24:]:
        assert not t['validation'] and not original[t['source_case']]['validation']
        np.testing.assert_allclose(np.array(t['xy'])-original[t['source_case']]['xy'],t['object_offset'],atol=1e-15,rtol=0)
        np.testing.assert_allclose(np.array(t['target'])-original[t['source_case']]['target'],t['target_offset'],atol=1e-15,rtol=0)
        assert t['home']==original[t['source_case']]['home'] and t['arm']==original[t['source_case']]['arm']
    bodies=sorted(set(t['arm']['seed'] for t in tasks))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(9,8))
    for ax,body in zip(axes.flat,bodies):
        for label,color,marker,ids in [('Original fitting','#288a5b','o',[i for i,t in enumerate(tasks[:24]) if not t['validation']]),('Added fitting','#377bb5','^',list(range(24,56))),('Unchanged held','#be6527','s',[i for i,t in enumerate(tasks[:24]) if t['validation']])]:
            pts=np.array([tasks[i]['xy'] for i in ids if tasks[i]['arm']['seed']==body]);ax.scatter(pts[:,0],pts[:,1],label=label,c=color,marker=marker,s=48)
        dof=next(len(t['home']) for t in tasks if t['arm']['seed']==body);ax.set_title(f'Training body {body} / {dof} joints');ax.set_xlabel('Object X (m)');ax.set_ylabel('Object Y (m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(ROOT/'placement_coverage.png',dpi=160);plt.close(fig)
    audit=dict(original24_exact=True,original_held_cases=[i for i,t in enumerate(original) if t['validation']],fitting_cases=[i for i,t in enumerate(tasks) if not t['validation']],new_placements=32,held_outcomes_used_for_placement_selection=False,source_cases_fitting_only=True,cases_sha256=fingerprint(ROOT/'cases.json'))
    (ROOT/'placement_audit.json').write_text(json.dumps(audit,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)

if __name__=='__main__':run()
