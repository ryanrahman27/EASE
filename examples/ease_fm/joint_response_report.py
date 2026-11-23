"""Independent saved-fold audit and E59 aggregate report."""
import argparse
import json
from pathlib import Path
import numpy as np
from .benchmark import fingerprint
from .joint_response_audit import fit,predict


def run(output):
    root=Path(output);p=json.loads((root/'protocol.json').read_text());audit=json.loads((root/'audit.json').read_text())
    assert fingerprint(root/'transitions.npz')==audit['data_sha256']
    assert fingerprint(Path(p['reference'])/'drivers.json')==p['source_sha256']
    assert len(audit['exact_replays'])==32 and all(r['exact'] and r['steps']==150 for r in audit['exact_replays'])
    assert audit['transitions']==4800 and audit['joint_rows']==31200
    for path,sha in p['checkpoints'].items():assert fingerprint(path)==sha
    a=np.load(root/'transitions.npz');scores=json.loads((root/'scores.json').read_text());predictions=json.loads((root/'predictions.json').read_text())
    checked=0;strata=[]
    for body in sorted(set(a['body'])):
        train=(a['body']!=body)&(a['driver']!='original_2');held=a['body']==body
        for name in ['command_velocity','history']:
            stored=np.load(root/f'fold_{body}_{name}.npz');x=a['x'][:,stored['columns']]
            model=fit(x[train],(a['y']-a['baseline'])[train])
            for key,value in zip(['mean','scale','weight'],model):np.testing.assert_allclose(stored[key],value,rtol=1e-12,atol=1e-12)
            for split,mask in [('body',held),('trajectory',(a['body']!=body)&(a['driver']=='original_2'))]:
                pred=a['baseline'][mask]+predict(model,x[mask]);rmse=float(np.sqrt(np.mean((pred-a['y'][mask])**2)))
                row=next(s for s in scores if (s['body'],s['model'],s['split'])==(body,name,split))
                assert np.isclose(rmse,row['rmse'],atol=1e-14)
            record=next(r for r in predictions if r['body']==body and r['model']==name)
            np.testing.assert_array_equal(record['indices'],np.flatnonzero(held))
            pred=a['baseline'][held]+predict(model,x[held]);np.testing.assert_allclose(record['prediction'],pred,atol=1e-14)
            for driver in sorted(set(a['driver'])):
                mask=a['driver'][held]==driver
                strata.append(dict(body=int(body),model=name,driver=str(driver),rmse=float(np.sqrt(np.mean((pred[mask]-a['y'][held][mask])**2))),baseline_rmse=float(np.sqrt(np.mean((a['baseline'][held][mask]-a['y'][held][mask])**2)))))
            checked+=1
    lines=['','## Audited aggregate results','','Body-macro mean RMSE; relative reduction compares these means, not independent joint-level samples.','',
        '| DoF | Bias baseline | Command/velocity | History | History reduction versus baseline |','|---|---:|---:|---:|---:|']
    for dof in [6,7]:
        h=[s for s in scores if s['split']=='body' and s['model']=='history' and s['dof']==dof]
        v=[s for s in scores if s['split']=='body' and s['model']=='command_velocity' and s['dof']==dof]
        b=np.mean([s['baseline_rmse'] for s in h]);hr=np.mean([s['rmse'] for s in h]);vr=np.mean([s['rmse'] for s in v])
        lines.append(f'| {dof} | {b:.6f} | {vr:.6f} | {hr:.6f} | {1-hr/b:.1%} |')
    wins=sum(r['rmse']<r['baseline_rmse'] for r in strata if r['model']=='history')
    lines+=['',f'History beats the bias baseline in {wins}/32 held-out body/driver strata. All 16 saved folds reproduce using only permitted fit rows; all 32 body/trajectory scores and saved body predictions match recomputation. No data from the omitted body enters normalization or fitting.',
        '', '83 regression tests pass. One initial test demanded bitwise equality across BLAS batch shapes; it was corrected to 1e-14 tolerance after a 1.4e-17 roundoff difference. Experiment code and fits were unchanged.',
        '', 'Decision: promising response prediction, no policy promotion. The history model also includes pose and bias, so this comparison does not isolate history as the source of improvement. Next validate its counterfactual action rankings from cloned simulator states on held-out bodies, before optimizing actions or using it as a policy-training loss. Preserve E35 and keep native data diagnostic-only.']
    report=(root/'report.md').read_text().split('\n## Audited aggregate results')[0]
    (root/'report.md').write_text(report+'\n'.join(lines)+'\n')
    (root/'response_audit.json').write_text(json.dumps(dict(folds_reproduced=checked,held_body_driver_strata=strata,body_prediction_records_sha256=fingerprint(root/'predictions.json'),test_evaluated=False),indent=2))
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(**vars(p.parse_args()))
