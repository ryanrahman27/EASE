"""Cross-check E60 against E58 branches and independent E59 predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
from .benchmark import fingerprint


def run(output,response):
    root=Path(output);response=Path(response)
    p=json.loads((root/'protocol.json').read_text());rows=json.loads((root/'records.json').read_text())
    assert fingerprint(Path(p['replay'])/'records.json')==p['replay_sha256']
    old=json.loads((Path(p['replay'])/'records.json').read_text())
    a=np.load(response/'transitions.npz');predictions=json.loads((response/'predictions.json').read_text())
    source_audit=json.loads((response/'audit.json').read_text());assert fingerprint(response/'transitions.npz')==source_audit['data_sha256']
    lookups={}
    for r in predictions:lookups[(r['body'],r['model'])]=dict(zip(r['indices'],r['prediction']))
    exact=0;matching=0;archived=[]
    for path in sorted((root/'trials').glob('*.json')):archived.extend(json.loads(path.read_text()))
    key=lambda r:(r['case'],r['driver'],r['index'])
    assert sorted(archived,key=key)==sorted(rows,key=key)
    for r in rows:
        baseline=next(s for s in old if key(s)==key(r))
        for c,b in zip(r['candidates'],baseline['candidates']):
            assert c['method']==b['method'] and c['branch']==b['branch'];exact+=1
            for score,value in b['scores'].items():assert c['scores'][score]==value
        c=next(c for c in r['candidates'] if c['method']==r['driver'])
        indices=np.flatnonzero((a['body']==r['case'])&(a['driver']==r['driver'])&(a['step']==r['index']))
        for kind in ['history','command_velocity']:
            pred=np.array([lookups[(r['case'],kind)][int(i)] for i in indices])
            mse=float(np.mean((pred-a['y'][indices])**2))
            np.testing.assert_allclose(c['response_errors'][kind],mse,atol=1e-14,rtol=1e-10)
            matching+=1
    assert exact==2880 and matching==576
    for path,sha in {**p['response_hashes'],**p['checkpoints']}.items():assert fingerprint(path)==sha
    result=dict(exact_replayed_branches=exact,independent_E59_prediction_checks=matching,trial_records_match=True,unchanged_weights=True,test_evaluated=False)
    (root/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--response',required=True);run(**vars(p.parse_args()))
