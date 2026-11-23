"""Check E61 affine conversion against every E59 fit and measured feature row."""
import argparse
import json
from pathlib import Path
import numpy as np
from .benchmark import fingerprint
from .joint_response_audit import predict


def run(response,output):
    root=Path(response);out=Path(output)
    with np.load(root/'transitions.npz') as archive:a={k:archive[k] for k in archive.files}
    assert fingerprint(root/'transitions.npz')==json.loads((root/'audit.json').read_text())['data_sha256']
    max_error=0.;count=0
    for body in sorted(set(a['body'])):
        fold=np.load(root/f'fold_{body}_history.npz');coef=fold['weight'][1:]/fold['scale']
        intercept=fold['weight'][0]-fold['mean']@coef
        x=a['x'][a['body']==body].copy();alpha=np.float32(1+coef[0]);offset=(-x[:,5]+intercept+x[:,1:]@coef[1:]).astype(np.float32)
        for command in [-.04,-.02,0.,.02,.04]:
            x[:,0]=command
            expected=command-x[:,5]+predict((fold['mean'],fold['scale'],fold['weight']),x)
            actual=alpha*np.float32(command)+offset
            error=float(np.max(np.abs(actual-expected)));assert error<1e-7
            max_error=max(max_error,error);count+=len(x)
    result=dict(affine_comparisons=count,max_absolute_float32_error=max_error,command_values=[-.04,-.02,0,.02,.04],response_data_sha256=fingerprint(root/'transitions.npz'))
    (out/'affine_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--response',required=True);p.add_argument('--output',required=True);run(**vars(p.parse_args()))
