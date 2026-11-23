"""Structural PiPER downward-tool workspace diagnostic; no policy rollouts."""
import argparse
import json
from pathlib import Path
import numpy as np
from .real_arm_pick_place import NativePickPlace,endpoint_ik


def run(assets,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);rng=np.random.default_rng(40100);rows=[]
    (root/'protocol.json').write_text(json.dumps(dict(scope='Post-pilot structural workspace diagnostic only; no policy-outcome filtering',x=[.2,.25,.3,.35],z=[.24,.28,.32,.36],y=0,base_heights=[0,.2],seed=40100),indent=2))
    for height in [0.,.2]:
        for x in [.2,.25,.3,.35]:
            env=NativePickPlace(assets,'piper',(x,0),(x,.06),height);home=env.observe()['q']
            for z in [.24,.28,.32,.36]:
                q=endpoint_ik(env,np.array([x,0,z]),rng,home)
                rows.append(dict(base_height=height,x=x,z=z,feasible=q is not None,q=None if q is None else q.tolist()))
            print('workspace',height,x,[r['z'] for r in rows if r['base_height']==height and r['x']==x and r['feasible']],flush=True)
    (root/'results.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--output',required=True);run(**vars(p.parse_args()))
