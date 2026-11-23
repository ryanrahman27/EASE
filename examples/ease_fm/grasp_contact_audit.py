"""E102: read-only 2ms contact and object-motion audit of placement misses."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from . import onpolicy_history as prior
from .arm_pick_place import PickPlace
from .mixed_sim import AXES
from .benchmark import fingerprint

ROOT=Path('outputs/ease_grasp_contact_audit_v1')


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    changes=json.loads((prior.ROOT/'summary.json').read_text())['changes']
    failures=sorted({(r['case'],r['method'],r['condition']) for r in changes if r['control'] in ('parent','start') and r['change']=='loss' and r['first_collision'] is None})
    selections=[]
    for case,method,mode in failures:
        choices=[m for m in ('parent','start') if json.loads((prior.ROOT/'trials'/f'{case}_{method}_{m}.json').read_text())['success']]
        assert choices
        for m in (choices[0],mode):
            if (case,method,m) not in selections:selections.append((case,method,m))
    parents,hashes=prior.previous.models();students,hh=prior.load_students();hashes.update(hh)
    starts,hh=prior.e99.load_students();hashes.update(hh);students.update({('start',s):starts[('retained',s)] for s in range(3)})
    tasks=json.loads((prior.previous.PRIOR/'cases.json').read_text());AXES.update(json.loads((prior.previous.PRIOR/'axis_catalog.json').read_text()))
    inputs={str(prior.ROOT/'summary.json'):fingerprint(prior.ROOT/'summary.json')}
    for case,method,mode in selections:
        p=prior.ROOT/'trials'/f'{case}_{method}_{mode}.json';inputs[str(p)]=fingerprint(p)
    spec=dict(experiment='E102',failures=failures,selections=selections,checkpoints=hashes,inputs=inputs,test_evaluated=False,
        design='All five E101 non-collision regressions against either original parent or starting student. Pair each with successful original parent, otherwise starting student, same case/seed. Outcome-selected diagnostic only. Instrument existing contacts calls after every2ms physics substep; record object position in gripper frame, tool/object positions, bilateral pad contacts and normal forces, cube-table contact and finger positions. Do not alter control or physics. Full serialized results must exactly replay E101.',
        limits='Contact sampling at2ms; contacts indicate instantaneous geometry/constraint state, not guaranteed grasp stability. Relative displacement can include object rotation/rolling. No causal intervention or new benchmark success estimate. Privileged simulator diagnostics and unchanged privileged planning. No training or final-test evaluation.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    original=PickPlace.contacts;counts=[]
    for case,method,mode in selections:
        samples=[];last_time=[None]
        def contacts(env):
            result=original(env)
            if getattr(env,'phase',None) in ('close','lift','transport','lower','release','retreat','settle') and env.data.time!=last_time[0]:
                last_time[0]=env.data.time;obs=env.observe();obj=env.data.xpos[env.obj].copy();forces={k:0. for k in ('gripper_left_pad','gripper_right_pad','table')};touch={k:False for k in forces}
                for i,contact in enumerate(env.data.contact[:env.data.ncon]):
                    pair={int(contact.geom1),int(contact.geom2)}
                    if env.gids['cube'] not in pair:continue
                    for name in forces:
                        if env.gids[name] in pair:
                            f=np.zeros(6);mujoco.mj_contactForce(env.model,env.data,i,f);forces[name]+=max(0.,float(f[0]));touch[name]=True
                samples.append(dict(time=float(env.data.time),step=env.steps,phase=env.phase,object=obj.tolist(),tool=obs['position'].tolist(),relative=(obs['rotation'].T@(obj-obs['position'])).tolist(),fingers=env.data.qpos[env.n:env.n+2].tolist(),touch=touch,normal_force=forces,xy_error_mm=float(1000*np.linalg.norm(obj[:2]-env.target))))
            return result
        with patch.object(PickPlace,'contacts',contacts):r=prior.evaluate(case,tasks[case],method,parents[method],students[(mode if mode!='parent' else 'replay',int(method[-1]))],mode)
        ref=prior.ROOT/'trials'/f'{case}_{method}_{mode}.json';assert json.loads(json.dumps(r))==json.loads(ref.read_text())
        stem=f'{case}_{method}_{mode}';(ROOT/'trials'/f'{stem}.json').write_text(json.dumps(r));(ROOT/'trials'/f'{stem}_contacts.json').write_text(json.dumps(samples))
        counts.append(dict(case=case,method=method,mode=mode,samples=len(samples),success=r['success']));print(stem,'exact replay;',len(samples),'contact samples',flush=True)
    for p,h in {**hashes,**inputs}.items():assert fingerprint(p)==h
    (ROOT/'verification.json').write_text(json.dumps(dict(exact_replays=len(selections),sample_counts=counts,hashes_unchanged=True,test_evaluated=False),indent=2))


if __name__=='__main__':run()
