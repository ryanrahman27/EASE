"""E52 fixed native development evaluation; no controller changes."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import torch
from .benchmark import fingerprint
from .real_arms import verify_assets
from .real_arm_reach import load_models
from .real_arm_pick_place import NativePickPlace
from .retention_fresh import GuardedPose
from .pick_place_control import calibrate
from .native_reliability import rollout


def run(assets, training, reference, output):
    torch.set_num_threads(4)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    train=Path(training);ref=Path(reference)
    assert json.loads((train/'status.json').read_text())['status']=='complete'
    verify_assets(assets)
    tasks=json.loads((ref/'cases.json').read_text());assert len(tasks)==12
    old=json.loads((ref/'records.json').read_text())
    (root/'protocol.json').write_text(json.dumps(dict(experiment='E52 native development',training=training,reference=reference,
        cases_sha256=fingerprint(ref/'cases.json'),reference_sha256=fingerprint(ref/'records.json'),
        selection='All six fixed final candidates, all twelve observed placements. Retain failed training gates as diagnostic. No controller change, no added probes, original 41000+case seeds. Compare to audited E46 original and analytical records.',test_evaluated=False),indent=2))
    (root/'source').mkdir()
    for name in ['control_training_eval.py','native_reliability.py','pick_place_control.py','retention_fresh.py']:
        shutil.copy2(Path(__file__).with_name(name),root/'source'/name)
    (root/'trials').mkdir();rows=[];hashes={}
    for condition in ['prediction','control']:
        models,hh=load_models(train/condition);hashes.update(hh)
        for seed in range(3):
            method=f'neural_{seed}'
            for i,t in enumerate(tasks):
                env=NativePickPlace(assets,t['robot'],t['xy'],t['target'],.2,True)
                controller=GuardedPose(env.n,models[method])
                with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
                    probe=calibrate(env,t['home'],controller,41000+i)
                    result=rollout(env,controller,object_release=t['robot']=='panda')
                baseline=next(r for r in old if r['method']==f'old_{seed}' and r['case']==i)
                assert probe==baseline['probe']
                row=dict(condition=condition,seed=seed,method=method,robot=t['robot'],case=i,probe=probe,**result)
                (root/'trials'/f'{condition}_{seed}_{i}.json').write_text(json.dumps(row))
                rows.append(row)
                print(condition,seed,i,result['success'],result['failure'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'records.json').write_text(json.dumps(rows))
    (root/'audit.json').write_text(json.dumps(dict(checkpoints=hashes,calibration_matches=len(rows),test_evaluated=False),indent=2))
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','training','reference','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))
