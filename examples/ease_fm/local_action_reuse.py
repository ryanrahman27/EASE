"""Verify exact prediction-only replication before reusing E52 gate records."""
import argparse
import json
from pathlib import Path
import shutil
from .benchmark import fingerprint


def run(training,reference):
    root=Path(training);old=Path(reference)
    assert json.loads((root/'status.json').read_text())['status']=='complete'
    gate=json.loads((old/'gate_prediction'/'audit.json').read_text());matches=[]
    for seed in range(3):
        a=root/'prediction'/f'seed_{seed}';b=old/'prediction'/f'seed_{seed}'
        x=json.loads((a/'training.json').read_text());y=json.loads((b/'training.json').read_text())
        assert fingerprint(a/'policy.pt')==fingerprint(b/'policy.pt')==x['policy_sha256']==y['policy_sha256']
        assert x['batch_sha256']==y['batch_sha256']
        assert [r[:2] for r in x['losses']]==[r[:2] for r in y['losses']]
        item=next(r for r in gate['summary'] if r['seed']==seed)
        assert item['policy_sha256']==x['policy_sha256']
        assert fingerprint(old/'gate_prediction'/f'seed_{seed}.json')==item['records_sha256']
        matches.append(dict(seed=seed,identical_checkpoint=True,identical_batches=True,identical_fit_anchor_losses=True))
    shutil.copytree(old/'gate_prediction',root/'gate_prediction')
    (root/'prediction_reuse.json').write_text(json.dumps(dict(reference=reference,matches=matches,reused_gate_episodes=768),indent=2))
    print(json.dumps(matches))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['training','reference']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))
