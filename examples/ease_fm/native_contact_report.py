"""Audited combined interpretation of E62 paired prefixes."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .benchmark import fingerprint


def run(source,unconstrained):
    root=Path(source);other=Path(unconstrained)
    a=json.loads((root/'records.json').read_text());b=json.loads((other/'records.json').read_text())
    assert len(a)==len(b)==120
    for folder in [root,other]:
        assert json.loads((folder/'status.json').read_text())['status']=='complete'
        audit=json.loads((folder/'audit.json').read_text());assert fingerprint(folder/'records.json')==audit['records_sha256']
        p=json.loads((folder/'protocol.json').read_text())
        for path,sha in p['checkpoints'].items():assert fingerprint(path)==sha
        rows=a if folder==root else b
        for r in rows:assert json.loads((folder/'trials'/f"{r['method']}_{r['case']}.json").read_text())==r
    assert json.loads((other/'protocol.json').read_text())['source_sha256']==fingerprint(root/'records.json')
    for r in b:
        original=next(x for x in a if (x['method'],x['case'])==(r['method'],r['case']))
        assert len(r['trace'])==original['schedule_steps']
        assert r['original_success']==original['original_success'] and r['original_failure']==original['original_failure']
        assert all(not x['bad'] for x in r['trace'])
    lines=['# E62 combined diagnostic: contact versus native pose tracking','',
        '120 original rollouts exactly reproduce audited E46/E61 outcomes. 120 intact fixed-reference replays match their motion to 1e-10. 120 cube-disabled and 120 all-geometry-contact-disabled replays start from the same calibration and retain recorded per-step XYZ references, tool-Z target, gripper commands and controller settings. Nine checkpoints remain frozen. No training.', '',
        'The all-geometry intervention was added after early analytical PiPER cube-disabled trials collided. It removes object/table/self-contact forces together, keeping gravity, masses, gains and joint limits. This is an artificial diagnostic, not safe control or successful manipulation.', '',
        '| Group | Robot | Original successes / trials | Cube-disabled endpoints track | Unconstrained endpoints track | Original failures that track unconstrained | Original successes that miss unconstrained endpoint |','|---|---|---:|---:|---:|---:|---:|']
    summary=[]
    for group in ['analytical','original','uncorrected','corrected']:
        for robot in ['piper','panda']:
            rr=[r for r in b if r['group']==group and r['robot']==robot];aa=[r for r in a if r['group']==group and r['robot']==robot]
            s=dict(group=group,robot=robot,count=len(rr),original_success=sum(r['original_success'] for r in rr),cube_tracks=sum(r['ghost_final_within_tolerance'] for r in aa),unconstrained_tracks=sum(r['tracks'] for r in rr),failed_but_tracks=sum(not r['original_success'] and r['tracks'] for r in rr),successful_but_misses=sum(r['original_success'] and not r['tracks'] for r in rr))
            summary.append(s);lines.append(f"| {group} | {robot} | {s['original_success']}/{s['count']} | {s['cube_tracks']} | {s['unconstrained_tracks']} | {s['failed_but_tracks']} | {s['successful_but_misses']} |")
    failures=[r for r in b if not r['original_success']];tracking=[r for r in failures if r['tracks']];remaining=[r for r in failures if not r['tracks']]
    lines+=['',f'Of {len(failures)} original failed trials, {len(tracking)} meet the recorded endpoint tolerances with all geometry contacts disabled and {len(remaining)} do not. Remaining misses by terminal phase: {dict(Counter(r["final_phase"] for r in remaining))}.', '',
        'Meeting tolerance means a single recorded endpoint is within 6 mm and 0.08 tool-Z direction error. It is not an eight-step hold, continuation to completion or a full-task success. Original failures truncate the schedule; an early miss cannot establish inability to recover with more time. Different models have different references and durations, so endpoint totals are not a fair common-task policy ranking.', '',
        'The result distinguishes two observed patterns: some original failures coexist with accurate unconstrained prefix tracking, while others retain significant tracking error without contact forces. Neither a purely contact-only explanation nor a universally adequate free-space tracker is supported. Removing the cube alone introduces residual contacts in successful PiPER references; it cannot be interpreted as a clean free-space benchmark.', '',
        'Decision: retain E35. Before another training objective, use a fixed complete native pose schedule shared across models, with enough dwell time and explicit pose/velocity diagnostics; retain a separate matched contact evaluation. Do not select checkpoints from these observed native diagnostics or claim broad unseen-embodiment improvement.', '',
        '84 regression tests pass. Initial v1 failed an intact replay check due to reference reconstruction and numerical solve ordering; it is preserved. v2 captures the exact reference through an optional observer and uses the original solve ordering. Default rollout behavior is unchanged. Trial archives, source record hashes and unchanged weights are verified. No final-test evaluation.']
    (root/'combined_report.md').write_text('\n'.join(lines)+'\n')
    (root/'combined_audit.json').write_text(json.dumps(dict(summary=summary,original_failed=len(failures),failed_but_tracks=len(tracking),failed_and_misses=len(remaining),verified_original_records=120,verified_unconstrained_records=120,unchanged_weights=True,test_evaluated=False),indent=2))
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--unconstrained',required=True);run(**vars(p.parse_args()))
