"""E63 phase acquisition and hold stability from saved full trajectories."""
import argparse
import json
from pathlib import Path
import numpy as np
from .benchmark import fingerprint


def first_hold(trace):
    streak=0
    for index,r in enumerate(trace):
        streak=streak+1 if r['goal_error']<.006 and r['direction_error']<.08 else 0
        if streak>=8:return index+1
    return None


def run(output):
    root=Path(output);rows=json.loads((root/'records.json').read_text());schedules=json.loads((root/'schedules.json').read_text())
    audit=json.loads((root/'audit.json').read_text());assert fingerprint(root/'records.json')==audit['records_sha256']
    assert fingerprint(root/'schedules.json')==audit['schedules_sha256'];details=[]
    for r in rows:
        for p in r['phases']:
            trace=[x for x in r['trace'] if x['phase']==p['phase']]
            schedule=[x for x in schedules[r['case']] if x['phase']==p['phase']]
            first=first_hold(trace);assert not p['passes'] or first is not None
            moving=[i for i,s in enumerate(schedule) if np.linalg.norm(np.asarray(s['reference'])-s['goal'])>1e-10]
            details.append(dict(method=r['method'],group=r['group'],seed=r['seed'],robot=r['robot'],case=r['case'],phase=p['phase'],first_hold_step=first,
                budget=len(trace),terminal_pass=p['passes'],acquired_then_lost=bool(first is not None and not p['passes']),
                terminal_speed=float(np.mean([x['joint_speed'] for x in trace[-8:]])),
                max_goal_error=max(x['goal_error'] for x in trace),clipped_fraction=float(np.mean([x['clipped'] for x in trace])),
                moving_tracking_rmse=float(np.sqrt(np.mean([trace[i]['tracking_error']**2 for i in moving]))) if moving else None))
    lines=['# E63 phase acquisition and stability','',
        'First hold is the earliest eight consecutive samples meeting the goal tolerances. These are full fixed-budget phases, not early-terminated original manipulation. Velocity is diagnostic only and does not change pass criteria. Missing holds are kept, not assigned an artificial latency.', '',
        '| Group | Robot | Never acquired / phase holds | Acquired then lost terminal hold | Median first hold, attained only (steps) | Mean final joint-speed norm (rad/s) |','|---|---|---:|---:|---:|---:|']
    for group in ['analytical','original','uncorrected','corrected']:
        for robot in ['piper','panda']:
            rr=[r for r in details if r['group']==group and r['robot']==robot];times=[r['first_hold_step'] for r in rr if r['first_hold_step'] is not None]
            lines.append(f'| {group} | {robot} | {sum(r["first_hold_step"] is None for r in rr)}/{len(rr)} | {sum(r["acquired_then_lost"] for r in rr)} | {np.median(times):.1f} | {np.mean([r["terminal_speed"] for r in rr]):.5f} |')
    lines+=['','Aggregates combine different phase lengths/targets. Per-case, per-phase acquisition, clipping, moving-reference error and terminal velocity are retained in dynamics.json; no independence or statistical-significance claim. Compare these against full-schedule passes, not as a replacement success measure.']
    (root/'dynamics.json').write_text(json.dumps(details,indent=2));(root/'dynamics.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(**vars(p.parse_args()))
