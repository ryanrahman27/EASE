"""Summarize E69 telemetry without converting associations into causes."""
import json
from collections import Counter
import numpy as np
import mujoco
from pathlib import Path
from .arm_pick_place import PickPlace
from .mixed_sim import Arm,AXES
from .grasp_diagnostics import ROOT


def run():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==112
    summaries=[];details=[];names={};tasks=[]
    for folder in ('outputs/ease_arm_pick_place_v2','outputs/ease_arm_pick_place_unseen_v1'):
        p=Path(folder);AXES.update(json.loads((p/'axis_catalog.json').read_text()));tasks+=json.loads((p/'cases.json').read_text())
    for case,t in enumerate(tasks):
        env=PickPlace(Arm(**t['arm']),t['xy'],t['target']);names[case]={}
        for g in range(env.model.ngeom):
            name=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_GEOM,g)
            body=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_BODY,int(env.model.geom_bodyid[g]))
            names[case][name or str(g)]=name or f'{body}/geom{g}'
    for row in rows:
        full=json.loads((ROOT/'trials'/f"{row['case']}_{row['method']}.json").read_text());trace=full['trace']
        first=row['first_collision'];loss=row['first_sustained_contact_loss']
        row['first_collision_phase']=trace[first]['phase'] if first is not None else None
        row['first_collision_pairs']=trace[first]['bad'] if first is not None else []
        row['first_collision_named_pairs']=[[names[row['case']][x] for x in pair] for pair in row['first_collision_pairs']]
        row['collision_before_or_at_loss']=first is not None and (loss is None or first<=loss)
        row['loss_before_collision']=loss is not None and first is not None and loss<first
        if not row['success']: details.append(row)
    for group in ('learned','analytical'):
        for success in (True,False):
            rr=[r for r in rows if (r['method']=='analytical')==(group=='analytical') and r['success']==success]
            summaries.append(dict(group=group,success=success,n=len(rr),collisions=sum(r['collision'] for r in rr),
                stable_close=sum(r['stable_close'] for r in rr),carry_contact_loss=sum(r['first_sustained_contact_loss'] is not None for r in rr),
                collision_before_or_at_loss=sum(r['collision_before_or_at_loss'] for r in rr),
                median_max_relative_drift_mm=float(np.median([r['max_carry_relative_drift']*1000 for r in rr])) if rr else None))
    failed=[r for r in rows if not r['success'] and r['method']!='analytical']
    phases=Counter(r['first_collision_phase'] for r in failed if r['collision'])
    pairs=Counter(' + '.join(sorted(p)) for r in failed for p in r['first_collision_named_pairs'])
    overview=dict(strata=summaries,learned_failure_first_collision_phases=dict(phases),learned_failure_first_collision_pairs=dict(pairs),
        learned_noncollision_failures=[r for r in failed if not r['collision']],
        learned_loss_before_collision=sum(r['loss_before_collision'] for r in failed))
    (ROOT/'summary.json').write_text(json.dumps(overview,indent=2));(ROOT/'failures.json').write_text(json.dumps(details,indent=2))
    lines=['# E69: grasp and tracking diagnostics','',
        'Read-only instrumentation exactly reproduces all 112 E68 baseline trajectories: 16 procedural placements across seven frozen estimators. This is a diagnostic replay of observed development cases, not a new success benchmark. No weights, controller settings, mechanics or actions changed.','',
        '| Group | Outcome | Trials | Collision | Stable close | Carry contact loss | Collision before loss or no loss | Median max relative drift (mm) |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['group']} | {'success' if s['success'] else 'failure'} | {s['n']} | {s['collisions']} | {s['stable_close']} | {s['carry_contact_loss']} | {s['collision_before_or_at_loss']} | {s['median_max_relative_drift_mm']:.1f} |")
    lines+=['','Learned failure first-collision phases: '+', '.join(f'{k}: {v}' for k,v in phases.items())+'.','',
        'Stable close means at least eight consecutive bilateral endpoint contacts anywhere in close. Carry loss means three consecutive missing bilateral endpoint contacts in lift, transport or lower; it does not require prior acquisition. Relative drift is tool-frame object translation from the final close sample; it is not a direct physical-slip label. Contact loss during lowering may reflect valid table support.','',
        'Collision chronology and tool/object displacement narrow candidate explanations but cannot isolate causal motion-estimator error from contact dynamics, yaw, sequencer decisions or model error. One-step prediction residuals use measured joint displacement under contact, and transition tracking errors include the commanded waypoint jump. Sensor values are diagnostic simulator state, never supplied to the policy.','',
        'All input/checkpoint hashes remain unchanged. All 112 full baseline traces and calibration probes reproduce exactly; 97 EASE-FM tests pass. Threshold tests cover contact acquisition, sustained-loss indexing and rejection of brief contact gaps. No reserved final-test rollout or native tuning was performed.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(overview,indent=2))


if __name__=='__main__':run()
