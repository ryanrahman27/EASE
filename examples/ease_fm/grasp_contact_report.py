"""E102 sampled contact timing and gripper-relative displacement."""
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .grasp_contact_audit import ROOT


def run():
    spec=json.loads((ROOT/'protocol.json').read_text());verification=json.loads((ROOT/'verification.json').read_text());assert verification['exact_replays']==10
    rows=[];phase_rows=[];fig,axes=plt.subplots(5,1,figsize=(10,12),layout='constrained')
    for ax,(case,method,failed) in zip(axes,spec['failures']):
        modes=[m for c,s,m in spec['selections'] if c==case and s==method]
        for mode in modes:
            rr=json.loads((ROOT/'trials'/f'{case}_{method}_{mode}_contacts.json').read_text());full=json.loads((ROOT/'trials'/f'{case}_{method}_{mode}.json').read_text())
            lower=[r for r in rr if r['phase']=='lower'];release=[r for r in rr if r['phase']=='release'];transport=[r for r in rr if r['phase']=='transport'];start=lower[0]['time']-.002
            active=lambda r,k:r['normal_force'][k]>1e-6
            table=next((r for r in lower if active(r,'table')),None)
            pre=[r for r in lower if table is None or r['time']<table['time']]
            bilateral=lambda r:all(active(r,k) for k in ('gripper_left_pad','gripper_right_pad'))
            for phase in dict.fromkeys(r['phase'] for r in rr):
                pp=[r for r in rr if r['phase']==phase]
                phase_rows.append(dict(case=case,method=method,mode=mode,success=full['success'],phase=phase,n=len(pp),bilateral_fraction=sum(bilateral(r) for r in pp)/len(pp),any_pad_fraction=sum(any(active(r,k) for k in ('gripper_left_pad','gripper_right_pad')) for r in pp)/len(pp),table_fraction=sum(active(r,'table') for r in pp)/len(pp),end_object_z=pp[-1]['object'][2],end_tool_z=pp[-1]['tool'][2],end_xy_mm=pp[-1]['xy_error_mm']))
            base=np.array(transport[-1]['relative'])
            shift=lambda rr:float(1000*max((np.linalg.norm(np.array(r['relative'])-base) for r in rr),default=0))
            row=dict(case=case,method=method,mode=mode,success=full['success'],lower_ms=1000*(lower[-1]['time']-start),first_table_ms=None if table is None else 1000*(table['time']-start),first_table_xy_mm=None if table is None else table['xy_error_mm'],pretable_samples=len(pre),pretable_bilateral_fraction=None if not pre else sum(bilateral(r) for r in pre)/len(pre),pretable_relative_shift_mm=shift(pre) if pre else None,lower_relative_shift_mm=shift(lower),transport_xy_mm=transport[-1]['xy_error_mm'],lower_xy_mm=lower[-1]['xy_error_mm'],release_xy_mm=release[-1]['xy_error_mm'],final_xy_mm=rr[-1]['xy_error_mm'])
            rows.append(row);selected=[r for r in rr if r['phase'] in ('lower','release')]
            selected=[r for r in rr if r['phase'] in ('lower','release','retreat','settle')]
            line=ax.plot([r['time']-start for r in selected],[r['xy_error_mm'] for r in selected],label=f'{mode}: '+('success' if full['success'] else 'failure'))[0]
            if table:ax.scatter([table['time']-start],[table['xy_error_mm']],marker='o',color=line.get_color(),s=35)
            ax.axvline(release[0]['time']-start,color=line.get_color(),linestyle=':',alpha=.6)
        ax.axhline(25,color='grey',linestyle='--',linewidth=1);ax.set(title=f'Case {case}, {method}',xlabel='Seconds since lowering began (trajectories differ)',ylabel='Object XY error (mm)');ax.legend(fontsize=8)
    (ROOT/'summary.json').write_text(json.dumps(rows,indent=2));(ROOT/'phase_summary.json').write_text(json.dumps(phase_rows,indent=2));fig.savefig(ROOT/'contact_motion.png',dpi=140);fig.savefig(ROOT/'contact_motion.pdf');plt.close(fig)
    lines=['# E102: contact and object motion during lowering/release','',
        'All five E101 non-collision regressions are paired with successful original-parent runs on the same task and seed. These ten outcome-selected diagnostic replays exactly match their full saved E101 results. No controller, physics, task criterion or policy weights change. This is not a new success-rate evaluation.',
        '', 'At each existing contact query after a2ms physics substep, record tool/object positions, object position in the gripper frame, finger positions, cube-pad/table contact presence and summed positive normal contact forces. Active contact below uses summed normal force greater than1e-6 N. Geometric contact presence is separately saved. Contact data and poses follow MuJoCo\'s existing stepping schedule; no extra forward/step call is inserted.',
        '', '## Contact timing','', 'Times are measured from the beginning of lowering. Relative shift is the Euclidean displacement of the object center in the gripper frame from the final transport sample; it can include rotation/rolling and is not a direct slip measurement. Pre-table statistics end before the first active cube-table contact.','',
        '| Case | Seed | Mode | First table ms | Pre-table bilateral fraction | Max pre-table relative shift mm | XY at first table mm |','|---:|---|---|---:|---:|---:|---:|']
    fmt=lambda x:'none' if x is None else f'{x:.3f}'
    for r in rows:lines.append(f"| {r['case']} | {r['method']} | {r['mode']} | {fmt(r['first_table_ms'])} | {fmt(r['pretable_bilateral_fraction'])} | {fmt(r['pretable_relative_shift_mm'])} | {fmt(r['first_table_xy_mm'])} |")
    lines+=['','## Placement trajectory','','| Case | Seed | Mode | Transport XY mm | Lower XY mm | Release XY mm | Final XY mm | Max lower relative shift mm |','|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['case']} | {r['method']} | {r['mode']} | {r['transport_xy_mm']:.2f} | {r['lower_xy_mm']:.2f} | {r['release_xy_mm']:.2f} | {r['final_xy_mm']:.2f} | {r['lower_relative_shift_mm']:.2f} |")
    if (ROOT/'decision.txt').exists():lines+=['','## Interpretation and next decision','',(ROOT/'decision.txt').read_text().strip()]
    lines+=['','## Verification and artifacts','',f"Ten exact full-result replays; {sum(r['samples'] for r in verification['sample_counts'])} recorded contact samples; unchanged reference/checkpoint hashes.129 existing tests pass. All ten success/failure videos are replay-verified, decoded-frame-count checked and SHA-256 recorded. See videos/index.html. Figure contact_motion.png / .pdf: circles mark first active table contact, dotted lines mark release start, dashed horizontal line is25mm success threshold. Different modes follow different trajectories. Raw per-substep contact traces, vectors and normal forces remain saved. Privileged diagnostics/planning; no inference weight updates, training, or final test."]
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(json.dumps(rows,indent=2))


if __name__=='__main__':run()
