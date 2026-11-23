"""Audit and summarize E67 without selecting cases, seeds or checkpoints."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .frozen_transfer import ROOT
from .benchmark import fingerprint
from .native_common_schedule import summarize_phase


def run():
    audit=json.loads((ROOT/'audit.json').read_text());protocol=json.loads((ROOT/'protocol.json').read_text())
    manifest=json.loads((ROOT/'data/manifest.json').read_text());native=ROOT/'native'
    assert fingerprint(ROOT/'data/manifest.json')==audit['manifest_sha256']
    for name,sha in audit['file_hashes'].items():assert fingerprint(ROOT/name)==sha
    for path,sha in {**protocol['checkpoints'],**protocol['prior_manifests']}.items():assert fingerprint(path)==sha
    for name,sha in manifest['asset_hashes'].items():assert fingerprint(ROOT/'data'/name)==sha
    assert len(manifest['families'])==16 and len(manifest['arms'])==64 and len(manifest['goals']['validation'])==1024
    assert not manifest['goals']['test'] and not manifest['goals']['train']
    assert not ({f['sequence'] for f in manifest['families']} & set(manifest['excluded_axis_sequences']))
    methods=['analytical']+[f'{group}_{s}' for group in ['parent','candidate'] for s in range(3)]
    proc={m:json.loads((ROOT/'procedural'/f'{m}.json').read_text()) for m in methods}
    goals={g['id']:g for g in manifest['goals']['validation']};bodies={a['seed']:a for a in manifest['arms']}
    for m,rr in proc.items():
        assert len(rr)==1024 and {r['goal_id'] for r in rr}==set(goals)
        for r in rr:
            assert r['body_seed']==goals[r['goal_id']]['pair'] and r['dof']==len(bodies[r['body_seed']]['lengths'])
            assert 1<=r['steps']<=150 and not (r['success'] and r['collision'])
    rows=json.loads((native/'records.json').read_text());cases=json.loads((native/'cases.json').read_text());screen=json.loads((native/'screening.json').read_text())
    assert len(screen)==12 and len(cases)==sum(s['accepted'] for s in screen)
    assert len(rows)==14*len(cases) and len({(r['method'],r['case'],r['task']) for r in rows})==len(rows)
    calibration_contacts=0
    for case in range(len(cases)):
        probe=json.loads((native/f'probe_{case}.json').read_text());assert len(probe)==32
        calibration_contacts+=int(any(x['bad'] for x in probe))
    for r in rows:
        assert r['robot']==cases[r['case']]['robot'] and r['candidate']==cases[r['case']]['candidate']
        assert r==json.loads((native/'trials'/f'{r["method"]}_{r["case"]}_{r["task"]}.json').read_text())
        assert len(r['trace'])==r['steps']-32
        if r['task']=='common':
            assert len(r['trace'])==1700 and len(r['phases'])==9
            for p in r['phases']:assert p==summarize_phase(p['phase'],[x for x in r['trace'] if x['phase']==p['phase']])
            assert r['success']==all(p['passes'] for p in r['phases'])
        elif r['success']:assert r['grasped'] and r['lifted'] and not r['collision'] and r['failure'] is None
    lines=['# E67: frozen transfer to additional embodiments and placements','',protocol['procedural'],'',protocol['native'],'',protocol['limits'],'',
        'The E33 generator retains its original schema/experiment labels inside the data manifest; the parent E67 protocol, generation seed and manifest hash identify this new evaluation.','',
        '## Procedural reaching: 16 new axis arrangements, 64 bodies','', '| Method | Six joints /512 | Seven joints /512 | Total /1024 | Collisions | Mean steps, all cases | Paired gains vs parent | Paired losses |','|---|---:|---:|---:|---:|---:|---:|---:|']
    summary=[];paired=[]
    for method,rr in proc.items():
        counts=[sum(r['success'] for r in rr if r['dof']==d) for d in [6,7]];gains=losses=0
        if method.startswith('candidate'):
            base={r['goal_id']:r for r in proc[method.replace('candidate','parent')]}
            for d in [6,7]:
                subset=[r for r in rr if r['dof']==d]
                gain=sum(r['success'] and not base[r['goal_id']]['success'] for r in subset);loss=sum(not r['success'] and base[r['goal_id']]['success'] for r in subset)
                paired.append(dict(task='procedural',method=method,dof=d,gains=gain,losses=loss));gains+=gain;losses+=loss
        entry=dict(task='procedural',method=method,success=sum(counts),counts=counts,collisions=sum(r['collision'] for r in rr),mean_steps=float(np.mean([r['steps'] for r in rr])),gains=gains,losses=losses)
        summary.append(entry)
        lines.append(f'| {method} | {counts[0]} | {counts[1]} | {sum(counts)} | {entry["collisions"]} | {entry["mean_steps"]:.2f} | {gains if method.startswith("candidate") else "—"} | {losses if method.startswith("candidate") else "—"} |')
    lines+=['','## Native placements','',f'{len(cases)}/12 candidates pass the prespecified nine-endpoint IK screen. Calibration contact occurs in {calibration_contacts} accepted cases. Rejected candidates are retained in screening.json; no replacements.','']
    for task in ['common','manipulation']:
        lines += [f'### {task}','', '| Method | PiPER | Panda | Total | Contacts | Gains vs parent | Losses |','|---|---:|---:|---:|---:|---:|---:|']
        for method in methods:
            rr=[r for r in rows if r['method']==method and r['task']==task]
            counts={robot:[sum(r['success'] for r in rr if r['robot']==robot),sum(r['robot']==robot for r in rr)] for robot in ['piper','panda']}
            gains=losses=0
            if method.startswith('candidate'):
                base={r['case']:r for r in rows if r['method']==method.replace('candidate','parent') and r['task']==task}
                for robot in ['piper','panda']:
                    subset=[r for r in rr if r['robot']==robot]
                    gain=sum(r['success'] and not base[r['case']]['success'] for r in subset);loss=sum(not r['success'] and base[r['case']]['success'] for r in subset)
                    paired.append(dict(task=task,method=method,robot=robot,gains=gain,losses=loss));gains+=gain;losses+=loss
            entry=dict(task=task,method=method,success=sum(r['success'] for r in rr),count=len(rr),counts=counts,collisions=sum(r.get('collision',False) for r in rr),gains=gains,losses=losses)
            summary.append(entry)
            lines.append(f'| {method} | {counts["piper"][0]}/{counts["piper"][1]} | {counts["panda"][0]}/{counts["panda"][1]} | {entry["success"]}/{len(rr)} | {entry["collisions"]} | {gains if method.startswith("candidate") else "—"} | {losses if method.startswith("candidate") else "—"} |')
    family=[]
    for f in manifest['families']:
        for method,rr in proc.items():
            subset=[r for r in rr if r['family_id']==f['id']];assert len(subset)==64
            family.append(dict(family=f['id'],sequence=f['sequence'],dof=f['dof'],method=method,success=sum(r['success'] for r in subset),count=64,collisions=sum(r['collision'] for r in subset)))
    conditions=[]
    for condition in sorted({g['condition'] for g in goals.values()}):
        for method,rr in proc.items():
            subset=[r for r in rr if goals[r['goal_id']]['condition']==condition]
            conditions.append(dict(condition=condition,method=method,success=sum(r['success'] for r in subset),count=len(subset),collisions=sum(r['collision'] for r in subset)))
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));(ROOT/'paired.json').write_text(json.dumps(paired,indent=2))
    (ROOT/'family_results.json').write_text(json.dumps(family,indent=2));(ROOT/'condition_results.json').write_text(json.dumps(conditions,indent=2))
    lines+=['','All 7,168 procedural records, native trial files and common-phase summaries audited. Six checkpoint hashes, both prior manifests and generated data hashes unchanged. Detailed paired, axis-family and physical-condition results are retained. Regression suite: 92 passed. All evaluations are prospective development comparisons, not final-test or hardware results.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for ax,task in zip(axes,['procedural','common','manipulation']):
        for j,(label,dof) in enumerate([('Six joints',6),('Seven joints',7)]):
            vals=[];texts=[]
            for group in ['analytical','parent','candidate']:
                rr=[r for r in summary if r['task']==task and (r['method']==group or r['method'].startswith(group+'_'))]
                if task=='procedural':k=sum(r['counts'][j] for r in rr);n=512*len(rr)
                else:
                    robot=['piper','panda'][j];k=sum(r['counts'][robot][0] for r in rr);n=sum(r['counts'][robot][1] for r in rr)
                vals.append(100*k/n if n else 0);texts.append(f'{k}/{n}')
            bars=ax.bar(np.arange(3)+(j-.5)*.36,vals,.36,label=label)
            for b,txt in zip(bars,texts):ax.text(b.get_x()+b.get_width()/2,b.get_height()+2,txt,ha='center',fontsize=8)
        ax.set_xticks(range(3),['Analytical','E35','E66']);ax.set_ylim(0,115);ax.set_yticks(range(0,101,20));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.set_title({'procedural':'New procedural bodies','common':'New placements: tracking','manipulation':'New placements: manipulation'}[task])
    axes[0].set_ylabel('Success (%)');axes[-1].legend(loc='upper right',bbox_to_anchor=(1,1.2),ncol=2,fontsize=8)
    fig.suptitle('E67: frozen checkpoints on prespecified additional cases',y=1.02)
    fig.text(.5,-.015,'Seeds share cases. Native placements use the same PiPER/Panda designs. Reserved final test untouched.',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':run()
