"""Audit all E41 development revisions without pooling different protocols."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .benchmark import fingerprint
from .real_arms import verify_assets


def run(assets,pilots,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);manifest=verify_assets(assets)
    lines=['# E41: native manipulation reliability development','',
        'Frozen E35 estimators; native PiPER/Panda mechanics and grippers. No new weight training or hardware evaluation.',
        'The first attempt caps Cartesian tracking error and is retained as a negative pilot. Revision 2 instead ramps the reference at 2 mm/control step while allowing full tracking correction. It uses 8 pose holds, stops failed phases, checks bilateral grasp, and adjusts grasp clearance. These are combined controller/fixture changes, not an isolated ablation.',
        'Revision 3 adds measured tool-to-object offset compensation at transport/lower phase entry, targeting object center 7 mm above resting height before release. It reuses the observed layouts; the paired expanded comparison isolates this change, but is development evidence. Endpoint IK screening covers nominal waypoints, not the subsequent object-offset-adjusted targets.',
        '','Each version is reported separately. Expanded placements were fixed before screening; endpoint feasibility does not filter on policy success. Final test remains reserved.','']
    summaries=[];audits=[];expanded={}
    for folder in map(Path,pilots):
        protocol=json.loads((folder/'protocol.json').read_text());cases=json.loads((folder/'cases.json').read_text())
        rows=json.loads((folder/'records.json').read_text());screen=json.loads((folder/'screening.json').read_text())
        if protocol['expanded']:expanded[protocol.get('revision',1)]=(folder,rows,cases)
        assert json.loads((folder/'status.json').read_text())['status']=='complete'
        assert protocol['commit']==manifest['commit'] and not protocol['test_evaluated']
        methods=protocol['methods'].split(',')
        assert len(rows)==len(cases)*len(methods)
        assert {(r['method'],r['case']) for r in rows}=={(m,i) for m in methods for i in range(len(cases))}
        for path,sha in protocol['checkpoints'].items():assert fingerprint(path)==sha
        for row in rows:
            assert row['robot']==cases[row['case']]['robot']
            assert row['steps']==len(row['probe'])+len(row['trace'])
            if row['success']:
                assert not row['collision'] and row['failure'] is None and row['grasped'] and row['lifted']
                assert len(row['phases'])==9 and all(p['reached'] for p in row['phases'])
        accepted=sum(s['accepted'] for s in screen)
        lines += [f'## {folder.name}','',f'Endpoint screening: {accepted}/{len(screen)} accepted. {len(rows)} complete rollouts.',
            '', '| Robot | Method | Success | Contacts | Mean total steps | Failure reasons |',
            '|---|---|---:|---:|---:|---|']
        for robot in ['piper','panda']:
            for method in methods:
                selected=[r for r in rows if r['robot']==robot and r['method']==method]
                if not selected:continue
                counts=Counter(r['failure'] for r in selected if not r['success'])
                summary=dict(version=folder.name,robot=robot,method=method,count=len(selected),
                    successes=sum(r['success'] for r in selected),contacts=sum(r['collision'] for r in selected),
                    mean_steps=float(np.mean([r['steps'] for r in selected])),failures=dict(counts))
                summaries.append(summary)
                reasons='; '.join(f'{k}: {v}' for k,v in sorted(counts.items())) or 'none'
                lines.append(f"| {robot} | {method} | {summary['successes']}/{len(selected)} | {summary['contacts']} | {summary['mean_steps']:.1f} | {reasons} |")
        lines += ['']
        audits.append(dict(folder=str(folder),records=len(rows),accepted=accepted,candidates=len(screen),
            records_sha256=fingerprint(folder/'records.json'),protocol_sha256=fingerprint(folder/'protocol.json'),
            source_sha256={p.name:fingerprint(p) for p in sorted((folder/'source').glob('*.py'))}))
    paired=[]
    if 2 in expanded and 3 in expanded:
        before,old,old_cases=expanded[2];after,new,new_cases=expanded[3]
        assert old_cases==new_cases
        lookup={(r['method'],r['case']):r for r in old}
        for row in new:
            previous=lookup[row['method'],row['case']]
            assert previous['probe']==row['probe']
            prefix=lambda r:[t for t in r['trace'] if t['phase'] in ('approach','descend','close','lift')]
            assert prefix(previous)==prefix(row)
        lines += ['## Matched expanded placement-offset comparison','',
            'All cases, calibration traces and pre-transport trajectories match exactly between revisions 2 and 3. Gains/losses below therefore concern the placement-offset intervention on observed development cases.',
            '', '| Robot | Method | Gained successes | Lost successes |', '|---|---|---:|---:|']
        for robot in ['piper','panda']:
            for method in methods:
                selected=[r for r in new if r['robot']==robot and r['method']==method]
                gains=sum(r['success'] and not lookup[method,r['case']]['success'] for r in selected)
                losses=sum(not r['success'] and lookup[method,r['case']]['success'] for r in selected)
                paired.append(dict(robot=robot,method=method,gains=gains,losses=losses))
                lines.append(f'| {robot} | {method} | {gains} | {losses} |')
        lines += ['']
    lines += ['The expanded set remains development evidence on two robot models and one cube/transfer direction. Fixed weights adapt measured motion statistics online; task sequencing and grasp checks are scripted. Different robot workspaces and geometry-specific grasp heights limit direct cross-robot comparisons. Early termination changes mean step counts; faster failures are not efficiency gains. No consistent neural superiority is established.','']
    (root/'report.md').write_text('\n'.join(lines));(root/'summary.json').write_text(json.dumps(summaries,indent=2))
    (root/'audit.json').write_text(json.dumps(dict(assets_verified=True,pilots=audits,paired=paired,test_evaluated=False),indent=2))
    if 2 in expanded and 3 in expanded:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
        labels=['Privileged','Analytical','Neural 0','Neural 1','Neural 2']
        for ax,robot in zip(axes,['piper','panda']):
            for revision,offset,color in [(2,-.19,'#8b9aa9'),(3,.19,'#197c80')]:
                rows=expanded[revision][1]
                scores=[sum(r['success'] for r in rows if r['robot']==robot and r['method']==m) for m in methods]
                bars=ax.bar(np.arange(len(methods))+offset,scores,width=.36,color=color,label=f'Revision {revision}')
                ax.bar_label(bars,padding=3)
            ax.set_title('PiPER (6 joints)' if robot=='piper' else 'Panda (7 joints)')
            ax.set_xticks(np.arange(len(methods)),labels,rotation=20,ha='right');ax.set_ylim(0,10)
            ax.set_yticks(range(10));ax.spines[['top','right']].set_visible(False)
        axes[0].set_ylabel('Successful tasks / 9');axes[1].legend(frameon=False)
        fig.suptitle('Observed development tasks: offset-aware placement comparison')
        fig.tight_layout();fig.savefig(root/'expanded_comparison.png',dpi=180);fig.savefig(root/'expanded_comparison.pdf');plt.close(fig)
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--pilots',nargs='+',required=True);p.add_argument('--output',required=True)
    run(**vars(p.parse_args()))
