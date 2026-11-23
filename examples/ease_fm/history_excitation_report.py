"""E94 spectral summary, scientific figure and reused execution evidence."""
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .history_excitation import ROOT,prior
from .benchmark import fingerprint


def run():
    h=json.loads((ROOT/'handoffs.json').read_text());audit=json.loads((ROOT/'audit.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text())
    assert audit['exact_replays']==46 and len(h)==96
    for p,v in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==v
    a=[r for r in h if r['applied']];table=[]
    for dof in (6,7):
        rr=[r for r in a if r['dof']==dof]
        for kind in ('long','recent'):
            table.append(dict(dof=dof,kind=kind,trials=len(rr),**{k:float(np.median([r[kind][k] for r in rr])) for k in ('energy','effective_rank','above_ridge','largest_fraction')}))
    phase=[]
    for p in sorted((ROOT/'profiles').glob('*.json')):
        r=json.loads(p.read_text())
        for name in dict.fromkeys(t['phase'] for t in r['profiles']):
            samples=[t for t in r['profiles'] if t['phase']==name]
            for kind in ('long','recent'):
                phase.append(dict(case=r['case'],method=r['method'],mode=r['mode'],group=r['group'],phase=name,kind=kind,
                    median_above_ridge=float(np.median([t[kind]['above_ridge'] for t in samples])),median_effective_rank=float(np.median([t[kind]['effective_rank'] for t in samples])),median_energy=float(np.median([t[kind]['energy'] for t in samples]))))
    (ROOT/'summary.json').write_text(json.dumps(dict(handoff_table=table,one_direction=sum(r['recent']['above_ridge']==1 for r in a),phase_summaries=phase),indent=2))
    changes=json.loads((prior.ROOT/'summary.json').read_text())['changes']
    loss7=next(r for r in changes if r['change']=='loss' and r['dof']==7)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for dof,color in ((6,'#277da8'),(7,'#dc7a24')):
        rr=[r for r in a if r['dof']==dof]
        axes[0,0].scatter([r['long']['above_ridge'] for r in rr],[r['recent']['above_ridge'] for r in rr],s=42,alpha=.35,label=f'{dof} joints',color=color)
    axes[0,0].plot([0,7],[0,7],'k--',lw=1);axes[0,0].set(xlabel='Long-history directions above ridge',ylabel='Recent32 directions above ridge',title='78 route-assisted handoffs',xlim=(-.3,7.3),ylim=(-.3,7.3));axes[0,0].legend()
    axes[0,1].boxplot([[r['recent']['energy']/max(r['long']['energy'],1e-30) for r in a if r['dof']==d] for d in (6,7)],tick_labels=['6 joints','7 joints'])
    axes[0,1].set(ylabel='Recent / long measured motion energy',title='Same handoff states; ridge excluded')
    for ax,case,method in ((axes[1,0],2,'parent_1'),(axes[1,1],loss7['case'],loss7['method'])):
        r=json.loads((ROOT/'profiles'/f'{case}_{method}_window32.json').read_text());samples=r['profiles']
        for kind,label in (('long','Long shadow'),('recent','Recent32')):ax.plot([t['step']-32 for t in samples],[t[kind]['above_ridge'] for t in samples],label=label,lw=1.3)
        close=next(t['step']-32 for t in samples if t['phase']=='close');ax.axvline(close,color='black',ls=':',label='Post-descent handoff')
        ax.set(title=f'New failure: case {case}, {method}',xlabel='Task control step',ylabel='Directions above ridge',ylim=(-.3,7.3));ax.legend(fontsize=8)
    fig.suptitle('E94: measured excitation, not a validated failure threshold',fontsize=14)
    fig.savefig(ROOT/'excitation.png',dpi=180);fig.savefig(ROOT/'excitation.pdf');plt.close(fig)
    out=ROOT/'videos';out.mkdir(exist_ok=True);old=prior.ROOT/'videos';manifest=json.loads((old/'manifest.json').read_text());videos=[]
    html='<html><head><meta charset="utf-8"><title>E94 excitation audit</title></head><body style="background:#111;color:#eee;font:16px sans-serif;margin:30px"><h1>Excitation audit: unchanged E93 executions</h1><p>No new controller behavior. These are reused E93 success/failure videos with verified hashes; E94 diagnostic replays are exact. Example selection remains outcome-informed.</p><img width="1000" src="../excitation.png">'
    for v in manifest['videos']:
        p=old/v['file'];assert fingerprint(p)==v['sha256'];videos.append(dict(file=str(p),sha256=v['sha256'],reused=True))
        html+=f'<h2>{v["file"]}</h2><video controls width="640" src="../../ease_local_history_v1/videos/{v["file"]}"></video>'
    (out/'manifest.json').write_text(json.dumps(dict(scope='Reused unchanged E93 execution videos, not new experiments.',videos=videos),indent=2));(out/'index.html').write_text(html+'</body></html>',encoding='utf-8')
    lines=['# E94: measured excitation retained by recent history','',
        'This read-only audit covers all96 saved E93 handoffs, including78 admitted-route pairs where history changes. Temporal diagnostics exactly replay all21 changed-outcome pairs and the earliest retained-success control per DOF, in both modes (46 tasks). Per-step statistics are recorded after the measured update. Long shadow and recent32 statistics are computed on the same executed trajectory; the shadow does not affect actions. No controller modification or new policy-success claim is made.',
        '', 'Subtract the1e-6 isotropic ridge before computing measured Gram spectra. Directions above ridge count measured eigenvalues greater than1e-6, tied to the existing regularizer rather than fitted to outcomes. Effective rank is squared total eigenvalue mass divided by sum of squared eigenvalues. These metrics describe excitation; they are not validated success/failure thresholds.',
        '', '| DOF | History | Handoffs | Median measured energy | Median effective rank | Median directions above ridge | Median largest-direction fraction |','|---|---|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['dof']} | {r['kind']} | {r['trials']} | {r['energy']:.6g} | {r['effective_rank']:.4f} | {r['above_ridge']:.1f} | {r['largest_fraction']:.6f} |")
    lines+=['','## Findings and decision','',
        'Recent32 contains only one direction above the ridge at66/78 admitted-route handoffs (24/30 six-joint and42/48 seven-joint). Median effective rank is approximately1 in both groups; the older weighted history has median effective rank1.205/1.251 and six directions above ridge in both groups. The weak directions are small relative to the dominant one, so longer history is not uniformly well-conditioned either. Recent history retains median48.4%/47.9% of long-history measured energy for six/seven joints.',
        '', 'This pattern is consistent with a straight joint-space descent exciting predominantly one joint combination. Sample count alone does not imply coverage of all joint directions. Windowing removes weaker independent directions alongside older potentially nonlocal data. Since this pattern occurs in both DOFs while E93 effects differ, it cannot alone explain every regression or justify a DOF-specific rule.',
        '', 'Next investigate an estimator-side diagnostic that preserves directional coverage while addressing locality. A controlled option is a small, fixed, separately budgeted local excitation probe at the handoff, with every contact and task cost retained, followed by frozen estimation. Predeclare its bounds and retention criteria before rollout; compare probe/no-probe under matched budgets and include old successes. No probe or replacement controller has been implemented or promoted in E94. Merely changing window length would not address the observed directional deficiency.',
        '', 'Verification: all46 full replays match E93 result dictionaries exactly, with96 saved handoff pairs audited and unchanged inputs/checkpoints. All119 tests pass, including zero-motion/ridge exclusion and known rank-one/isotropic cases. Existing six E93 executed videos are referenced with verified hashes rather than relabeled as new successes. No training or reserved-test evaluation occurred.',
        '', '## Artifacts','',
        '- Figure: `outputs/ease_history_excitation_v1/excitation.png` and `.pdf`.',
        '- Per-step spectra: `profiles/`; all-handoff statistics: `handoffs.json`; aggregate phase summaries: `summary.json`.',
        '- Gallery with reused success/failure videos: `outputs/ease_history_excitation_v1/videos/index.html`.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print(json.dumps(table,indent=2))


if __name__=='__main__':run()
