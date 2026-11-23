"""Audit and summarize the fixed-checkpoint broad development evaluation."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import mujoco
from .benchmark import fingerprint
from .paired_data import register
from .mixed_sim import Arm, MixedReach


def read(path):return json.loads(Path(path).read_text())


def report(output):
    root=Path(output);m=register(root/'data');protocol=read(root/'protocol.json')
    assert read(root/'status.json')['status']=='complete'
    assert not m['goals']['test'] and not m['goals']['train'] and not m['episodes'] and not m['calibrations']
    assert m['prior_manifest_sha256']==protocol['prior_manifest_sha256']
    arms={a['seed']:Arm(**a) for a in m['arms']};goals={g['id']:g for g in m['goals']['validation']}
    assert len(arms)==64 and len(goals)==1024 and len(m['families'])==16
    sequences=[f['sequence'] for f in m['families']]
    assert len(set(sequences))==16 and not set(sequences)&set(m['excluded_axis_sequences'])
    assert sum(len(s)==6 for s in sequences)==sum(len(s)==7 for s in sequences)==8
    for f in m['families']:
        aa=[arms[b] for b in f['bodies']]
        assert aa[0].lengths==aa[2].lengths and aa[1].lengths==aa[3].lengths
        assert aa[0].masses==aa[2].masses and aa[1].masses==aa[3].masses
        assert aa[0].strength==aa[1].strength==1 and aa[0].damping==aa[1].damping==2
        assert aa[2].strength==aa[3].strength and aa[2].damping==aa[3].damping
        for home in f['homes']:
            for arm in aa:
                env=MixedReach(replace(arm,home=tuple(home)),synchronize=True);env.reset(0,[0,0,.9],spread=0)
                jac=np.zeros((3,arm.dof));mujoco.mj_jacSite(env.model,env.data,jac,None,0)
                assert np.linalg.matrix_rank(jac,tol=1e-7)==3
    for g in goals.values():
        arm=replace(arms[g['pair']],home=tuple(g['home']))
        env=MixedReach(arm,synchronize=True);obs=env.reset(0,g['xyz'],spread=0)
        distance=np.linalg.norm(obs['task'][:3].astype(float)-g['xyz'])
        assert .06-1e-6<=distance<=.18+1e-6
        env.data.qpos[:]=g['joint_target'];mujoco.mj_forward(env.model,env.data)
        assert not env.data.ncon and max(abs(x) for x in g['joint_target'])<2.2
        np.testing.assert_allclose(env.data.site_xpos[0],g['xyz'],atol=1e-10)
    expected={(g['pair'],g['id']) for g in goals.values()}
    rows={};summary=[];evidence={};family_summary=[]
    methods=['privileged_reference','fixed_probe_analytical','analytical_cold','neural_0','neural_1','neural_2']
    def metrics(records):
        return dict(successes=sum(r['success'] for r in records),trials=len(records),collisions=sum(r['collision'] for r in records),
            mean_steps=float(np.mean([r['steps'] for r in records])),
            success_within={str(b):sum(r['success'] and r['steps']<=b for r in records) for b in (10,25,50,100,150)})
    for method in methods:
        path=root/f'{method}.json';records=read(path);rows[method]=records;evidence[str(path)]=fingerprint(path)
        assert len(records)==1024 and {(r['body_seed'],r['goal_id']) for r in records}==expected
        assert all(r['condition']==goals[r['goal_id']]['condition'] and r['dof']==arms[r['body_seed']].dof for r in records)
        assert all(r['success'] or r['steps']==150 for r in records)
        assert all(not r['success'] or (r['error']<.03 and not r['collision']) for r in records)
        cells={}
        for condition in sorted({g['condition'] for g in goals.values()}):
            selected=[r for r in records if r['condition']==condition];assert len(selected)==128
            cells[condition]=metrics(selected)
        summary.append(dict(method=method,**metrics(records),by_dof={str(n):metrics([r for r in records if r['dof']==n]) for n in (6,7)},conditions=cells))
        for f in m['families']:
            selected=[r for r in records if r['body_seed'] in f['bodies']];assert len(selected)==64
            family_summary.append(dict(method=method,sequence=f['sequence'],dof=f['dof'],**metrics(selected)))
    for e in protocol['checkpoints']:
        folder=Path(e['folder']);meta=read(folder/'training.json')
        assert fingerprint(folder/'policy.pt')==meta['policy_sha256']==e['policy_sha256']
        assert meta['manifest_sha256']==protocol['prior_manifest_sha256']
    baseline={(r['body_seed'],r['goal_id']):r for r in rows['fixed_probe_analytical']}
    paired=[]
    for seed in range(3):
        records=rows[f'neural_{seed}']
        paired.append(dict(seed=seed,neural_only=sum(r['success'] and not baseline[r['body_seed'],r['goal_id']]['success'] for r in records),
            analytical_only=sum(not r['success'] and baseline[r['body_seed'],r['goal_id']]['success'] for r in records),
            both_success=sum(r['success'] and baseline[r['body_seed'],r['goal_id']]['success'] for r in records)))
    structural=read(root/'data'/'structural_audit.json')
    exclusions={k:sum(r['kind']==k and not r['accepted'] for r in structural) for k in ('axis','home','goal')}
    (root/'summary.json').write_text(json.dumps(dict(overall=summary,families=family_summary,paired=paired,structural_exclusions=exclusions),indent=2))
    (root/'audit.json').write_text(json.dumps(dict(manifest_sha256=fingerprint(root/'data'/'manifest.json'),
        checkpoints_unchanged=3,families=16,bodies=64,homes_verified=128,reachable_endpoints_verified=1024,
        policy_and_reference_records=6144,evidence_sha256=evidence,test_evaluated=False,
        report_source_sha256=fingerprint(Path(__file__))),indent=2))
    lines=['# E33: broader embodiment development evaluation','',
        'All three motion-Transformer checkpoints are frozen. This separate development suite contains 16 new axis arrangements (eight per joint count), four crossed geometry/dynamics variants, two starting postures, and eight goals per body/posture: 64 physical bodies and 1,024 cases. Every arrangement is disjoint from the registered prior families, including reserved axes. No final-test policy evaluation occurred.','',
        'Axes may begin with X, Y, or Z. Nominal link multipliers are 0.8–1.2 and masses 0.7–1.3; geometry-shift multipliers are 0.55–1.55 with masses 0.5–1.8. Nominal actuator strength/damping are 1/2; shifted strength is sampled from 0.3–2.0 and damping from 0.3–5.0. Each family shares two joint homes and target joint offsets across its four variants. Absolute Cartesian goals differ with geometry; this is not an isolated Cartesian-observation ablation.','',
        'Goals are forward-kinematic endpoints, 6–18 cm from the reset position, at collision-free joint configurations within ±2.2 rad. Initial configurations must be contact-free and rank 3. Structural filtering does not establish collision-free paths or closed-loop solvability. No expert or policy success filtering was applied. The privileged expert is an informative reference, not a guaranteed upper bound.','',
        'Version 1 stopped during geometry generation on an intrinsically rank-deficient axis family, before controller evaluation. It is preserved. Version 2 adds a recorded generic-pose rank check before accepting a sequence; no controller outcomes informed this change.','',
        '## Overall results','',
        'All controllers start without a separate calibration. The analytical fixed probe is +0.01 rad on every joint at task step one, charged to its budget. Horizon, action limit, 3 cm / five-step success rule, and collision rule are unchanged. Failures consume 150 steps.','',
        '| Controller | Success / 1024 | Six-joint / 512 | Seven-joint / 512 | Collisions | Mean steps | Success by 25 | Success by 50 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summary:
        lines.append(f"| {s['method']} | {s['successes']} | {s['by_dof']['6']['successes']} | {s['by_dof']['7']['successes']} | {s['collisions']} | {s['mean_steps']:.2f} | {s['success_within']['25']} | {s['success_within']['50']} |")
    lines+=['','## Crossed conditions','','Each cell is successes /128 (64 per joint count).','',
        '| Condition | Analytical fixed probe | Neural 0 | Neural 1 | Neural 2 |','|---|---:|---:|---:|---:|']
    lookup={s['method']:s for s in summary}
    for c in sorted(lookup['neural_0']['conditions']):
        lines.append('| '+c+' | '+' | '.join(str(lookup[k]['conditions'][c]['successes']) for k in ('fixed_probe_analytical','neural_0','neural_1','neural_2'))+' |')
    lines+=['','## Paired task outcomes','','| Model seed | Neural only | Analytical only | Both successful |','|---:|---:|---:|---:|']
    for p in paired:lines.append(f"| {p['seed']} | {p['neural_only']} | {p['analytical_only']} | {p['both_success']} |")
    lines+=['','## Per-family outcomes','','Each cell is successes /64.','',
        '| Axis sequence | Joint count | Analytical fixed probe | Neural 0 | Neural 1 | Neural 2 |','|---|---:|---:|---:|---:|---:|']
    for f in m['families']:
        values=[next(r['successes'] for r in family_summary if r['method']==method and r['sequence']==f['sequence']) for method in ('fixed_probe_analytical','neural_0','neural_1','neural_2')]
        lines.append(f"| {f['sequence']} | {f['dof']} | "+' | '.join(str(v) for v in values)+' |')
    lines+=['',f'Structural exclusions: {exclusions}. All rejected candidates are retained in the structural audit. Independent verification reproduces all 1,024 target endpoints, checks 128 reset configurations, and validates crossed physical parameters, nonoverlapping axes, case keys, outcome consistency, and unchanged checkpoint hashes. All 6,144 policy/reference records are hashed.','',
        'This tests transfer to a broader but still procedural family of six/seven-joint serial arms under clean state observations and position-only reaching. Models, gains, and budgets were not tuned on these results. Repeated goals and physical variants are clustered by 16 axis families; training-seed repeats are not independent morphology samples. No broad real-world, manipulation, noise-robustness, or learned-superiority claim should be inferred from pooled task counts alone.','']
    (root/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
    for ax,n in zip(axes,(6,7)):
        ff=[f for f in m['families'] if f['dof']==n];x=np.arange(len(ff))
        for i,method in enumerate(('fixed_probe_analytical','neural_0','neural_1','neural_2')):
            vals=[next(r['successes']/64*100 for r in family_summary if r['method']==method and r['sequence']==f['sequence']) for f in ff]
            ax.plot(x,vals,marker='o',label=method.replace('_',' '),linewidth=1.4)
        ax.set(title=f'{n}-joint unseen axis arrangements',ylim=(0,105),xticks=x,xticklabels=[f['sequence'] for f in ff])
        ax.tick_params(axis='x',rotation=55);ax.grid(alpha=.2)
    axes[0].set_ylabel('Success (%)');axes[1].legend(fontsize=8,loc='lower left')
    fig.suptitle('E33 frozen checkpoints: broader kinematics, geometry, and dynamics')
    fig.tight_layout();fig.savefig(root/'broad_transfer.png',dpi=160);plt.close(fig)
    print('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    report(**vars(parser.parse_args()))
