"""Audited E179 paired fitting report and video selection."""
from pathlib import Path
import shutil
import numpy as np
import mujoco
from .metric_readiness import ROOT,REFERENCE,CONDITIONS
from .abstaining_task_value_gate import read,save,fingerprint

def run():
    metrics={};details={};selected={0,26,43,62,67};checks=0;tasks=read(ROOT/'cases.json')
    budgets=dict(approach=80,descend=100,probe=32,close=40,lift=100,transport=100,lower=100,release=40,retreat=80,settle=40)
    for condition in CONDITIONS:
        root=ROOT/condition;rows=read(root/'records.json');assert len(rows)==48
        assert read(root/'verification.json')['episodes']==48
        result=[]
        for r in rows:
            case=r['case'];stem=f'{case}_e120_ensemble_frozen_bias';t=read(root/'trials'/f'{stem}.json');logs=read(root/'clearance'/f'{stem}.json');states=np.load(root/'states'/f'{stem}.npz')['states']
            baseline=read(ROOT/'baseline/trials'/f'{stem}.json');bl=read(ROOT/'baseline/clearance'/f'{stem}.json')
            assert t['calibration']==baseline['calibration'] and t['initial_estimator']==baseline['initial_estimator']
            np.testing.assert_array_equal(logs[0]['full_qpos_before'],bl[0]['full_qpos_before']);np.testing.assert_array_equal(logs[0]['full_qvel_before'],bl[0]['full_qvel_before'])
            assert len(states)==len(logs)==len(t['trace']) and t['steps']==len(states)+32
            routed=any(l['route_active'] for l in logs)
            for p in t['phases']:assert p['steps']<=(300 if p['phase']=='descend' and routed else budgets[p['phase']])
            for l in logs:
                np.testing.assert_array_equal(l['bias'],t['initial_estimator']['bias'])
                if l['eligible']:
                    j=np.linalg.solve(l['estimator_gram'],l['estimator_cross']).T if condition in ('analytical_metric','combined') else np.array(l['estimator_jacobian'],dtype=np.float64)
                    np.testing.assert_allclose(l['metric'],j.T@j+.015**2*np.eye(6),atol=1e-12,rtol=0)
                if condition in ('universal_readiness','combined') and l['phase']=='close':
                    assert l['handoff_control']
                    if not l['close_ready']:assert l['aperture']==0
                if l['aborted']:assert l['aperture']==0 and l['handoff_control']
                checks+=1
            close=[l for l in logs if l['phase']=='close'];first=close[0]
            model=mujoco.MjModel.from_xml_path(str(root/'states'/f'{stem}.xml'));data=mujoco.MjData(model)
            data.qpos[:]=first['full_qpos_before'];data.qvel[:]=first['full_qvel_before'];mujoco.mj_forward(model,data)
            site=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');obj=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'object')
            goal=np.array(first['pick_goal']) if first['pick_goal'] is not None else np.r_[tasks[case]['xy'],.24]
            gap=data.site_xpos[site]-goal;axis=np.linalg.norm(data.site_xmat[site].reshape(3,3)[:,2]-[0,0,1]);drift=np.linalg.norm(data.xpos[obj,:2]-goal[:2])
            readiness=dict(xy_error_mm=1000*float(np.linalg.norm(gap[:2])),z_error_mm=1000*float(abs(gap[2])),axis_error=float(axis),object_drift_mm=1000*float(drift))
            expected_ready=bool(np.linalg.norm(gap[:2])<=.008 and abs(gap[2])<=.012 and axis<.08 and drift<=.008)
            if first['handoff_control']:assert first['close_ready']==expected_ready
            if condition in ('universal_readiness','combined'):
                lift=next(l for l in logs if l['phase']=='lift')
                assert lift['aborted']==(not first['close_ready'] or close[-1]['close_good']<5)
            # Guard-only must be identical to baseline until close, including probes.
            if condition=='universal_readiness':
                idx=next(i for i,l in enumerate(logs) if l['phase']=='close')
                assert t['trace'][:idx]==baseline['trace'][:idx]
                np.testing.assert_array_equal(states[:idx],np.load(ROOT/'baseline/states'/f'{stem}.npz')['states'][:idx])
            if condition=='baseline':
                ref=read(REFERENCE/'trials'/f'{stem}.json');assert t['trace']==ref['trace']
                np.testing.assert_array_equal(states,np.load(REFERENCE/'states'/f'{stem}.npz')['states'])
            loss=baseline['success'] and not t['success']
            if loss or t['collision']!=baseline['collision']:selected.add(case)
            first_bad=next((dict(step=i,phase=x['phase'],contacts=x['bad']) for i,x in enumerate(t['trace']) if x['bad']),None)
            result.append(dict(case=case,family=r['family'],body=r['body'],success=t['success'],collision=t['collision'],steps=t['steps'],loss=loss,routed=routed,aborted=logs[-1]['aborted'],guard_active=first['handoff_control'],close_ready=first['close_ready'],would_be_ready=expected_ready,close_entry=readiness,final_two_pad_steps=close[-1]['close_good'],first_bad=first_bad,phases=t['phases']))
        details[condition]=result
        metrics[condition]=dict(episodes=48,success=sum(r['success'] for r in result),collisions=sum(r['collision'] for r in result),losses=[r['case'] for r in result if r['loss']],aborts=sum(r['aborted'] for r in result),routed=sum(r['routed'] for r in result),total_steps=sum(r['steps'] for r in result),median_steps=float(np.median([r['steps'] for r in result])))
    save(ROOT/'summary.json',metrics);save(ROOT/'details.json',details);save(ROOT/'verification.json',dict(steps_audited=checks,baseline_exact_replays=48,preflight_exact_replays=5,metric_verified=True,universal_guard_verified=True,guard_only_preclose_prefix_exact=True,identical_calibration=True,budgets_preserved=True,held_evaluated=False))
    lines=['# E179: projection metric and universal grasp-readiness on fitting cases','','Frozen E120 nominal policy and correction weights in every condition. Existing 48 six-joint fitting cases; privileged geometry/object/contact protections and release fix retained. A prespecified 2x2 ablation, not a new unseen-body evaluation. Baseline exactly reproduces all 48 original successes.','','| Condition | Success | Collision episodes | Lost baseline successes | Abort episodes | Total steps | Median steps |','|---|---:|---:|---|---:|---:|---:|']
    for c,m in metrics.items():lines.append(f"| {c} | {m['success']}/48 | {m['collisions']} | {m['losses']} | {m['aborts']} | {m['total_steps']} | {m['median_steps']} |")
    lines+=['','The analytical-metric conditions change only J used in the clearance projection metric, using current measured-history gram/cross statistics; E120 nominal actions remain neural. The universal-readiness conditions extend the existing routed close-anchor/ready/abort behavior to every grasp. This includes holding the close-entry joint posture, not merely adding a boolean check. Thresholds and interaction caps are inherited unchanged; no retries are introduced.','','All conditions retain calibration32 and probe32. Remaining original phase caps and early exits are unchanged; only an accepted route permits descent up to300 rather than100 steps. Actual interaction totals include calibration. Aborts still consume the existing later-phase budgets while holding/opening, and can therefore cost more steps.','','The full cohort and raw trajectories are retained, including failures. Baseline success is already48/48, so candidates can only preserve or regress this fitting score; this experiment cannot establish improved generalization. E178 unseen outcomes motivated these candidate mechanisms, but no unseen episode is executed here.','','## Audits and artifacts','',f'{checks} steps audited for the requested metric, frozen bias, guard behavior and original phase limits. Five exact preflight replays and48 exact full baseline replays. Guard-only trajectories match baseline exactly up to closing. Results: summary.json and details.json; protocol.json specifies the intervention before outcomes; each condition retains trials, states, actions, histories, clearance and planning artifacts. Videos are executed-state replays, verified against saved position/object traces.','']
    (ROOT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4));names=list(metrics);x=np.arange(4)
    ax.bar(x-.18,[metrics[c]['success'] for c in names],.36,label='Task successes');ax.bar(x+.18,[metrics[c]['collisions'] for c in names],.36,label='Collision episodes',color='#b34545');ax.set_xticks(x,['E120 baseline','Analytical metric','Universal readiness','Both']);ax.set_ylim(0,52);ax.set_ylabel('Episodes out of48');ax.set_title('E179 | fitting only | frozen E120 nominal policy');ax.legend();fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    for folder in ('trials','states'):(ROOT/folder).mkdir(exist_ok=True)
    videos=[]
    for c in CONDITIONS:
        for case in sorted(selected):
            old=f'{case}_e120_ensemble_frozen_bias';new=f'{case}_{c}_frozen_bias';t=read(ROOT/c/'trials'/f'{old}.json');t['method']=c;save(ROOT/'trials'/f'{new}.json',t)
            for ext in ('npz','xml'):shutil.copy2(ROOT/c/'states'/f'{old}.{ext}',ROOT/'states'/f'{new}.{ext}')
            videos.append(dict(case=case,method=c,mode='frozen_bias'))
    save(ROOT/'video_records.json',videos);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'status.json',dict(status='audited_videos_pending',videos=len(videos)));print(metrics,flush=True)

def finish():
    videos=read(ROOT/'videos/manifest.json')['videos'];assert len(videos)==len(read(ROOT/'video_records.json'))
    for v in videos:assert v['all_saved_states_match_trace'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    for p,h in {**read(ROOT/'frozen.json'),**read(ROOT/'protocol.json')['checkpoints']}.items():assert fingerprint(p)==h
    summary=read(ROOT/'summary.json');section='\n## E179: fitting projection metric and universal readiness\n\n'+str(summary)+'\n\nPrespecified fitting-only2x2 ablation; frozen E120 nominal policy/checkpoints, protections and release fix.192 episodes plus5 exact preflight replays; all48 baseline trajectories exact. Report: `outputs/ease_metric_readiness_v1/report.md`; videos: `videos/index.html`.\n'
    interpretation=ROOT/'interpretation.md'
    if interpretation.exists():section+='\n'+interpretation.read_text(encoding='utf-8')+'\n'
    review=Path('outputs/ease_iteration_review_v1')
    for p in (Path(__file__).with_name('TECHNICAL_REPORT.md'),review/'report.md'):
        s=p.read_text(encoding='utf-8')
        if '## E179:' not in s:p.write_text(s+section,encoding='utf-8')
    p=review/'WORKING_STATE.md';s=p.read_text(encoding='utf-8')
    if not s.startswith('# Latest checkpoint: E179'):p.write_text('# Latest checkpoint: E179 complete\n'+section+'\nAll runs, audits and videos complete. E120 remains baseline; no held execution.\n\n---\n'+s,encoding='utf-8')
    studies=read(review/'studies.json')
    if not any(x['experiment']=='E179' for x in studies):studies.append(dict(experiment='E179',folder=str(ROOT),report_exists=True,videos=len(videos)));save(review/'studies.json',studies)
    save(ROOT/'status.json',dict(status='complete',episodes=192,preflight=5,baseline_exact_replays=48,videos=len(videos),held_evaluated=False,weights_changed=False));save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print('E179 complete',flush=True)

if __name__=='__main__':
    import sys
    finish() if len(sys.argv)>1 and sys.argv[1]=='finish' else run()
