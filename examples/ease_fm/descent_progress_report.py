"""E171 full trajectory audit, paired outcomes, report and video artifacts."""
from pathlib import Path
from collections import Counter
import shutil
import numpy as np
import mujoco
from .descent_progress import ROOT,BASE,REFERENCE,CONDITIONS,prior,geometry_ids,distances,distance_gradient
from .link_clearance_report import outcome_counts
from .abstaining_task_value_gate import read,save,fingerprint

def run():
    refs={'E120_release':(BASE,[r for r in read(BASE/'records.json') if r['dof']==6]),'E170_protected':(REFERENCE,read(REFERENCE/'records.json'))}
    summary={'references':{k:outcome_counts(rows,root) for k,(root,rows) in refs.items()},'conditions':{}};checks={};selected={}
    for c in CONDITIONS:
        root=ROOT/c;rows=read(root/'records.json');assert len(rows)==48 and read(root/'verification.json')['episodes']==48
        paired={}
        for key,(_,rs) in refs.items():
            lookup={r['case']:r for r in rs};assert set(lookup)=={r['case'] for r in rows}
            paired[key]=dict(rescues=[r['case'] for r in rows if r['success'] and not lookup[r['case']]['success']],losses=[r['case'] for r in rows if not r['success'] and lookup[r['case']]['success']])
        post_count=0;gradient_count=0;objective_checks=0;changed=Counter();fallbacks=0;first_contacts=[];allowances=[];probe_changes=0
        for r in rows:
            stem=f"{r['case']}_e120_ensemble_frozen_bias";t=read(root/'trials'/f'{stem}.json');ref=read(REFERENCE/'trials'/f'{stem}.json');logs=read(root/'clearance'/f'{stem}.json');states=np.load(root/'states'/f'{stem}.npz')['states'];assert len(states)==len(logs)==len(t['trace'])
            assert t['calibration']==ref['calibration'] and t['initial_estimator']==ref['initial_estimator'];assert t['initial_estimator']['bias']==t['final_estimator']['bias']
            n=next(i for i,l in enumerate(logs) if l['phase']=='descend');assert t['trace'][:n]==ref['trace'][:n];np.testing.assert_array_equal(states[:n],np.load(REFERENCE/'states'/f'{stem}.npz')['states'][:n])
            m=mujoco.MjModel.from_xml_path(str(root/'states'/f'{stem}.xml'));d=mujoco.MjData(m);scratch=mujoco.MjData(m);ids,table=geometry_ids(m,6,True);history=[];checked=False;cost_checked=False
            first=next((i for i,x in enumerate(t['trace']) if x['bad']),None)
            if first is not None:first_contacts.append(dict(case=r['case'],step=first,phase=logs[first]['phase'],pairs=t['trace'][first]['bad'],predicted_distance=logs[first]['predicted_distance'],distance_after=logs[first]['distance_after'],allowance=logs[first]['allowance']))
            for l,state in zip(logs,states):
                expected=np.zeros(9)
                if l['phase']=='descend':expected[6:]=.003 if c=='fixed_allowance' else .001+min(.020,max(history[-8:],default=0.))
                np.testing.assert_array_equal(expected,l['allowance']);effective=np.array(l['distance_before'])-expected/prior.ALPHA;np.testing.assert_array_equal(effective,l['effective_distance'])
                if l['phase'] in ('lower','release'):assert l['goal'][2]==.265
                np.testing.assert_array_equal(np.clip(np.array(l['q_before'])+np.clip(l['final_action'],-.04,.04),-2.5,2.5)-l['q_before'],l['executed_command'])
                d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_forward(m,d);np.testing.assert_allclose(distances(m,d,ids,table),l['distance_after'],atol=1e-12,rtol=0);post_count+=1
                if np.max(np.abs(np.array(l['final_action'])-l['nominal_action']))>1e-10:changed[l['phase']]+=1
                if l['phase']=='descend':allowances.append(float(expected[-1]))
                expected_direct=bool(l['phase']=='descend' and (c=='direct_task' or np.max(np.abs(l['q_before']))>=2.4))
                assert l['direct_task']==expected_direct
                if expected_direct:
                    d.qpos[:]=l['full_qpos_before'];d.qvel[:]=l['full_qvel_before'];mujoco.mj_forward(m,d)
                    site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');j=np.array(l['estimated_jacobian']);error=np.r_[np.array(l['goal'])-d.site_xpos[site],.15*(np.array([0.,0.,1.])-d.site_xmat[site].reshape(3,3)[:,2])]
                    target=j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*error)+l['bias'];np.testing.assert_allclose(target,l['optimization_target'],atol=1e-12,rtol=0);np.testing.assert_allclose(j.T@j+.015**2*np.eye(6),l['metric'],atol=1e-12,rtol=0)
                    if not cost_checked and np.max(np.abs(target))>.04 and l['solve']['solver_success']:
                        competitor,ci,_=prior.project(np.clip(target,-.04,.04),np.array(l['bias']),np.array(l['q_before']),effective,np.array(l['gradient']),np.array(l['metric']))
                        if ci['solver_success']:
                            h=np.array(l['metric']);cost=lambda u:.5*(u-target)@h@(u-target)
                            assert cost(np.array(l['final_action']))<=cost(competitor)+1e-7
                            objective_checks+=1;cost_checked=True
                else:np.testing.assert_array_equal(l['optimization_target'],l['nominal_action'])
                if l['eligible']:
                    g=np.array(l['gradient']);bias=np.array(l['bias']);u=np.array(l['final_action']);rhs=g@bias-prior.ALPHA*(effective-prior.MARGIN)
                    np.testing.assert_allclose(rhs,l['rhs'],atol=1e-14,rtol=0)
                    if l['solve']['solver_success']:assert np.min(g@u-rhs)>=-1e-7
                    if l['solve']['fallback']:
                        q=np.array(l['q_before']);np.testing.assert_array_equal(u,np.clip(bias,np.maximum(-.04,-2.5-q),np.minimum(.04,2.5-q)));fallbacks+=1
                    prediction=np.array(l['distance_before'])+g@(u-bias);np.testing.assert_allclose(prediction,l['predicted_distance'],atol=1e-14,rtol=0)
                    if l['phase']=='descend':history.append(float(max(0.,np.max(prediction[6:]-np.array(l['distance_after'])[6:]))))
                    if not checked:
                        d.qpos[:]=l['full_qpos_before'];d.qvel[:]=l['full_qvel_before'];mujoco.mj_forward(m,d);np.testing.assert_allclose(distances(m,d,ids,table),l['distance_before'],atol=1e-12,rtol=0);before=d.qpos.copy();np.testing.assert_allclose(distance_gradient(m,d,scratch,ids,table,6),g,atol=1e-10,rtol=0);np.testing.assert_array_equal(before,d.qpos);checked=True;gradient_count+=1
                else:np.testing.assert_array_equal(l['nominal_action'],l['final_action'])
            probe_changes+=r['changed_probe_steps']
        counts=outcome_counts(rows,root);summary['conditions'][c]=dict(outcomes=counts,paired=paired,changed_commands_by_phase=dict(changed),fallbacks=fallbacks,probe_changed_commands=probe_changes,allowance_quantiles_m=np.quantile(allowances,[0,.5,.9,1]).tolist(),first_contacts=first_contacts,strict_screen=counts['success']>44 and counts['collisions']==0 and not paired['E170_protected']['losses'])
        checks[c]=dict(post_state_checks=post_count,gradient_checks=gradient_count,matched_feasible_objective_checks=objective_checks,direct_task_targets_and_gate_verified=True,causal_allowances_verified=True,actual_commands_verified=True,accepted_constraints_verified=True,pre_descent_exact=True,calibration_and_frozen_bias_preserved=True)
        chosen={26}
        for category in ([r['case'] for r in rows if r['success']],[r['case'] for r in rows if not r['success'] and not r['collision']],paired['E170_protected']['rescues'],paired['E170_protected']['losses']):
            if category:chosen.add(category[0])
        selected[c]=sorted(chosen)
    for p,h in read(ROOT/'frozen.json').items():assert fingerprint(p)==h
    save(ROOT/'summary.json',summary);save(ROOT/'verification.json',dict(conditions=checks,new_interventions=96,exact_replays=4,held_evaluated=False,privileged=True))
    videos=[]
    for folder in ('states','trials'):(ROOT/folder).mkdir(exist_ok=True)
    for c,cases in selected.items():
        for case in cases:
            for source,method in [(ROOT/c,'privileged_'+c),(REFERENCE,'privileged_E170'),(BASE,'baseline_E120_release')]:
                if any(v['case']==case and v['method']==method for v in videos):continue
                old=f'{case}_e120_ensemble_frozen_bias';new=f'{case}_{method}_frozen_bias';t=read(source/'trials'/f'{old}.json');t['method']=method;save(ROOT/'trials'/f'{new}.json',t)
                for ext in ('xml','npz'):shutil.copy2(source/'states'/f'{old}.{ext}',ROOT/'states'/f'{new}.{ext}')
                videos.append(dict(case=case,method=method,mode='frozen_bias',success=t['success'],dof=6))
    save(ROOT/'video_records.json',videos)
    lines=['# E171: descent progress under active constraints','','Privileged fitting-only control diagnostic. Frozen E120, measured-error descent allowance, probe protection,265mm release fix, original phase budgets and success criteria are retained.96 intervention episodes and4 exact E170 reference replays completed. No held evaluation or retraining.','','| Controller | Success /48 | Collisions | Collision-free failures | Rescues / losses vs E170 |','|---|---:|---:|---:|---|']
    for k,v in summary['references'].items():lines.append(f"| {k} | {v['success']} | {v['collisions']} | {v['collision_free_failures']} | — |")
    for c,v in summary['conditions'].items():
        x=v['outcomes'];p=v['paired']['E170_protected'];lines.append(f"| {c} | {x['success']} | {x['collisions']} | {x['collision_free_failures']} | {p['rescues']} / {p['losses']} |")
    lines+=['','Diagnosis: all four remaining reference failures spend100 descent steps and end with joint5 at its upper bound, active clearance constraints, and little or negative late progress. Case26 also has large orientation error. Full48-case saved-state diagnosis is retained in diagnosis.json.','', 'The original controller clips each damped least-squares joint command before projecting it into constraints. The new objective minimizes the E120-estimated task residual directly under the same command, joint-target and clearance constraints. Its quadratic center is the unclipped damped least-squares action; its metric remains J.T J +0.015 squared I. direct_task applies during every descent step; limit_gated applies only when any measured absolute joint coordinate is at least2.4rad. These settings were frozen before outcomes. No added steps, oracle task Jacobian, goal change, retraining or relaxation of safety/scoring is included.','', 'The E170 causal measured-error allowance and all probe/palm/link protections remain in both variants. E120 weights and calibrated bias are unchanged. Full simulator geometry is still privileged, and the local model can be inaccurate. This is not evidence of learned-only adaptation or independent transfer.','', 'Verification covers four exact reference replays including stalled cases, every saved geometry state and executed command, full-state finite-difference gradients, causal error allowances, accepted constraint inequalities, target/gate reconstruction, frozen calibration and identical pre-descent prefixes. The quadratic task-objective equivalence was checked before interventions. All failures, collisions and solver fallbacks are retained. Videos replay actual saved states with trace/frame-count checks;80ms video sampling can miss contacts scored at2ms.','', 'A positive screen requires more than44 successes, no collisions and no loss of any E170 success. Outcomes are reported against both E170 and original E120+release. No candidate automatically replaces the baseline.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vals=list(summary['references'].values())+[summary['conditions'][c]['outcomes'] for c in CONDITIONS];labels=['E120','E170',*CONDITIONS];fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key in zip(axes,['success','collisions','collision_free_failures']):
        ns=[v[key] for v in vals];ax.bar(labels,ns);ax.set_title(key);ax.set_ylim(0,48);ax.tick_params(axis='x',rotation=25)
        for i,n in enumerate(ns):ax.text(i,n+.5,str(n),ha='center')
    fig.suptitle('E171 privileged descent task progress | fitting only');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    template=Path(__file__).with_name('link_clearance_videos.py').read_text().replace('link_clearance_diagnostic import ROOT','descent_progress import ROOT').replace('E168','E171').replace('link-clearance comparison','descent task-progress comparison').replace('PRIVILEGED link/table constraints around E120 versus unchanged E120+release. Fitting cases only; successes, failures and stalls retained.','PRIVILEGED direct task objective with E170 protection; matched E170 and E120+release references. Fitting cases only; all outcomes retained.')
    Path(__file__).with_name('descent_progress_videos.py').write_text(template)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'status.json',dict(status='verified_videos_pending',videos=len(videos)))
    print({c:{k:v for k,v in x.items() if k!='first_contacts'} for c,x in summary['conditions'].items()},flush=True)

def finish():
    videos=read(ROOT/'videos/manifest.json')['videos'];assert len(videos)==len(read(ROOT/'video_records.json'))
    for v in videos:assert v['all_saved_states_match_trace'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    s=read(ROOT/'summary.json');text='; '.join(f"{c}: {v['outcomes']['success']}/48 success, {v['outcomes']['collisions']} collisions, {v['paired']['E170_protected']} versus E170" for c,v in s['conditions'].items())
    section='\n## E171: descent task-progress objective\n\n'+text+'. E170 protected reference44/48 with0 collisions; E120+release37/48. Privileged fitting diagnostic only,96 new episodes,4 exact replays. E120/probe protection/release preserved. Report: `outputs/ease_descent_progress_v1/report.md`; saved success/failure videos: `videos/index.html`.\n'
    review=Path('outputs/ease_iteration_review_v1')
    for p in (Path(__file__).with_name('TECHNICAL_REPORT.md'),review/'report.md'):
        old=p.read_text(encoding='utf-8')
        if '## E171:' not in old:p.write_text(old+section,encoding='utf-8')
    p=review/'WORKING_STATE.md';old=p.read_text(encoding='utf-8')
    if not old.startswith('# Latest checkpoint: E171'):p.write_text('# Latest checkpoint: E171 complete\n'+section+'\nAll runs, audits and videos complete.\n\n---\n'+old,encoding='utf-8')
    studies=read(review/'studies.json')
    if not any(x['experiment']=='E171' for x in studies):studies.append(dict(experiment='E171',folder=str(ROOT),report_exists=True,videos=len(videos)));save(review/'studies.json',studies)
    save(ROOT/'status.json',dict(status='complete',new_episodes=96,exact_replays=4,videos=len(videos),held_evaluated=False,E120_unchanged=True,probe_protection_preserved=True))
    save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print('E171 complete',len(videos),'videos',flush=True)

if __name__=='__main__':
    import sys
    finish() if len(sys.argv)>1 and sys.argv[1]=='finish' else run()
