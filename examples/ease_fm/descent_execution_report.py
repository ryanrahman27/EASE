"""E170 full trajectory audit, paired outcomes, report and video artifacts."""
from pathlib import Path
from collections import Counter
import shutil
import numpy as np
import mujoco
from .descent_execution import ROOT,BASE,REFERENCE,CONDITIONS,prior,geometry_ids,distances,distance_gradient
from .link_clearance_report import outcome_counts
from .abstaining_task_value_gate import read,save,fingerprint

def run():
    refs={'E120_release':(BASE,[r for r in read(BASE/'records.json') if r['dof']==6]),'E169_protected':(REFERENCE,read(REFERENCE/'records.json'))}
    summary={'references':{k:outcome_counts(rows,root) for k,(root,rows) in refs.items()},'conditions':{}};checks={};selected={}
    for c in CONDITIONS:
        root=ROOT/c;rows=read(root/'records.json');assert len(rows)==48 and read(root/'verification.json')['episodes']==48
        paired={}
        for key,(_,rs) in refs.items():
            lookup={r['case']:r for r in rs};assert set(lookup)=={r['case'] for r in rows}
            paired[key]=dict(rescues=[r['case'] for r in rows if r['success'] and not lookup[r['case']]['success']],losses=[r['case'] for r in rows if not r['success'] and lookup[r['case']]['success']])
        post_count=0;gradient_count=0;changed=Counter();fallbacks=0;first_contacts=[];allowances=[];probe_changes=0
        for r in rows:
            stem=f"{r['case']}_e120_ensemble_frozen_bias";t=read(root/'trials'/f'{stem}.json');ref=read(REFERENCE/'trials'/f'{stem}.json');logs=read(root/'clearance'/f'{stem}.json');states=np.load(root/'states'/f'{stem}.npz')['states'];assert len(states)==len(logs)==len(t['trace'])
            assert t['calibration']==ref['calibration'] and t['initial_estimator']==ref['initial_estimator'];assert t['initial_estimator']['bias']==t['final_estimator']['bias']
            n=next(i for i,l in enumerate(logs) if l['phase']=='descend');assert t['trace'][:n]==ref['trace'][:n];np.testing.assert_array_equal(states[:n],np.load(REFERENCE/'states'/f'{stem}.npz')['states'][:n])
            m=mujoco.MjModel.from_xml_path(str(root/'states'/f'{stem}.xml'));d=mujoco.MjData(m);scratch=mujoco.MjData(m);ids,table=geometry_ids(m,6,True);history=[];checked=False
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
        counts=outcome_counts(rows,root);summary['conditions'][c]=dict(outcomes=counts,paired=paired,changed_commands_by_phase=dict(changed),fallbacks=fallbacks,probe_changed_commands=probe_changes,allowance_quantiles_m=np.quantile(allowances,[0,.5,.9,1]).tolist(),first_contacts=first_contacts,strict_screen=counts['success']>41 and counts['collisions']<4 and not paired['E169_protected']['losses'])
        checks[c]=dict(post_state_checks=post_count,gradient_checks=gradient_count,causal_allowances_verified=True,actual_commands_verified=True,accepted_constraints_verified=True,pre_descent_exact=True,calibration_and_frozen_bias_preserved=True)
        chosen={26}
        for category in ([r['case'] for r in rows if r['success']],[r['case'] for r in rows if not r['success'] and not r['collision']],paired['E169_protected']['rescues'],paired['E169_protected']['losses']):
            if category:chosen.add(category[0])
        selected[c]=sorted(chosen)
    for p,h in read(ROOT/'frozen.json').items():assert fingerprint(p)==h
    save(ROOT/'summary.json',summary);save(ROOT/'verification.json',dict(conditions=checks,new_interventions=96,exact_replays=4,held_evaluated=False,privileged=True))
    videos=[]
    for folder in ('states','trials'):(ROOT/folder).mkdir(exist_ok=True)
    for c,cases in selected.items():
        for case in cases:
            for source,method in [(ROOT/c,'privileged_'+c),(REFERENCE,'privileged_E169'),(BASE,'baseline_E120_release')]:
                if any(v['case']==case and v['method']==method for v in videos):continue
                old=f'{case}_e120_ensemble_frozen_bias';new=f'{case}_{method}_frozen_bias';t=read(source/'trials'/f'{old}.json');t['method']=method;save(ROOT/'trials'/f'{new}.json',t)
                for ext in ('xml','npz'):shutil.copy2(source/'states'/f'{old}.{ext}',ROOT/'states'/f'{new}.{ext}')
                videos.append(dict(case=case,method=method,mode='frozen_bias',success=t['success'],dof=6))
    save(ROOT/'video_records.json',videos)
    lines=['# E170: descent execution-error allowances','','Privileged fitting diagnostic. E120, frozen bias, probe protection and the265mm release fix remain unchanged. All96 new episodes and4 exact protected-reference replays completed; no held evaluation or retraining.','','| Controller | Success /48 | Collisions | Collision-free failures | Rescues / losses vs protected reference |','|---|---:|---:|---:|---|']
    for k,v in summary['references'].items():lines.append(f"| {k} | {v['success']} | {v['collisions']} | {v['collision_free_failures']} | — |")
    for c,v in summary['conditions'].items():
        x=v['outcomes'];p=v['paired']['E169_protected'];lines.append(f"| {c} | {x['success']} | {x['collisions']} | {x['collision_free_failures']} | {p['rescues']} / {p['losses']} |")
    lines+=['','Only descent gripper rows receive extra allowance. Fixed:3mm one-step error; history:1mm plus maximum positive clearance prediction error over the previous8 eligible descent steps (error capped20mm). Same settings for every case, frozen before outcomes. The allowance is subtracted from the linear constraint slack, equivalently from distance divided by barrier gain0.5. Real geometry and strict scoring remain unchanged. Non-descent phases retain E169 both constraints. Activation uses effective distance to engage earlier.','', 'The history allowance is based exclusively on already-executed measured errors and resets per episode. It does not update E120 weights or its frozen bias. Full simulator gripper geometry remains privileged. The conservative infinite tabletop plane and arm finite-difference gradients are retained; no future dynamics rollout is used. This is not learned-only transfer or guaranteed safety.','', 'Reference fitting errors were diagnosed before interventions: maximum downward clearance overprediction4.74mm,99th percentile4.20mm. Four exact E169 replays passed. All post-state geometry, actual executed commands, causal history windows, linear constraints, fallback actions, full-state gradient reconstructions and unchanged pre-descent prefixes were verified. Every failure is retained. summary.json contains paired outcomes against E120+release and E169, progress/contact categories, allowances and first-contact prediction errors.','', 'Videos replay saved states and verify tool/object traces and decoded frame counts.80ms video sampling may miss contacts scored at2ms. No candidate automatically replaces the baseline.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    vals=list(summary['references'].values())+[summary['conditions'][c]['outcomes'] for c in CONDITIONS];labels=['E120','E169',*CONDITIONS];fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key in zip(axes,['success','collisions','collision_free_failures']):
        ns=[v[key] for v in vals];ax.bar(labels,ns);ax.set_title(key);ax.set_ylim(0,48);ax.tick_params(axis='x',rotation=25)
        for i,n in enumerate(ns):ax.text(i,n+.5,str(n),ha='center')
    fig.suptitle('E170 privileged descent execution allowances | fitting only');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    template=Path(__file__).with_name('link_clearance_videos.py').read_text().replace('link_clearance_diagnostic import ROOT','descent_execution import ROOT').replace('E168','E170').replace('link-clearance comparison','descent execution-error comparison').replace('PRIVILEGED link/table constraints around E120 versus unchanged E120+release. Fitting cases only; successes, failures and stalls retained.','PRIVILEGED descent error allowances with probe protection; matched E169 and E120+release references. Fitting cases only; successes and failures retained.')
    Path(__file__).with_name('descent_execution_videos.py').write_text(template)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'status.json',dict(status='verified_videos_pending',videos=len(videos)))
    print({c:{k:v for k,v in x.items() if k!='first_contacts'} for c,x in summary['conditions'].items()},flush=True)

def finish():
    videos=read(ROOT/'videos/manifest.json')['videos'];assert len(videos)==len(read(ROOT/'video_records.json'))
    for v in videos:assert v['all_saved_states_match_trace'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    s=read(ROOT/'summary.json');text='; '.join(f"{c}: {v['outcomes']['success']}/48 success, {v['outcomes']['collisions']} collisions, {v['paired']['E169_protected']} versus E169" for c,v in s['conditions'].items())
    section='\n## E170: descent execution-error allowances\n\n'+text+'. E169 protected reference41/48 with4 collisions; E120+release37/48. Privileged fitting diagnostic only,96 new episodes,4 exact replays. E120/probe protection/release preserved. Report: `outputs/ease_descent_execution_v1/report.md`; saved success/failure videos: `videos/index.html`.\n'
    review=Path('outputs/ease_iteration_review_v1')
    for p in (Path(__file__).with_name('TECHNICAL_REPORT.md'),review/'report.md'):
        old=p.read_text(encoding='utf-8')
        if '## E170:' not in old:p.write_text(old+section,encoding='utf-8')
    p=review/'WORKING_STATE.md';old=p.read_text(encoding='utf-8')
    if not old.startswith('# Latest checkpoint: E170'):p.write_text('# Latest checkpoint: E170 complete\n'+section+'\nAll runs, audits and videos complete.\n\n---\n'+old,encoding='utf-8')
    studies=read(review/'studies.json')
    if not any(x['experiment']=='E170' for x in studies):studies.append(dict(experiment='E170',folder=str(ROOT),report_exists=True,videos=len(videos)));save(review/'studies.json',studies)
    save(ROOT/'status.json',dict(status='complete',new_episodes=96,exact_replays=4,videos=len(videos),held_evaluated=False,E120_unchanged=True,probe_protection_preserved=True))
    save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print('E170 complete',len(videos),'videos',flush=True)

if __name__=='__main__':
    import sys
    finish() if len(sys.argv)>1 and sys.argv[1]=='finish' else run()
