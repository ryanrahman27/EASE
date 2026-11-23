"""Independent saved-state audit and paired fitting report for E169."""
from pathlib import Path
from collections import Counter
import shutil
import numpy as np
import mujoco
from .probe_gripper_clearance import ROOT,BASE,REFERENCE,CONDITIONS,geometry_ids,distances,distance_gradient,prior
from .link_clearance_report import outcome_counts
from .abstaining_task_value_gate import read,save,fingerprint

def run():
    baseline=[r for r in read(BASE/'records.json') if r['dof']==6]
    reference=read(REFERENCE/'records.json')
    summary={'baseline':outcome_counts(baseline,BASE),'reference':outcome_counts(reference,REFERENCE),'conditions':{}}
    checks={};selection={}
    for condition in CONDITIONS:
        root=ROOT/condition;rows=read(root/'records.json');assert len(rows)==48 and read(root/'verification.json')['episodes']==48
        counts=outcome_counts(rows,root);paired={}
        for label,ref in [('E120_release',baseline),('E168_task',reference)]:
            lookup={r['case']:r for r in ref};assert set(lookup)=={r['case'] for r in rows}
            paired[label]={'rescues':[r['case'] for r in rows if r['success'] and not lookup[r['case']]['success']], 'losses':[r['case'] for r in rows if not r['success'] and lookup[r['case']]['success']]}
        phases=Counter();first_links=Counter();post_checks=0;gradient_checks=0;fallbacks=0;excitation=[];contact_errors=[]
        for r in rows:
            stem=f"{r['case']}_e120_ensemble_frozen_bias";logs=read(root/'clearance'/f'{stem}.json');t=read(root/'trials'/f'{stem}.json');b=read(BASE/'trials'/f'{stem}.json');states=np.load(root/'states'/f'{stem}.npz')['states'];assert len(logs)==len(states)==len(t['trace'])
            assert t['calibration']==b['calibration'] and t['initial_estimator']==b['initial_estimator'];assert t['initial_estimator']['bias']==t['final_estimator']['bias']
            m=mujoco.MjModel.from_xml_path(str(root/'states'/f'{stem}.xml'));d=mujoco.MjData(m);scratch=mujoco.MjData(m);ids,table=geometry_ids(m,6,condition!='probe_only');checked=False
            labels={mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or str(g) for g in ids[:6]}
            first=next((x for x in t['trace'] if any('table' in pair and set(pair)&labels for pair in x['bad'])),None)
            if first:first_links[first['phase']]+=1
            first_bad=next((i for i,x in enumerate(t['trace']) if x['bad']),None)
            if first_bad is not None:
                l=logs[first_bad];predicted=None if l['gradient'] is None else (np.array(l['distance_before'])+np.array(l['gradient'])@(np.array(l['final_action'])-l['bias'])).tolist()
                contact_errors.append(dict(case=r['case'],step=first_bad,phase=l['phase'],pairs=t['trace'][first_bad]['bad'],eligible=l['eligible'],solver_success=l['solve']['solver_success'],distance_before=l['distance_before'],predicted_distance=predicted,actual_distance=l['distance_after'],actual_substep_min=l['substep_min'],motion_prediction_error_norm=float(np.linalg.norm(states[first_bad,:6]-l['q_before']-(np.array(l['final_action'])-l['bias'])))))
            for i,(l,state) in enumerate(zip(logs,states)):
                np.testing.assert_array_equal(np.clip(np.array(l['q_before'])+np.clip(l['final_action'],-.04,.04),-2.5,2.5)-l['q_before'],l['executed_command'])
                if l['phase'] in ('lower','release'):assert l['goal'][2]==.265
                d.qpos[:]=state[:m.nq];d.qvel[:]=state[m.nq:];mujoco.mj_forward(m,d)
                np.testing.assert_allclose(distances(m,d,ids,table),l['distance_after'],atol=1e-12,rtol=0);post_checks+=1
                if l['solve']['attempted']:phases[l['phase']]+=1
                if l['eligible']:
                    g=np.array(l['gradient']);u=np.array(l['final_action']);bias=np.array(l['bias']);q=np.array(l['q_before']);rhs=g@bias-prior.ALPHA*(np.array(l['distance_before'])-prior.MARGIN)
                    np.testing.assert_allclose(rhs,l['rhs'],atol=1e-14,rtol=0)
                    if l['solve']['solver_success']:assert np.min(g@u-rhs)>=-1e-7
                    if l['solve']['fallback']:
                        np.testing.assert_array_equal(u,np.clip(bias,np.maximum(-.04,-2.5-q),np.minimum(.04,2.5-q)));fallbacks+=1
                    if not checked and i>0:
                        # Recover preceding full state so jaw positions are also exact.
                        if i>0:d.qpos[:]=states[i-1,:m.nq];d.qvel[:]=states[i-1,m.nq:]
                        d.qpos[:6]=q;mujoco.mj_forward(m,d)
                        np.testing.assert_allclose(distances(m,d,ids,table),l['distance_before'],atol=1e-12,rtol=0)
                        saved=d.qpos.copy();np.testing.assert_allclose(distance_gradient(m,d,scratch,ids,table,6),g,atol=1e-10,rtol=0);np.testing.assert_array_equal(saved,d.qpos);checked=True;gradient_checks+=1
                else:np.testing.assert_array_equal(l['nominal_action'],l['final_action'])
            probe=[i for i,l in enumerate(logs) if l['phase']=='probe'];assert len(probe)==32
            other=REFERENCE if condition=='probe_only' else ROOT/'gripper_only'
            if condition in ('probe_only','both'):
                n=probe[0];ot=read(other/'trials'/f'{stem}.json');assert t['trace'][:n]==ot['trace'][:n];np.testing.assert_array_equal(states[:n],np.load(other/'states'/f'{stem}.npz')['states'][:n])
            dq=np.array([states[i,:6]-logs[i]['q_before'] for i in probe]);ev=np.linalg.eigvalsh(dq.T@dq)
            ref_logs=read(REFERENCE/'clearance'/f'{stem}.json');ref_states=np.load(REFERENCE/'states'/f'{stem}.npz')['states'];rdq=np.array([ref_states[i,:6]-l['q_before'] for i,l in enumerate(ref_logs) if l['phase']=='probe']);rev=np.linalg.eigvalsh(rdq.T@rdq)
            excitation.append(dict(case=r['case'],changed_probe_steps=r['changed_probe_steps'],gram_eigenvalues=ev.tolist(),reference_gram_eigenvalues=rev.tolist()))
        summary['conditions'][condition]=dict(outcomes=counts,paired=paired,qp_steps_by_phase=dict(phases),fallbacks=fallbacks,first_link_contact_phases=dict(first_links),first_contact_prediction_diagnostics=contact_errors,probe_excitation=excitation,passes_strict_reference_screen=counts['success']>39 and counts['collisions']<=7 and not paired['E168_task']['losses'])
        checks[condition]=dict(post_state_geometry_checks=post_checks,gradient_checks=gradient_checks,actual_commands_and_constraints=True,matched_preprobe_prefix=True,release_and_frozen_bias_preserved=True)
        chosen={26,27}
        for success in (True,False):
            candidates=[r['case'] for r in rows if r['success']==success]
            if candidates:chosen.add(candidates[0])
        for values in paired['E168_task'].values():
            if values:chosen.add(values[0])
        selection[condition]=sorted(chosen)
    for p,h in read(ROOT/'frozen.json').items():assert fingerprint(p)==h
    save(ROOT/'summary.json',summary);save(ROOT/'verification.json',dict(conditions=checks,new_intervention_episodes=144,corrected_exact_replays=4,archived_initial_exact_replays=4,privileged=True,held_evaluated=False))
    videos=[]
    for folder in ('trials','states'):(ROOT/folder).mkdir(exist_ok=True)
    for condition,cases in selection.items():
        for case in cases:
            for source,method in [(ROOT/condition,'privileged_'+condition),(REFERENCE,'privileged_E168'),(BASE,'baseline_E120_release')]:
                if any(v['case']==case and v['method']==method for v in videos):continue
                old=f'{case}_e120_ensemble_frozen_bias';new=f'{case}_{method}_frozen_bias';t=read(source/'trials'/f'{old}.json');t['method']=method;save(ROOT/'trials'/f'{new}.json',t)
                for ext in ('xml','npz'):shutil.copy2(source/'states'/f'{old}.{ext}',ROOT/'states'/f'{new}.{ext}')
                videos.append(dict(case=case,method=method,mode='frozen_bias',success=t['success'],dof=6))
    save(ROOT/'video_records.json',videos)
    lines=['# E169: privileged probe and gripper clearance','','Fitting-only diagnostic with frozen E120 and the 265mm release fix. No held-body evaluation, retraining, or learned-only transfer claim.','','| Controller | Success /48 | Collision episodes | Collision-free failures | Rescues / losses vs E168 |','|---|---:|---:|---:|---|']
    for label,x in [('E120 + release',summary['baseline']),('E168 task projection',summary['reference'])]:lines.append(f"| {label} | {x['success']} | {x['collisions']} | {x['collision_free_failures']} | — |")
    for c,v in summary['conditions'].items():
        x=v['outcomes'];p=v['paired']['E168_task'];lines.append(f"| {c} | {x['success']} | {x['collisions']} | {x['collision_free_failures']} | {p['rescues']} / {p['losses']} |")
    lines+=['','All 144 intervention episodes are retained. Four corrected exact reference replays passed; four earlier exact replays are archived separately. Initial calibration, E120 weights, frozen bias, phase budgets, jaw commands and strict success/contact scoring are unchanged. The scheduled probe remains 32 steps, but protected variants modify its executed arm commands; per-case measured-motion Gram spectra and changed-command counts are saved in summary.json.','', 'Arm constraints retain E168 bounded-table capsule distances. New palm/pad constraints use the lowest box vertex above the infinite tabletop plane: conservative outside the table footprint. Before any intervention outcome, native box/table queries failed a directional-gradient preflight. The failed implementation and protocol are archived, and the geometric replacement passed independent derivative checks. This is a disclosed preflight correction, not outcome-driven tuning.','', 'The same task-metric linearized QP, 3mm margin and 30mm activation apply. Probe protection acts after the scheduled probe command is formed. Gripper gradients hold jaw positions fixed; jaw actions are unchanged. Ground, self and object contact are not constrained. Linearized feasibility does not guarantee nonlinear clearance. All fallbacks and actual contacts remain scored.','', 'summary.json includes paired rescues/losses against both references, first arm-link contact phases, collision types, phase progress, solver fallbacks and probe excitation. Verification reconstructs every saved post-state distance and actual command, accepted constraints, independent gradients and matching pre-probe prefixes. Success/failure videos replay saved states; 80ms video sampling can miss contacts scored at 2ms. E120 remains the baseline regardless of this diagnostic result.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    values=[summary['baseline'],summary['reference']]+[summary['conditions'][c]['outcomes'] for c in CONDITIONS];labels=['E120','E168',*CONDITIONS];fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key in zip(axes,['success','collisions','collision_free_failures']):
        nums=[v[key] for v in values];ax.bar(labels,nums);ax.set_title(key);ax.set_ylim(0,48);ax.tick_params(axis='x',rotation=25)
        for i,n in enumerate(nums):ax.text(i,n+.5,str(n),ha='center')
    fig.suptitle('E169 privileged coverage diagnostic | fitting cases');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    template=Path(__file__).with_name('link_clearance_videos.py').read_text().replace('link_clearance_diagnostic import ROOT','probe_gripper_clearance import ROOT').replace('E168','E169').replace('link-clearance comparison','probe/gripper clearance comparison');Path(__file__).with_name('probe_gripper_videos.py').write_text(template)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    save(ROOT/'status.json',dict(status='verified_videos_pending',videos_planned=len(videos)))
    print({c:{k:v for k,v in x.items() if k not in ('probe_excitation','first_contact_prediction_diagnostics')} for c,x in summary['conditions'].items()},flush=True)

def finish():
    manifest=read(ROOT/'videos/manifest.json')['videos'];assert len(manifest)==len(read(ROOT/'video_records.json'))
    for v in manifest:assert v['all_saved_states_match_trace'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    s=read(ROOT/'summary.json');paragraph='; '.join(f"{c}: {v['outcomes']['success']}/48 success, {v['outcomes']['collisions']} collisions, paired vs E168 {v['paired']['E168_task']}" for c,v in s['conditions'].items())
    section='\n## E169: privileged probe/gripper coverage\n\n'+paragraph+'\n\nE120+release remains37/48; E168 task projection39/48.144 fitting intervention episodes; no held test or model update. Complete results and saved success/failure videos: `outputs/ease_probe_gripper_clearance_v1/report.md` and `videos/index.html`. Conservative gripper-plane correction documented before intervention outcomes.\n'
    for p in [Path(__file__).with_name('TECHNICAL_REPORT.md'),Path('outputs/ease_iteration_review_v1/report.md')]:
        text=p.read_text(encoding='utf-8')
        if '## E169:' not in text:p.write_text(text+section,encoding='utf-8')
    review=Path('outputs/ease_iteration_review_v1');studies=read(review/'studies.json')
    if not any(x['experiment']=='E169' for x in studies):studies.append(dict(experiment='E169',folder=str(ROOT),report_exists=True,videos=len(manifest)));save(review/'studies.json',studies)
    p=review/'WORKING_STATE.md';old=p.read_text(encoding='utf-8')
    if not old.startswith('# Latest checkpoint: E169'):p.write_text('# Latest checkpoint: E169 complete\n'+section+'\nAll runs, audits and videos complete. E120 and release fix unchanged.\n\n---\n'+old,encoding='utf-8')
    save(ROOT/'status.json',dict(status='complete',new_episodes=144,exact_replays=4,archived_replays=4,videos=len(manifest),held_evaluated=False,E120_unchanged=True))
    save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
    print('E169 complete',len(manifest),'verified videos',flush=True)

if __name__=='__main__':
    import sys
    finish() if len(sys.argv)>1 and sys.argv[1]=='finish' else run()
