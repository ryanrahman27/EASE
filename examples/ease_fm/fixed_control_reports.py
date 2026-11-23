"""Archive E154 aggregate gains and E155/E156 rejected diagnostics honestly."""
import shutil
from pathlib import Path
import numpy as np
from .fixed_uniform_control import ROOT as CONTROL
from .history_estimator_control import ROOT as OLD
from .prior_projector_control import ROOT as PROJECTOR
from .lift_posture_control import ROOT as LIFT
from .conservative_history_weights import ROOT as CONSERVATIVE
from .abstaining_task_value_gate import read,save,fingerprint,DATA

def video_driver(root,module,experiment,title,description):
    st=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT',module+' import ROOT').replace("ROOT/'records.json'","ROOT/'video_records.json'").replace('E128',experiment).replace('Task-value ranking diagnosis',title).replace('Outcome-selected excluded-fitting-family pairs: E120 on versus short correction-off then E120. Diagnosis only; no new policy.',description);Path(__file__).with_name(module+'_videos.py').write_text(st)

def append(experiment,text,root):
    report=Path(__file__).with_name('TECHNICAL_REPORT.md');content=report.read_text(encoding='utf-8');assert f'## {experiment}:' not in content;report.write_text(content+f'\n## {experiment}: fixed-kernel continuation\n\n'+text+f' Report: `{root}/report.md`.\n',encoding='utf-8')

def run():
    s=read(CONTROL/'summary.json');selected={}
    for context in ('standard','posture'):
        root=CONTROL/context;rows=read(root/'records.json')
        for r in rows:
            stem=f"{r['case']}_learned_ensemble_frozen_bias";t=read(root/'trials'/f'{stem}.json');a=read(root/'actions'/f'{stem}.json');h=dict(np.load(root/'histories'/f'{stem}.npz'));assert len(a)==len(t['trace'])==len(h['query']);np.testing.assert_array_equal(h['dq_history'][1:,-1],h['query'][:-1]);np.testing.assert_array_equal(h['dx_history'][1:,-1],h['target'][:-1]);np.testing.assert_allclose(np.diff(np.array([v['q_before'] for v in a]),axis=0),h['query'][:-1,:r['dof']],atol=1e-8,rtol=1e-5);assert t['initial_estimator']['bias']==t['final_estimator']['bias']
        for p,h in read(root/'frozen.json').items():assert fingerprint(p)==h
    save(CONTROL/'execution_verification.json',dict(episodes=192,history_order_and_actual_joint_queries=True,frozen_bias=True,source_and_model_hashes_unchanged=True));text='Standard-controller learned weights complete77/96 versus matched uniform74 and E120 ensemble72; collisions17 versus20 and21. Six joints37/48 versus36/48; seven40/48 versus uniform38 and E12036. Paired versus uniform:5 rescues/2 losses; versus E120 ensemble:9 rescues/4 losses. This is an adaptive fitting-development result, not held or unseen-body evidence.'
    lines=['# E154: fixed-kernel learned weights in complete tasks','',text,'','The posture condition completes74/96 versus matched uniform76 and E120 ensemble70. It is not selected. The standard condition meets the aggregate/dof/collision requirements but fails the separate legacy-gain retention rule at case13. Case13 is a prior individual-seed E120 success; the matched E120 ensemble itself also fails it. This distinction is preserved rather than calling it an ensemble regression. The original advancement screen remains failed; fresh placements remain closed.','', '| Context | Learned success | Uniform success | E120 ensemble success | Learned collisions |','|---|---:|---:|---:|---:|']
    for context,v in s['contexts'].items():
        m=v['methods'];lines.append(f"| {context} | {m['learned_fixed_uniform']['all']['success']}/96 | {m['uniform32']['all']['success']}/96 | {m['e120_ensemble']['all']['success']}/96 | {m['learned_fixed_uniform']['all']['collisions']} |")
    lines+=['','All192 candidate trajectories, ordered histories and actions are saved and independently aligned. Same fixed kernels, task schedule, calibration, frozen bias and probe rules as each reference. No physics selector or target-body weight update. Videos show both gains and regressions and explicitly label the legacy single-seed comparison.',''];(CONTROL/'report.md').write_text('\n'.join(lines));append('E154',text+' Legacy case13 retention fails; no fresh test.',CONTROL)
    for r in s['contexts']['standard']['paired']['uniform32']['changes']:
        selected[r['case'],'learned_ensemble','standard']=CONTROL/'standard';selected[r['case'],'uniform32','standard']=OLD
    selected[13,'learned_ensemble','standard']=CONTROL/'standard';selected[13,'on_1','legacy_E120_seed1']=DATA/'collection'
    for success in (True,False):
        for n in (6,7):
            rows=[r for r in read(CONTROL/'posture/records.json') if r['success']==success and r['dof']==n]
            if rows:selected[rows[0]['case'],'learned_ensemble','posture']=CONTROL/'posture'
    for folder in ('trials','states'):(CONTROL/folder).mkdir(exist_ok=True)
    video=[]
    for (case,method,context),source in selected.items():
        oldstem=f'{case}_{method}_frozen_bias';alias=f'{method}_{context}';stem=f'{case}_{alias}_frozen_bias';t=read(source/'trials'/f'{oldstem}.json')
        if context=='legacy_E120_seed1':assert t['success']
        t['source_method']=t['method'];t['method']=alias;save(CONTROL/'trials'/f'{stem}.json',t)
        for ext in ('npz','xml'):shutil.copy2(source/'states'/f'{oldstem}.{ext}',CONTROL/'states'/f'{stem}.{ext}')
        video.append(dict(case=case,method=alias,mode='frozen_bias',success=t['success'],collision=t['collision']))
    save(CONTROL/'video_records.json',video)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4));names=['Learned standard','Uniform standard','E120 ensemble','Learned posture'];ax.bar(names,[77,74,72,74]);ax.set_ylim(0,96);ax.set_ylabel('Completed fitting tasks /96');ax.tick_params(axis='x',rotation=15);ax.set_title('E154 | all gains and regressions retained');fig.tight_layout();fig.savefig(CONTROL/'comparison.png',dpi=150);plt.close(fig);video_driver(CONTROL,'fixed_uniform_control','E154','Fixed-kernel learned control','Outcome-selected fitting gains and regressions. Legacy E120 seed1 is a separate individual-seed comparison, not the matched ensemble reference. Fresh test closed.')
    for root,experiment,module in ((PROJECTOR,'E155','prior_projector_control'),(LIFT,'E156','lift_posture_control')):
        d=read(root/'diagnostic_summary.json');rows=read(root/'records.json');assert len(rows)==12 and not d['ready_for_full'];lines=[f'# {experiment}: outcome-informed fitting diagnostic','','Only cases13,1,6,10, selected because of previous outcomes. These12 episodes are diagnostic and do not estimate general success rates. The projection uses measured calibration/history only; there is no physics selector or exact geometry.','', '| Case | Uniform | Learned | E120 ensemble |','|---|---:|---:|---:|']
        for case in (13,1,6,10):lines.append('| '+str(case)+' | '+' | '.join('success' if next(r['success'] for r in rows if r['case']==case and r['method']==m) else 'failure' for m in ('uniform32','learned_ensemble','e120_ensemble'))+' |')
        detail='All-phase projection rescues learned case13 but loses case6; diagnostic gate fails.' if experiment=='E155' else 'Lift-only projection does not recover learned case13 and loses case6. Every pre-lift trace and saved state matches the original standard controller exactly; all non-lift actions are unchanged. Diagnostic gate fails.'
        lines+=['',detail+' No full96-placement or fresh evaluation was performed. All trials/states and success/failure videos are retained; E120 unchanged.',''];(root/'report.md').write_text('\n'.join(lines));append(experiment,detail+' No full/fresh evaluation.',root);save(root/'video_records.json',rows);fig,ax=plt.subplots(figsize=(7,3));methods=['uniform32','learned_ensemble','e120_ensemble'];ax.bar(methods,[sum(r['success'] for r in rows if r['method']==m) for m in methods]);ax.set_ylim(0,4);ax.set_title(experiment+' | four outcome-selected diagnostic cases');fig.tight_layout();fig.savefig(root/'comparison.png',dpi=150);plt.close(fig);video_driver(root,module,experiment,'Selected posture diagnostic','Four outcome-selected fitting cases only; not a representative success rate. Full and fresh evaluation stopped after the diagnostic failed.');shutil.copy2(__file__,root/'source'/Path(__file__).name)
    m=read(CONSERVATIVE/'summary.json');(CONSERVATIVE/'report.md').write_text('# E157: conservative history weighting\n\nOne fixed.25 power on E153 relative weights followed by recency-mass normalization reduces the possible max/min weight ratio from25 to sqrt(5). Network checkpoints, kernel.98 and zero-centered ridge1e-6 are unchanged. No temperature sweep or retraining.\n\nEnsemble measured position RMSE0.404mm versus uniform0.415 (about5.2% lowerMSE), improving both joint counts. This deliberately gives up some of the full model\'s0.388mm accuracy to test a smaller control perturbation. Weight bounds, normalized mass and unchanged weights were checked. No task-success gain is claimed at this stage. Actual control is evaluated separately in E158; fresh set remains closed.\n');append('E157','Fixed conservative learned weights0.404mm vs uniform0.415; no retraining or temperature sweep. Prediction only.',CONSERVATIVE);shutil.copy2(__file__,CONTROL/'source'/Path(__file__).name);print('E154 prepared',len(video),'videos; E155/E15612 each; reports archived',flush=True)

if __name__=='__main__':run()
