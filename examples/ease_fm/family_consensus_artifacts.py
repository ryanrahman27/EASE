"""Archive E137, including unsuccessful confidence rules."""
import shutil
from pathlib import Path
import numpy as np
from .family_consensus_gate import ROOT,MOTION,VALUE,MARGINS,ridge,predict
from .structured_response import fit
from .short_horizon_paired import data,score
from .abstaining_task_value_gate import read,save,fingerprint,COLLECT

def run():
    _,_,_,folds,meta,tasks=data();d=np.load(MOTION/'dataset.npz');v=np.load(VALUE/'dataset.npz');o=np.load(ROOT/'oof.npz');families=o['family'];adv=np.zeros_like(o['advantage']);progress=np.zeros_like(o['progress'])
    for f in range(4):
        va=folds==f
        for k,excluded in enumerate(sorted(set(families[~va]))):
            tr=(~va)&(families!=excluded);m=ridge(v['x'][tr],v['y'][tr]);s=fit('scalar',d['effect'][tr],d['target'][tr],d['phase'][tr]);saved=np.load(ROOT/'models'/f'fold_{f}_exclude_{excluded}.npz')
            for name,val in {**m,'scalar':s,'training':np.flatnonzero(tr)}.items():np.testing.assert_array_equal(val,saved[name])
            adv[k,va],progress[k,va]=predict(m,s,v['x'][va],d['effect'][va],d['common'][va],d['error'][va])
    np.testing.assert_array_equal(adv,o['advantage']);np.testing.assert_array_equal(progress,o['progress']);grid=[score((adv.min(0)>m)&(progress.min(0)>1e-5),dict(value_margin=m,motion_margin=1e-5),meta,tasks) for m in MARGINS];assert grid==read(ROOT/'grid.json')
    for p,h in {**read(ROOT/'frozen.json'),**read(ROOT/'protocol.json')['inputs']}.items():assert fingerprint(p)==h
    save(ROOT/'verification.json',dict(family_deletion_models_reproduced=True,predictions_exact=True,grid_exact=True,fresh_test_unexecuted=True,inputs_unchanged=True))
    primary=read(ROOT/'primary.json');wanted={}
    for dof in (6,7):
        for success in (True,False):
            rr=[r for r in primary['episodes'] if r['phase'] is not None and r['success']==success and len(tasks[r['case']]['home'])==dof]
            if not rr:rr=[r for r in primary['episodes'] if r['success']==success and len(tasks[r['case']]['home'])==dof]
            if rr:
                r=rr[0]
                for phase in (r['phase'],None):wanted[r['case'],r['seed'],phase]=True
    records=[]
    for case,seed,phase in wanted:
        method=f'on_{seed}' if phase is None else f'off_{phase}_{seed}';stem=f'{case}_{method}_frozen_bias';r=read(COLLECT/'trials'/f'{stem}.json')
        for folder,exts in (('trials',('json',)),('states',('npz','xml'))):
            for ext in exts:shutil.copy2(COLLECT/folder/f'{stem}.{ext}',ROOT/folder/f'{stem}.{ext}')
        records.append(dict(case=case,method=method,mode='frozen_bias',success=r['success'],collision=r['collision']))
    save(ROOT/'records.json',records)
    lines=['# E137: family-deletion confidence','','Same structured response/task-value gate as E136, with six leave-one-training-family-out models inside each outer fitting fold. Require every model to exceed the fixed local progress threshold and task-value margin. Eight family-deletion final models were also saved. No original held or fresh test was executed.','', '| Task-value margin | Success /144 | Rescues | Lost successes | Switches |','|---|---:|---:|---:|---:|']
    for r in grid:lines.append(f"| {r['rule']['value_margin']} | {r['success']} | {r['gains']} | {r['losses']} | {r['switches']} |")
    lines += ['', 'Primary margin 0.05 yields 97/144, zero rescues/losses, nine interventions that do not change success. Zero-loss selection fully abstains. Confidence reduces intervention but does not preserve the gains; E120 remains unchanged. All OOF models and predictions reproduced. Videos include representative successes and failures, including unchanged outcomes.','', 'The 48 fresh randomized placements remain reserved and unexecuted. Next, expand fitting data independently (E138 seed 138000), without converting this test set into training data.','']
    (ROOT/'report.md').write_text('\n'.join(lines));p=Path(__file__).with_name('TECHNICAL_REPORT.md');text=p.read_text(encoding='utf-8');marker='## E137: family-deletion confidence';assert marker not in text
    p.write_text(text+'\n'+marker+'\n\nPrimary consensus:97/144, no gains/losses,9 neutral interventions. No-loss selection abstains.32 family-deletion models saved and OOF predictions verified. Fresh48-placement test reserved, unexecuted. Report: `outputs/ease_family_consensus_gate_v1/report.md`.\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(['E120','E136 fitting','E137 primary'],[97,102,97]);ax.set_ylim(0,144);ax.set_ylabel('Fitting successes /144');ax.set_title('Family consensus removes gains');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    s=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','family_consensus_gate import ROOT').replace('E128','E137').replace('Task-value ranking diagnosis','Family-deletion confidence').replace('Diagnosis only; no new policy.','Primary confidence gate: unchanged task outcomes, not promoted.')
    Path(__file__).with_name('family_consensus_videos.py').write_text(s);print('E137 verified; prepared',len(records),'videos',flush=True)

if __name__=='__main__':run()
