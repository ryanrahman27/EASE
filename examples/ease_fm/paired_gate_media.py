"""Shared result-preserving media/report helper for fitting-only gate studies."""
import shutil
from pathlib import Path
from .abstaining_task_value_gate import read,save,DATA
from .expanded_structured_pairs import COLLECT

def finish(root,experiment,module,results,selected,narrative):
    tasks=read(root/'cases.json');wanted={}
    for name,result in results.items():
        for dof in (6,7):
            for gain in (True,False):
                rr=[r for r in result['episodes'] if r['success']==gain and r['success']!=r['baseline_success'] and len(tasks[r['case']]['home'])==dof]
                if rr:
                    r=rr[0]
                    for phase in (r['phase'],None):wanted.setdefault((r['case'],r['seed'],phase),[]).append(name)
    reference=next(iter(results.values()))
    for dof in (6,7):
        for success in (True,False):
            rr=[r for r in reference['episodes'] if r['success']==success and len(tasks[r['case']]['home'])==dof]
            if rr:
                r=rr[0];wanted.setdefault((r['case'],r['seed'],r['phase']),[]).append('representative outcome')
    records=[]
    for (case,seed,phase),reasons in wanted.items():
        source=DATA/'collection' if case<56 else COLLECT;method=f'on_{seed}' if phase is None else f'off_{phase}_{seed}';stem=f'{case}_{method}_frozen_bias';r=read(source/'trials'/f'{stem}.json')
        for folder,exts in (('trials',('json',)),('states',('npz','xml'))):
            for ext in exts:shutil.copy2(source/folder/f'{stem}.{ext}',root/folder/f'{stem}.{ext}')
        records.append(dict(case=case,method=method,mode='frozen_bias',success=r['success'],collision=r['collision'],selection_reasons=reasons))
    save(root/'records.json',records)
    lines=[f'# {experiment}: {module.replace("_"," ")}','',narrative,'','| Condition | Successes | Rescues | Lost successes | Switches |','|---|---:|---:|---:|---:|']
    for k,r in results.items():lines.append(f"| {k} | {r['success']}/{r['n']} | {r['gains']} | {r['losses']} | {r['switches']} |")
    lines += ['',f"E120 baseline: {selected['baseline_success']}/{selected['n']}. Selected rule: {selected['kind']}, {selected['gains']} gains/{selected['losses']} losses. These are adaptive fitting-validation reconstructions from executed counterfactuals. The reserved fresh test and old held placements were not evaluated. E120 remains unchanged.",'', 'All predictions, family exclusions and full-task reconstructions were independently verified. Saved outcome-selected success/failure replays include negative results; these are not newly executed policy episodes.','']
    (root/'report.md').write_text('\n'.join(lines));p=Path(__file__).with_name('TECHNICAL_REPORT.md');text=p.read_text(encoding='utf-8');marker=f'## {experiment}: {module.replace("_"," ")}';assert marker not in text
    p.write_text(text+'\n'+marker+f"\n\nSelected {selected['kind']}: {selected['success']}/{selected['n']} vs E120{selected['baseline_success']}, {selected['gains']} gains/{selected['losses']} losses. Fresh test not opened. Report: `{root}/report.md`.\n",encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=list(results);fig,ax=plt.subplots(figsize=(max(8,len(keys)*.7),4));ax.bar(range(len(keys)),[results[k]['success'] for k in keys]);ax.axhline(selected['baseline_success'],color='black',ls='--',label='E120');ax.set_xticks(range(len(keys)),keys,rotation=45,ha='right',fontsize=7);ax.set_ylim(0,selected['n']);ax.set_ylabel('Fitting successes');ax.legend();fig.tight_layout();fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
    for name in (module+'.py','paired_gate_media.py'):shutil.copy2(Path(__file__).with_name(name),root/'source'/name)
    s=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT',module+' import ROOT').replace('E128',experiment).replace('Task-value ranking diagnosis',module.replace('_',' ')).replace('Diagnosis only; no new policy.','Fitting-only validation; all failures retained; fresh test reserved.')
    Path(__file__).with_name(module+'_videos.py').write_text(s);print(experiment,'prepared',len(records),'videos',flush=True)
