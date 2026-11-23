"""Save complete learned/prior/exact-geometry comparisons with honest video labels."""
import shutil
from pathlib import Path
from .abstaining_task_value_gate import read,save,fingerprint
from .learned_calibration_control import ROOT as LEARNED
from .calibration_prior_control import ROOT as PRIOR
from .history_estimator_control import ROOT as OLD
from .exact_map_control import ROOT as EXACT

def run():
    sources={LEARNED:read(LEARNED/'records.json'),PRIOR:read(PRIOR/'records.json'),OLD:read(OLD/'records.json'),EXACT:read(EXACT/'records.json')};combined=sum(sources.values(),[])
    for root,experiment,module,primary,comparisons,text in [
      (LEARNED,'E147','learned_calibration_control','learned_prior',('calibration_prior','e120_ensemble','uniform32'),'The learned calibration-prior estimator completes71/96, compared with67/96 for its matched analytical prior (8 rescues,4 regressions),72/96 for E120 ensemble and74/96 for uniform32. All learned-versus-prior changes are seven-joint cases. It fails the frozen advancement screen: E120 ensemble is not exceeded and fitting gain placement13 remains unsuccessful. No held/fresh evaluation or E120 replacement.'),
      (EXACT,'E149','exact_map_control','privileged_exact_map',('learned_prior','e120_ensemble','uniform32'),'PRIVILEGED REFERENCE ONLY. Exact current kinematics completes76/96 with16 collision episodes:35/48 six-joint,41/48 seven-joint. This is the same fixed task controller and measured frozen bias, not an upper bound on all possible policies. Exact kinematics improves seven-joint performance but does not recover the six-joint failures of the calibration-prior method. It is never used by a learned estimator or training objective.')]:
        rows=sources[root];refs=[r for r in combined if r['method'] in comparisons];methods=(primary,*comparisons);lines=[f'# {experiment}: complete fitting control','',text,'','All96 placements are known-body fitting/development cases. Exact calibration and initial statistics, fixed bias, checkpoint and source hashes were verified. No future outcomes select a seed or checkpoint. E120 remains immutable.','', '| Controller | All success | Six joints | Seven joints | Collisions |','|---|---:|---:|---:|---:|'];stats={}
        for method in methods:
            rr=[r for r in rows+refs if r['method']==method];stats[method]={str(n):dict(n=len(ss),success=sum(r['success'] for r in ss),collisions=sum(r['collision'] for r in ss)) for n in ('all',6,7) if (ss:=[r for r in rr if n=='all' or r['dof']==n])};v=stats[method];lines.append(f"| {method} | {v['all']['success']}/96 | {v['6']['success']}/48 | {v['7']['success']}/48 | {v['all']['collisions']} |")
        pairs={};selected={}
        for method in comparisons:
            pp=[dict(case=r['case'],dof=r['dof'],before=next(v['success'] for v in refs if v['case']==r['case'] and v['method']==method),after=r['success']) for r in rows];pairs[method]=dict(gains=sum(p['after'] and not p['before'] for p in pp),losses=sum(p['before'] and not p['after'] for p in pp),changes=[p for p in pp if p['before']!=p['after']])
            for gain in (True,False):
                changes=[p for p in pp if p['after']==gain and p['before']!=p['after']]
                if changes:
                    case=changes[0]['case']
                    for m in (primary,method):selected[case,m]=next(r for r in rows+refs if r['case']==case and r['method']==m)
        for n in (6,7):
            for success in (True,False):
                rr=[r for r in rows if r['dof']==n and r['success']==success]
                if rr:r=rr[0];selected[r['case'],primary]=r
        save(root/'paired_comparison.json',pairs);lines+=['','Paired gains/losses and all changed placements are preserved in paired_comparison.json. Gallery clips are outcome-selected examples of successes and failures, not rate estimates.',''];(root/'report.md').write_text('\n'.join(lines))
        for (case,method),r in selected.items():
            if method==primary:continue
            source=next(p for p,rr in sources.items() if any(v['case']==case and v['method']==method for v in rr));stem=f'{case}_{method}_frozen_bias'
            for folder,exts in (('trials',('json',)),('states',('npz','xml'))):
                for ext in exts:shutil.copy2(source/folder/f'{stem}.{ext}',root/folder/f'{stem}.{ext}')
        save(root/'video_records.json',list(selected.values()));report=Path(__file__).with_name('TECHNICAL_REPORT.md');content=report.read_text(encoding='utf-8');assert f'## {experiment}:' not in content;report.write_text(content+f'\n## {experiment}: complete calibration-prior comparison\n\n'+text+f' Report: `{root}/report.md`.\n',encoding='utf-8')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,4));ax.bar(methods,[stats[m]['all']['success'] for m in methods]);ax.set_ylim(0,96);ax.tick_params(axis='x',rotation=15);ax.set_ylabel('Completed fitting tasks /96');ax.set_title(experiment+' | known bodies'+(' | privileged reference' if experiment=='E149' else ''));fig.tight_layout();fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
        st=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT',module+' import ROOT').replace("ROOT/'records.json'","ROOT/'video_records.json'").replace('E128',experiment).replace('Task-value ranking diagnosis','Calibration-prior comparison').replace('Outcome-selected excluded-fitting-family pairs: E120 on versus short correction-off then E120. Diagnosis only; no new policy.','Outcome-selected complete fitting successes and failures. Known bodies; fresh test closed. Exact-map clips are explicitly privileged references.')
        if experiment=='E149':st=st.replace('Measured-history control | No physics selector or geometry filter','PRIVILEGED EXACT MAP where labeled | Fitting reference only')
        Path(__file__).with_name(module+'_videos.py').write_text(st);shutil.copy2(__file__,root/'source'/Path(__file__).name);print(experiment,'prepared',len(selected),'videos',flush=True)

if __name__=='__main__':run()
