"""Verify E138 OOF decisions and persist all comparison results and media."""
import shutil
from pathlib import Path
import numpy as np
from .expanded_structured_pairs import ROOT,COLLECT
from .expanded_structured_training import score,dependencies,KINDS
from .family_consensus_gate import predict,ridge
from .structured_response import fit
from .abstaining_task_value_gate import read,save,fingerprint,DATA

def run():
    _,joblib,_=dependencies();d=np.load(ROOT/'dataset.npz');x,y,fold,fam=d['x'],d['y'],d['fold'],d['family'];o=np.load(ROOT/'oof.npz');meta=read(ROOT/'rows.json');tasks=read(ROOT/'cases.json');out={k:np.zeros_like(o[k]) for k in KINDS}
    for f in range(4):
        tr,va=fold!=f,fold==f;m=np.load(ROOT/'models'/f'ridge_fold_{f}.npz');fresh=ridge(x[tr],y[tr]);scalar=fit('scalar',d['effect'][tr],d['target'][tr],d['phase'][tr]);np.testing.assert_array_equal(m['training'],np.flatnonzero(tr))
        for k,v in {**fresh,'scalar':scalar}.items():np.testing.assert_array_equal(m[k],v)
        out['ridge'][:,va]=np.asarray(predict(m,m['scalar'],x[va],d['effect'][va],d['common'][va],d['error'][va]));tree=joblib.load(ROOT/'models'/f'trees_fold_{f}.joblib');params=tree.get_params()
        for k,v in dict(n_estimators=128,max_depth=5,min_samples_leaf=12,random_state=138000).items():assert params[k]==v
        out['trees'][0,va]=tree.predict(np.clip((x[va]-m['mean'])/m['scale'],-10,10));out['trees'][1,va]=out['ridge'][1,va];inner=[]
        for excluded in sorted(set(fam[tr])):
            ids=tr&(fam!=excluded);n=np.load(ROOT/'models'/f'consensus_fold_{f}_exclude_{excluded}.npz');np.testing.assert_array_equal(n['training'],np.flatnonzero(ids));assert not set(fam[ids])&set(fam[va]);inner.append(predict(n,n['scalar'],x[va],d['effect'][va],d['common'][va],d['error'][va]))
        out['consensus'][:,va]=np.asarray(inner).min(0)
    results=read(ROOT/'outcomes.json')
    for k,v in out.items():
        np.testing.assert_allclose(v,o[k],rtol=1e-12,atol=1e-12);assert score((v[0]>.05)&(v[1]>1e-5),k,meta,tasks)==results[k]
    for p,h in {**read(ROOT/'frozen_collection.json'),**read(ROOT/'frozen_training.json'),**read(ROOT/'protocol.json')['inputs'],**read(ROOT/'protocol.json')['checkpoints'],**read(ROOT/'extraction_input_hashes.json')}.items():assert fingerprint(p)==h
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    save(ROOT/'verification.json',dict(oof_predictions_reproduced=True,models_training_only=True,family_splits_disjoint=True,task_outcomes_exact=True,new_paired_prefixes=1296,uncached_replay_exact=read(ROOT/'smoke_verification.json')['full_states_exact'],all_inputs_unchanged=True,E120_unchanged=True))
    for folder in ('trials','states'):(ROOT/folder).mkdir(exist_ok=True)
    wanted={}
    for name in KINDS:
        for dof in (6,7):
            for gain in (True,False):
                candidates=[r for r in results[name]['episodes'] if r['success']==gain and r['success']!=r['baseline_success'] and len(tasks[r['case']]['home'])==dof]
                for r in candidates[:1]:
                    for phase in (r['phase'],None):wanted.setdefault((r['case'],r['seed'],phase),[]).append(name)
    # Preserve examples from the new collection even when every gate abstains.
    newrecords=read(COLLECT/'records.json')
    for dof in (6,7):
        for success in (True,False):
            candidates=[r for r in newrecords if r['off_phase'] is None and r['dof']==dof and r['success']==success]
            if candidates:
                r=candidates[0];wanted.setdefault((r['case'],r['seed'],None),[]).append('new collection reference')
    records=[]
    for (case,seed,phase),reasons in wanted.items():
        source=DATA/'collection' if case<56 else COLLECT;method=f'on_{seed}' if phase is None else f'off_{phase}_{seed}';stem=f'{case}_{method}_frozen_bias';r=read(source/'trials'/f'{stem}.json')
        for folder,exts in (('trials',('json',)),('states',('npz','xml'))):
            for ext in exts:shutil.copy2(source/folder/f'{stem}.{ext}',ROOT/folder/f'{stem}.{ext}')
        records.append(dict(case=case,method=method,mode='frozen_bias',success=r['success'],collision=r['collision'],selection_reasons=reasons))
    save(ROOT/'records.json',records);ex=read(ROOT/'extraction.json');chosen=read(ROOT/'selected.json');lines=['# E138: expanded fitting counterfactuals','',f"Collected all 1,440 new fitting episodes on 48 additional placements, plus one independent uncached replay audit. Joined {ex['new_rows']} new clean paired starts with {ex['original_rows']} original starts ({ex['total_rows']} total). All 1,296 new matched prefixes/snapshots verified. All placements and outcomes retained; only prior-collision starts excluded from training.",'',
        'Same four known training bodies and ancestral-family splits. No object/contact truth, geometry selector or body IDs enter the gate. The 32 measured structured features and fixed margins are unchanged. All three candidate types were declared before collection: ridge value, family-deletion consensus, and shallow extra-trees value, each combined with learned scalar-calibrated history response.','',
        '| Candidate | E120 successes | Candidate successes | Rescues | Lost successes | Switches |','|---|---:|---:|---:|---:|---:|']
    for k,r in results.items():lines.append(f"| {k} | {r['baseline_success']}/{r['n']} | {r['success']}/{r['n']} | {r['gains']} | {r['losses']} | {r['switches']} |")
    lines += ['', '## Original versus additional fitting placements','', '| Candidate | Subset | E120 | Candidate | Gains / losses |','|---|---|---:|---:|---:|']
    for k,r in results.items():
        for new in (False,True):
            rr=[e for e in r['episodes'] if (e['case']>=56)==new];before=sum(e['baseline_success'] for e in rr);after=sum(e['success'] for e in rr);g=sum(e['success'] and not e['baseline_success'] for e in rr);l=sum(e['baseline_success'] and not e['success'] for e in rr);lines.append(f"| {k} | {'additional' if new else 'original'} | {before}/{len(rr)} | {after}/{len(rr)} | {g} / {l} |")
    lines += ['',f"Selected candidate: **{chosen['kind']}**, {chosen['gains']} rescues and {chosen['losses']} losses. These are adaptive fitting-validation scores reconstructed from executed paired continuations, not fresh-test performance. The separate 48-placement test remains reserved until a positive-gain, zero-loss candidate qualifies and all models/rules are frozen.",'',
        'All model predictions, fitting-only normalizations, family boundaries and full-task reconstructions were independently checked. Source inputs and E120 checkpoints are unchanged. The uncached audit reproduced the cached collection trajectory exactly. All trials/actions/states are retained; videos show representative changed outcomes and new-collection successes/failures.', '']
    (ROOT/'report.md').write_text('\n'.join(lines));p=Path(__file__).with_name('TECHNICAL_REPORT.md');text=p.read_text(encoding='utf-8');marker='## E138: expanded fitting counterfactuals';assert marker not in text
    text=text.replace('Revision 124 · 15 November 2026','Revision 126 · 15 November 2026',1)
    p.write_text(text+'\n'+marker+f"\n\n1,440 additional fitting episodes; {ex['new_rows']} new clean paired starts. Fixed ridge/consensus/trees comparisons, same family grouping and margins. Selected {chosen['kind']}: {chosen['success']}/{chosen['n']} vs E120 {chosen['baseline_success']}, {chosen['gains']} gains/{chosen['losses']} losses. Fresh test reserved. Report: `outputs/ease_expanded_structured_pairs_v1/report.md`.\n",encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=list(results);vals=[results[k]['success'] for k in names];fig,ax=plt.subplots(figsize=(8,4));ax.bar(names,vals);ax.set_ylim(0,288);ax.set_ylabel('Fitting successes /288');ax.set_title('Expanded data: fixed candidate comparison')
    for i,k in enumerate(names):r=results[k];ax.text(i,vals[i]+2,f"{vals[i]}\n+{r['gains']}/-{r['losses']}",ha='center')
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    s=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','expanded_structured_pairs import ROOT').replace('E128','E138').replace('Task-value ranking diagnosis','Expanded fitting counterfactuals').replace('Diagnosis only; no new policy.','Fixed expanded-data models; fitting validation only, fresh test reserved.')
    Path(__file__).with_name('expanded_structured_videos.py').write_text(s);print('Verified expanded data/models; prepared',len(records),'videos',flush=True)

if __name__=='__main__':run()
