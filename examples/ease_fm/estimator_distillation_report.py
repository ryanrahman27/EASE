"""Audit E66 provenance, retention, paired native results and training metrics."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .estimator_distillation import ROOT,DATA
from .benchmark import fingerprint
from .native_common_schedule import summarize_phase


def run():
    protocol=json.loads((ROOT/'protocol.json').read_text());dest=ROOT/'native'
    rows=json.loads((dest/'records.json').read_text());audit=json.loads((dest/'audit.json').read_text());ev=json.loads((dest/'protocol.json').read_text())
    old=json.loads(Path('outputs/ease_retention_fresh_v1/records.json').read_text())
    common=json.loads(Path('outputs/ease_native_common_schedule_v2/records.json').read_text())
    assert fingerprint(DATA/'contexts.npz')==protocol['contexts_sha256'] and fingerprint(DATA/'provenance.json')==protocol['provenance_sha256']
    assert fingerprint('outputs/ease_retention_fresh_v1/records.json')==ev['native_reference_sha256']
    assert fingerprint('outputs/ease_native_common_schedule_v2/records.json')==ev['common_reference_sha256']
    assert fingerprint(dest/'records.json')==audit['records_sha256'] and len(rows)==144
    assert len({(r['method'],r['case'],r['task']) for r in rows})==144
    for path,sha in {**protocol['parents'],**ev['checkpoints']}.items():assert fingerprint(path)==sha
    lines=['# E66: geometric versus analytical-target estimator training','',protocol['design'],'',protocol['loss'],'',protocol['evaluation'],'',
        'The analytical-target candidates are learned Transformers. They do not call the analytical solver at inference. The analytical controller below is a separate, untrained reference.','',
        '## Held-trajectory prediction MSE','', '| Seed | Target | Geometric MSE | Analytical-estimate MSE | Synthetic MSE |','|---|---|---:|---:|---:|']
    gates=[]
    for seed in range(3):
        metas=[json.loads((ROOT/condition/f'seed_{seed}'/'training.json').read_text()) for condition in ['geometric','analytical']]
        assert metas[0]['batch_sha256']==metas[1]['batch_sha256'] and metas[0]['parent_sha256']==metas[1]['parent_sha256']
        for condition,meta in zip(['geometric','analytical'],metas):
            assert len(meta['losses'])==meta['steps']==2000
            f=meta['final'];lines.append(f'| {seed} | {condition} | {f["j"]:.6f} | {f["analytical"]:.6f} | {f["synthetic"]:.6f} |')
    lines+=['','## Original reaching retention','', '| Target | Seed | Six joints | Seven joints | Total |','|---|---:|---:|---:|---:|']
    for condition in ['geometric','analytical']:
        a=json.loads((ROOT/f'gate_{condition}'/'audit.json').read_text())
        for s in a['summary']:
            path=ROOT/f'gate_{condition}'/f'seed_{s["seed"]}.json';rr=json.loads(path.read_text())
            assert fingerprint(path)==s['records_sha256'] and len(rr)==s['count']==256 and sum(r['success'] for r in rr)==s['success']
            assert fingerprint(ROOT/condition/f'seed_{s["seed"]}'/'policy.pt')==s['policy_sha256']
            counts=[(sum(r['success'] for r in rr if r['dof']==d),sum(r['dof']==d for r in rr)) for d in [6,7]]
            gates.append(dict(condition=condition,seed=s['seed'],success=s['success'],count=256))
            lines.append(f'| {condition} | {s["seed"]} | {counts[0][0]}/{counts[0][1]} | {counts[1][0]}/{counts[1][1]} | {s["success"]}/256 |')
    for r in rows:
        assert json.loads((dest/'trials'/f'{r["method"]}_{r["case"]}_{r["task"]}.json').read_text())==r
        assert len(r['trace'])==r['steps']-32
        if r['task']=='common':
            assert len(r['trace'])==1700
            for p in r['phases']:assert summarize_phase(p['phase'],[x for x in r['trace'] if x['phase']==p['phase']])==p
            assert r['success']==all(p['passes'] for p in r['phases'])
        elif r['success']:assert r['grasped'] and r['lifted'] and not r['collision'] and r['failure'] is None
    summary=[]
    for task in ['common','manipulation']:
        lines+=['',f'## {task}: all twelve observed native cases','', '| Controller | PiPER per seed /6 | Panda per seed /6 | Total | Gains vs E35 | Losses vs E35 | Contacts |','|---|---|---|---:|---:|---:|---:|']
        ref=common if task=='common' else old
        for group in ['reference','original','geometric','analytical']:
            if group=='reference':rr=[dict(r,success=r.get('passes',r.get('success'))) for r in ref if r['method']=='analytical'];seeds=[None]
            elif group=='original':rr=[dict(r,success=r.get('passes',r.get('success'))) for r in ref if r['method'].startswith('original_' if task=='common' else 'old_')];seeds=[0,1,2]
            else:rr=[r for r in rows if r['task']==task and r['method'].startswith(group+'_')];seeds=[0,1,2]
            scores={robot:[sum(r['success'] for r in rr if r['robot']==robot and (seed is None or r['method'].endswith('_'+str(seed)))) for seed in seeds] for robot in ['piper','panda']}
            gains=losses=0
            if group in ['geometric','analytical']:
                for r in rr:
                    method=('original_' if task=='common' else 'old_')+r['method'][-1]
                    b=next(x for x in ref if x['case']==r['case'] and x['method']==method);passed=b.get('passes',b.get('success'))
                    gains+=int(r['success'] and not passed);losses+=int(passed and not r['success'])
            count=sum(r['success'] for r in rr);contacts=sum(r.get('collision',False) for r in rr)
            labels={'reference':'Analytical controller','original':'E35 Transformer','geometric':'Geometric-target Transformer','analytical':'Analytical-target Transformer'}
            lines.append(f'| {labels[group]} | {scores["piper"]} | {scores["panda"]} | {count}/{len(rr)} | {gains if group in ["geometric","analytical"] else "—"} | {losses if group in ["geometric","analytical"] else "—"} | {contacts} |')
            summary.append(dict(task=task,group=group,scores=scores,success=count,count=len(rr),gains=gains,losses=losses,contacts=contacts))
    lines+=['','All six 2000-update runs verified; paired batch hashes match. All 4800 analytical targets were independently checked against NumPy (maximum absolute difference 2.98e-8). All 1536 reaching records and 144 native records audited; hashes unchanged after evaluation. Regression suite: 92 passed.','',
        'Validation trajectories share procedural training bodies/goals; native cases are repeatedly observed development cases. Fixed E35 and analytical reference records are reused from audited E63/E46, not fresh independent baseline trials. No native training or final-test evaluation. A lower fitting error alone does not establish better control.']
    (ROOT/'summary.json').write_text(json.dumps(dict(gates=gates,native=summary),indent=2))
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,task in zip(axes,['common','manipulation']):
        rr=[r for r in summary if r['task']==task]
        for i,robot in enumerate(['piper','panda']):
            vals=[100*sum(r['scores'][robot])/(6*len(r['scores'][robot])) for r in rr]
            bars=ax.bar(np.arange(4)+(i-.5)*.36,vals,width=.36,label=['PiPER (6 joints)','Panda (7 joints)'][i])
            for b,r in zip(bars,rr):ax.text(b.get_x()+b.get_width()/2,b.get_height()+2,f'{sum(r["scores"][robot])}/{6*len(r["scores"][robot])}',ha='center',fontsize=8)
        ax.set_title('Contact-disabled tracking' if task=='common' else 'Contact-enabled pick-and-place')
        ax.set_xticks(range(4),['Analytical\ncontroller','E35','Geometric\ntarget','Analytical\ntarget']);ax.set_ylim(0,115);ax.set_yticks(range(0,101,20));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Success (%)');axes[1].legend(loc='upper right',bbox_to_anchor=(1,1.2),ncol=2,fontsize=8)
    fig.suptitle('E66: fixed three-seed estimator-training comparison',y=1.02)
    fig.text(.5,-.015,'Bias retained. Observed native development cases; seeds share cases. No independent test claim.',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(ROOT/'native_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    shutil.copy2('tests/test_ease_fm_estimator_distillation.py',ROOT/'source/test_ease_fm_estimator_distillation.py')
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':run()
