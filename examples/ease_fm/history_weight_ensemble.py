"""E143: prespecified all-seed averaging, never best-seed selection."""
from pathlib import Path
import shutil
import numpy as np
from .anchored_history_weights import ROOT as PRIOR
from .learned_history_weights import ROOT as DATA_ROOT
from .abstaining_task_value_gate import read,save,fingerprint

ROOT=Path('outputs/ease_history_weight_ensemble_v1')

def run():
    ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir();inputs=[PRIOR/'oof.npz',PRIOR/'references.npz',PRIOR/'summary.json',PRIOR/'frozen.json',DATA_ROOT/'dataset.npz']
    save(ROOT/'protocol.json',dict(experiment='E143',inputs={str(p):fingerprint(p) for p in inputs},design='E142 average improves position MSE about5.9% over the tuned constant, but one individual seed is slightly worse and fails its all-seed screen. Test a new fixed predictor averaging all three Jacobians. Do not choose a seed or ensemble size using validation. Evaluate all three leave-one-seed-out pairs as robustness diagnostics. Averaging Jacobians is equivalent to averaging the linear next-motion predictions for a fixed measured joint query.',
        gate='All-three mean position MSE<=.95x selected constant; scaled rotation MSE<=1.05x; neither joint count worse; each leave-one-seed-out pair position MSE no worse than constant. No task-success labels or held/fresh outcomes used. No retraining. This is an ensemble method, not a single-seed Transformer gain.',
        control='If the fixed ensemble passes, compare complete fitting episodes against E120, an E120 ensemble, the tuned constant kernel, uniform32, and original analytical history. Preserve all regressions. E120 remains immutable; no promotion from prediction scores alone.'))
    d=np.load(DATA_ROOT/'dataset.npz');truth=d['target'];dof=d['mask'].sum(1);o=np.load(PRIOR/'oof.npz')['prediction'];constant=np.load(PRIOR/'references.npz')['selected'];pred={'all_three':o.mean(0),**{f'without_{i}':np.delete(o,i,axis=0).mean(0) for i in range(3)},'constant':constant};metrics={}
    for name,p in pred.items():
        metrics[name]={}
        for count in ('all',6,7):
            ids=np.ones(len(dof),bool) if count=='all' else dof==count;e=p[ids]-truth[ids];metrics[name][str(count)]=dict(position_mse=float(np.mean(e[:,:,:3]**2)),position_rmse_mm=float(np.sqrt(np.mean(e[:,:,:3]**2))*1000),rotation_mse=float(np.mean(e[:,:,3:]**2)),rotation_rmse_mm=float(np.sqrt(np.mean(e[:,:,3:]**2))*1000))
    a,b=metrics['all_three'],metrics['constant'];ratio=a['all']['position_mse']/b['all']['position_mse'];passed=ratio<=.95 and a['all']['rotation_mse']<=1.05*b['all']['rotation_mse'] and all(a[str(k)]['position_mse']<=b[str(k)]['position_mse'] for k in (6,7)) and all(metrics[f'without_{i}']['all']['position_mse']<=b['all']['position_mse'] for i in range(3))
    np.savez_compressed(ROOT/'oof.npz',**pred);save(ROOT/'summary.json',dict(metrics=metrics,position_mse_ratio=ratio,qualifies_for_control=bool(passed),task_rollouts=0,fresh_evaluated=False,E120_unchanged=True));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'oof.npz',ROOT/'summary.json',ROOT/'source'/Path(__file__).name]});print(read(ROOT/'summary.json'),flush=True)

if __name__=='__main__':run()
