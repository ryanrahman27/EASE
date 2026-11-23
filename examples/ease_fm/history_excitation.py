"""E94: read-only excitation audit of E93 histories and exact trajectory replays."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import local_history as prior
from .mixed_sim import AXES
from .benchmark import fingerprint

ROOT=Path('outputs/ease_history_excitation_v1')


def excitation(gram):
    # The existing update preserves a1e-6 isotropic ridge exactly in real arithmetic.
    covariance=np.asarray(gram)-1e-6*np.eye(len(gram))
    eigen=np.maximum(np.linalg.eigvalsh((covariance+covariance.T)/2),0)
    energy=float(eigen.sum());squares=float(eigen@eigen)
    return dict(eigenvalues=eigen.tolist(),energy=energy,effective_rank=energy*energy/squares if squares>0 else 0.,
        above_ridge=int(np.sum(eigen>1e-6)),smallest_to_ridge=float(eigen[0]/1e-6),largest_fraction=float(eigen[-1]/energy) if energy else 0.)


def replay(case,t,method,model,mode):
    profiles=[]
    class AuditPose(prior.LocalPose):
        def __init__(self,n,model):
            super().__init__(n,model);self.shadow=1e-6*np.eye(n);self.audit_env=None
        def update(self,before,after,command):
            dq=after['q']-before['q'];self.shadow=.98*self.shadow+np.outer(dq,dq)+2e-8*np.eye(self.n)
            super().update(before,after,command)
            if self.audit_env is not None:
                recent=1e-6*np.eye(self.n)
                for q,_ in self.history:recent=.98*recent+np.outer(q,q)+2e-8*np.eye(self.n)
                profiles.append(dict(step=self.audit_env.steps,phase=self.audit_env.phase,history_count=len(self.history),long=excitation(self.shadow),recent=excitation(recent)))
    def attach(env):env.controller.audit_env=env
    with patch.object(prior,'LocalPose',AuditPose):
        result=prior.evaluate(case,t,method,model,mode,attach)
    old=json.loads((prior.ROOT/'trials'/f'{case}_{method}_{mode}.json').read_text());assert result==old
    assert len(profiles)==len(result['trace'])
    assert all(p['phase']==t['phase'] for p,t in zip(profiles,result['trace']))
    return result,profiles


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'profiles').mkdir();(ROOT/'source').mkdir()
    summary=json.loads((prior.ROOT/'summary.json').read_text());records=json.loads((prior.ROOT/'records.json').read_text())
    selected=[dict(case=r['case'],method=r['method'],group=r['change']) for r in summary['changes']]
    for dof in (6,7):
        candidates=[r for r in records if r['mode']=='window32' and r['dof']==dof and r['success'] and r['local_history_applied'] and next(b['success'] for b in records if b['case']==r['case'] and b['method']==r['method'] and b['mode']=='long_history')]
        r=min(candidates,key=lambda r:(r['case'],r['method']));selected.append(dict(case=r['case'],method=r['method'],group='retained_control'))
    paths=[prior.ROOT/'summary.json',prior.ROOT/'records.json',prior.ROOT/'protocol.json',*sorted((prior.ROOT/'trials').glob('*.json'))]
    inputs={str(p):fingerprint(p) for p in paths};models,hashes=prior.models()
    spec=dict(experiment='E94',selection=selected,inputs=inputs,checkpoints=hashes,test_evaluated=False,
        design='Read all96 saved pre-close handoff statistic pairs. Replay all21 E93 changed outcome pairs plus earliest retained success per DOF, both modes:46 unchanged full tasks. Record post-update long-history and last32 joint-displacement Gram spectra every task step. Shadow long history follows actual executed trajectory; recent statistics use same32 transitions and original weighting. No action/statistic mutations, new data selection rule, hyperparameter tuning or retraining.',
        metrics='Subtract1e-6 isotropic ridge before PSD eigendecomposition; clip negative roundoff eigenvalues to0. Energy=sum eigenvalues; effective rank=(sum eigenvalues)^2/sum squares, zero for zero motion; count eigenvalues>1e-6 uses existing ridge magnitude, not an outcome-fitted threshold. Report largest-direction energy fraction. These are excitation diagnostics, not proofs of estimator error or safety.',
        scope='All handoffs are covered; temporal replays are outcome-selected and dependent. Compare long/recent on the same actual trajectory. Six E93 videos are reused with verified hashes because execution is unchanged, and labeled as reused evidence.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('history_excitation.py','local_history.py','learned_path_transfer.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    handoffs=[]
    for r in records:
        if r['mode']!='window32':continue
        h=json.loads((prior.ROOT/'trials'/f"{r['case']}_{r['method']}_window32.json").read_text())['handoff']
        handoffs.append(dict(case=r['case'],method=r['method'],dof=r['dof'],applied=r['local_history_applied'],long=excitation(h['gram']),recent=excitation(h['applied_gram']) if r['local_history_applied'] else None))
    (ROOT/'handoffs.json').write_text(json.dumps(handoffs,indent=2))
    tasks=json.loads((prior.PRIOR/'cases.json').read_text());AXES.update(json.loads((prior.PRIOR/'axis_catalog.json').read_text()));rows=[]
    for s in selected:
        for mode in ('long_history','window32'):
            r,p=replay(s['case'],tasks[s['case']],s['method'],models[s['method']],mode)
            (ROOT/'profiles'/f"{s['case']}_{s['method']}_{mode}.json").write_text(json.dumps(dict(**s,mode=mode,profiles=p)))
            rows.append(dict(**s,mode=mode,dof=r['dof'],success=r['success'],collision=r['collision'],steps=r['steps']))
        print(s,'exact replay verified',flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2));(ROOT/'audit.json').write_text(json.dumps(dict(exact_replays=len(rows),handoff_pairs=96,hashes_unchanged=True,test_evaluated=False),indent=2));(ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()
