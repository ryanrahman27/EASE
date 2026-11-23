"""Isolate task-objective QP from the frozen E170 collision-free controller."""
from pathlib import Path

def run():
    s=Path(__file__).with_name('descent_execution.py').read_text().replace('E170','E171')
    s=s.replace("REFERENCE=Path('outputs/ease_probe_gripper_clearance_v1/both')","REFERENCE=Path('outputs/ease_descent_execution_v1/history_allowance')")
    s=s.replace("ROOT=Path('outputs/ease_descent_execution_v1')","ROOT=Path('outputs/ease_descent_progress_v1')")
    s=s.replace("CONDITIONS=('fixed_allowance','history_allowance')","CONDITIONS=('direct_task','limit_gated')").replace('REPLAYS=(0,1,13,27)','REPLAYS=(0,26,62,67)')
    start=s.index('def prepare():');end=s.index('\ndef run(condition):',start)
    s=s[:start]+'''def prepare():
    assert (ROOT/'diagnosis.json').exists();(ROOT/'source').mkdir();shutil.copy2(BASE/'cases.json',ROOT/'cases.json')
    save(ROOT/'protocol.json',dict(experiment='E171',scope='Fitting-only privileged control diagnostic; two variants x48 six-joint cases and4 exact E170 history-allowance replays. E170 reference44/48 with0 collisions. Original E120+release37/48 remains baseline. No held execution or training.',diagnosis='All four failures saturate joint5 near+2.5rad during descent, with active clearance constraints and little or negative late progress. Nominal damped least-squares action is independently clipped before metric projection, potentially distorting its task direction.',intervention='direct_task: on every descent step, minimize the original E120-estimated task residual directly subject to all original bounds/clearance constraints. limit_gated: do this only if any measured absolute arm joint coordinate>=2.4rad. Equivalent QP target is the UNCLIPPED damped least-squares command J.T solve(J J.T+.015^2 I,.35 error)+frozen_bias, with metric J.T J+.015^2 I. Original projection minimizes distance to the CLIPPED command. No new Jacobian or goal source. All thresholds and objectives frozen before outcomes.',preserved='History-based descent error allowance from E170, arm/palm/pad constraints, probe protection/schedule/aperture, frozen E120 checkpoints and bias, release265mm, phase budgets and strict scoring. No extra steps or outcome-based fallback. Same solver and logged hold fallback; no safety guarantee.',screen='More than44 successes, zero collisions, zero lost E170 successes. Retain and report every outcome; no automatic baseline replacement.',inputs={str(p):fingerprint(p) for p in (REFERENCE/'records.json',BASE/'records.json',ROOT/'diagnosis.json',Path(__file__).with_name('descent_execution.py'),Path(prior.__file__))}))
    for name in (*CONDITIONS,'replay'):
        root=ROOT/name;root.mkdir()
        for folder in ('trials','states','actions','histories','clearance'):(root/folder).mkdir()
    for name in ('descent_progress.py','descent_execution.py','probe_gripper_clearance.py','link_clearance_diagnostic.py','build_descent_progress.py','descent_progress_diagnosis.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py')]})
''' + s[end:]
    s=s.replace("if self.phase=='descend' and condition!='replay':","if self.phase=='descend':")
    s=s.replace(".003 if condition=='fixed_allowance' else .001+min(.020,max(error_history[-8:],default=0.))",".001+min(.020,max(error_history[-8:],default=0.))")
    s=s.replace('eligible=bool(np.min(effective)<=prior.ACTIVATION)',"direct=bool(self.phase=='descend' and condition!='replay' and (condition=='direct_task' or np.max(np.abs(before['q']))>=2.4))\n                optimization_target=nominal.copy();eligible=bool(direct or np.min(effective)<=prior.ACTIVATION)")
    old="metric=j.T@j+.015**2*np.eye(self.n);action,solve,rhs=prior.project(nominal,c.bias,before['q'],effective,g,metric)"
    new="""metric=j.T@j+.015**2*np.eye(self.n)
                    if direct:
                        error=np.r_[self.goal-before['position'],.15*(np.array([0.,0.,1.])-before['rotation'][:,2])]
                        optimization_target=j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*error)+c.bias
                    action,solve,rhs=prior.project(optimization_target,c.bias,before['q'],effective,g,metric)"""
    assert old in s;s=s.replace(old,new)
    s=s.replace('logs.append(dict(allowance=', 'logs.append(dict(direct_task=direct,optimization_target=optimization_target.tolist(),estimated_jacobian=None if not eligible else j.tolist(),allowance=')
    Path(__file__).with_name('descent_progress.py').write_text(s)

if __name__=='__main__':run()
