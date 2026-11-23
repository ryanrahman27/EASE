"""E172 local late-stall posture recovery, preserving E170 outside activation."""
from pathlib import Path

def run():
    s=Path(__file__).with_name('descent_progress.py').read_text().replace('E171','E172').replace("ROOT=Path('outputs/ease_descent_progress_v1')","ROOT=Path('outputs/ease_descent_stall_v1')").replace("CONDITIONS=('direct_task','limit_gated')","CONDITIONS=('stall_recovery',)")
    start=s.index('def prepare():');end=s.index('\ndef run(condition):',start)
    s=s[:start]+'''def prepare():
    assert (ROOT/'diagnosis_60_local.json').exists();(ROOT/'source').mkdir();shutil.copy2(BASE/'cases.json',ROOT/'cases.json')
    save(ROOT/'protocol.json',dict(experiment='E172',scope='One fixed fitting candidate x48 six-joint cases, plus4 exact E170 replays. Adaptive fitting development after E171 regressions; no held evidence. E17044/48 zero collisions remains protected reference; E120+release primary baseline unchanged.',gate='Measured local late descent stall: at least60 descent observations,25mm<position error<75mm, less than1mm error reduction over previous19 intervals, any absolute joint>=2.4rad. Latch for rest of descent. All thresholds frozen before E172 outcomes. Read-only gate diagnosis on E170 triggers only cases26,62,65,67; no case IDs or body identities enter control. Earlier broader stall rules also triggered successful case43 and are archived.',recovery='When active, direct E120 task objective plus joint-limit posture regularization: H=J.T J+.03^2 I, preferred relative command -0.02*sign(q) on joints abs(q)>2.35, zero elsewhere. Unconstrained center solve(H,J.T*.35*error+.03^2*posture)+frozen_bias, projected with same bounds and clearance constraints. Outside active descent use exact E170 action/protection. No extra steps, new task Jacobian, phase goal or aperture change.',preserved='E120 weights/frozen bias, calibration, history-based descent clearance allowance, probe protection/schedule,265mm release, original task budgets and strict scoring. Same QP solver and logged fallback. Privileged geometry diagnostic; not guaranteed safety or learned-only transfer.',screen='More than44 successes, zero collisions, no lost E170 successes. Also report all four descent errors and exact trace/state retention for every nonactivated episode.',inputs={str(p):fingerprint(p) for p in (REFERENCE/'records.json',ROOT/'diagnosis_60_local.json',Path(__file__).with_name('descent_progress.py'))}))
    for name in (*CONDITIONS,'replay'):
        root=ROOT/name;root.mkdir()
        for folder in ('trials','states','actions','histories','clearance'):(root/folder).mkdir()
    for name in ('descent_stall.py','descent_progress.py','descent_execution.py','probe_gripper_clearance.py','link_clearance_diagnostic.py','build_descent_stall.py','descent_stall_diagnosis.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py')]})
''' + s[end:]
    old="direct=bool(self.phase=='descend' and condition!='replay' and (condition=='direct_task' or np.max(np.abs(before['q']))>=2.4))"
    new="""if self.phase=='descend':
                    pe=float(np.linalg.norm(self.goal-before['position']));self.descent_errors.append(pe)
                    if condition!='replay' and len(self.descent_errors)>=60 and .025<pe<.075 and self.descent_errors[-20]-pe<.001 and np.max(np.abs(before['q']))>=2.4:self.recovery_active=True
                direct=bool(self.phase=='descend' and self.recovery_active);posture=np.zeros(self.n)"""
    assert old in s;s=s.replace(old,new)
    s=s.replace("optimization_target=j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*error)+c.bias","posture=np.where(np.abs(before['q'])>2.35,-.02*np.sign(before['q']),0.);metric=j.T@j+.03**2*np.eye(self.n)\n                        optimization_target=np.linalg.solve(metric,j.T@(.35*error)+.03**2*posture)+c.bias")
    s=s.replace('logs.append(dict(direct_task=direct,','logs.append(dict(posture=posture.tolist(),descent_observations=len(self.descent_errors),direct_task=direct,')
    s=s.replace('env.clearance_controller=c;','env.descent_errors=[];env.recovery_active=False;env.clearance_controller=c;')
    Path(__file__).with_name('descent_stall.py').write_text(s)

if __name__=='__main__':run()
