"""Build isolated E170 runner from audited E169, without editing the reference."""
from pathlib import Path

def run():
    p=Path(__file__).with_name('probe_gripper_clearance.py');s=p.read_text()
    s=s.replace('E169','E170').replace("REFERENCE=prior.ROOT/'task_projection'","REFERENCE=Path('outputs/ease_probe_gripper_clearance_v1/both')")
    s=s.replace("ROOT=Path('outputs/ease_probe_gripper_clearance_v1')","ROOT=Path('outputs/ease_descent_execution_v1')")
    s=s.replace("CONDITIONS=('probe_only','gripper_only','both')","CONDITIONS=('fixed_allowance','history_allowance')")
    start=s.index('def prepare():');end=s.index('\ndef run(condition):',start)
    s=s[:start]+'''def prepare():
    ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir();shutil.copy2(BASE/'cases.json',ROOT/'cases.json')
    save(ROOT/'protocol.json',dict(experiment='E170',scope='Privileged fitting diagnostic: two fixed variants x48 six-joint cases. E120+release baseline37/48; E169 both/probe-protected reference41/48 with4 collisions. Four exact E169 both replays before interventions. No held execution or training.',intervention='Keep E169 both geometry/probe protection. Only descent palm/pad rows receive a one-step clearance execution-error allowance. fixed_allowance:3mm; history_allowance:1mm plus maximum positive predicted-minus-actual clearance error over previous8 eligible descent steps, capped at20mm before the1mm floor. This uses past measured errors, never future outcomes. Same fixed parameters for every case. Convert allowance to effective-distance subtraction allowance/alpha, leaving actual geometry and reporting unchanged. Initial history empty; reset each episode. No outcome-dependent tuning.',preserved='E120 weights, frozen estimator bias, calibration, probe schedule/aperture, all phase budgets/goals and265mm release fix, strict contact/success scoring. Same linear QP, activation and fallback. No safety guarantee. Full gripper transforms privileged; gripper plane conservative outside table footprint.',screen='More than41 successes, fewer than4 collision episodes and zero lost reference successes. Report all failures and stalls even if screen fails.',inputs={str(p):fingerprint(p) for p in (REFERENCE/'records.json',BASE/'records.json',Path(__file__).with_name('probe_gripper_clearance.py'),Path(prior.__file__))}))
    for name in (*CONDITIONS,'replay'):
        root=ROOT/name;root.mkdir()
        for folder in ('trials','states','actions','histories','clearance'):(root/folder).mkdir()
    for name in ('descent_execution.py','probe_gripper_clearance.py','link_clearance_diagnostic.py','build_descent_execution.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py')]})
''' + s[end:]
    s=s.replace("protect_probe=condition in ('probe_only','both');gripper=condition in ('gripper_only','both')","protect_probe=True;gripper=True")
    s=s.replace('logs=[]','logs=[];error_history=[]')
    s=s.replace("eligible=bool((self.phase!='probe' or protect_probe) and np.min(d)<=prior.ACTIVATION)","allowance=np.zeros(len(d));effective=d.copy()\n                if self.phase=='descend' and condition!='replay':\n                    allowance[6:]=.003 if condition=='fixed_allowance' else .001+min(.020,max(error_history[-8:],default=0.))\n                    effective-=allowance/prior.ALPHA\n                eligible=bool(np.min(effective)<=prior.ACTIVATION)")
    s=s.replace("prior.project(nominal,c.bias,before['q'],d,g,metric)","prior.project(nominal,c.bias,before['q'],effective,g,metric)")
    s=s.replace("logs.append(dict(phase=self.phase,", "prediction=None if g is None else d+g@(np.asarray(action)-c.bias)\n                if self.phase=='descend' and prediction is not None:error_history.append(float(max(0.,np.max(prediction[6:]-post[6:]))))\n                logs.append(dict(allowance=allowance.tolist(),effective_distance=effective.tolist(),predicted_distance=None if prediction is None else prediction.tolist(),full_qpos_before=qpos.tolist(),full_qvel_before=qvel.tolist(),phase=self.phase,")
    # Both variants retain exact E169 replay logic; remove obsolete pre-probe branch.
    start=s.index("        elif condition=='probe_only':");end=s.index('        r.update(',start);s=s[:start]+s[end:]
    s=s.replace("r.update(condition=condition,", "r.update(condition=condition,")
    Path(__file__).with_name('descent_execution.py').write_text(s)

if __name__=='__main__':run()
