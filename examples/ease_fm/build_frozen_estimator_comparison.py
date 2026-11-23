"""E176 adapter: preserve E175 controller, change estimator only."""
from pathlib import Path

def run():
    base=Path(__file__).parent;s=(base/'early_object.py').read_text()
    s=s.replace('E175','E176').replace("REFERENCE=Path('outputs/ease_object_preserving_v1/guarded_handoff')","REFERENCE=Path('outputs/ease_early_object_recovery_v1/early_guard')").replace("ROOT=Path('outputs/ease_early_object_recovery_v1')","ROOT=Path('outputs/ease_frozen_estimator_comparison_v1')").replace("CONDITIONS=('early_guard',)","CONDITIONS=('e120_ensemble','analytical')")
    start=s.index('def prepare():');end=s.index('\ndef run(condition):',start)
    s=s[:start]+'''def prepare():
    ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir();shutil.copy2(BASE/'cases.json',ROOT/'cases.json')
    _,_,hashes=old.broad.setup()
    save(ROOT/'protocol.json',dict(experiment='E176',scope='48 six-joint fitting cases x2 estimators=96 full episodes;5 exact E175 preflight replays. E120 full48 must also exactly reproduce E175. No held bodies or training.',methods='E120 unchanged three-model reference/correction ensembles and bounded residual versus original analytical LocalPose: solve(gram,cross).T, forgetting.98, ridge floor1e-6; accumulated history before probe, last32 transitions from probe onward. Both update motion statistics online and freeze identical calibrated bias. Analytical has no neural correction.',controller='Exact E175 early risk and late stall gates, strict object-safe grasp-first planner, route tracking, probe/table/gripper protection, causal clearance allowance, guarded close/lift,265mm release, task goals and scoring. Same simulator geometry/object/contact privilege. No estimator-specific retuning or case-ID controller rules.',budget='Both:32 calibration steps,32 pre-close probe steps, same action bounds/dt. Approach80, descend100 or300 only after accepted route, close40,lift100,transport100,lower100,release40,retreat80,settle40. Same reach early-exit rule. Maximum944 control steps including calibration when routed,744 otherwise. Actual phase lengths may differ; report them, no forced padding or extra retries.',analysis='Report paired successes/losses, collisions, grasp/lift, trigger and abort counts, per-family results, total/median actual interactions and prediction errors. One deterministic paired rollout per fitting case; placements are not independent morphology samples. No superiority claim from a tie or a fitting-only result.',videos='Save all traces/states/actions/histories. Render both methods for cases0,26,43,62,65,67 and every discordant success/collision case, plus any missing success/failure category. Include archived E174 case26 failure for context, labeled historical and excluded from paired counts.',checkpoints=hashes,reference_sha256=fingerprint(REFERENCE/'records.json'),privileged=True))
    for name in (*CONDITIONS,'replay'):
        root=ROOT/name;root.mkdir()
        for folder in ('trials','states','actions','histories','clearance','planning'):(root/folder).mkdir()
    names=('frozen_estimator_comparison.py','build_frozen_estimator_comparison.py','early_object.py','early_object_plan.py','early_object_diagnosis.py','object_preserving_plan.py','object_preserving_planner.py','history_estimator_control.py','bounded_history_correction.py','training_estimator_core.py','local_history.py','pick_place_control.py','retention_fresh.py','arm_pick_place.py','local_probe.py','link_clearance_diagnostic.py')
    for name in names:shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py')]})
    save(ROOT/'status.json',dict(status='prepared',held_evaluated=False))
''' +s[end:]
    s=s.replace("root=ROOT/condition;protect_probe=True;gripper=True", "root=ROOT/condition;method='e120_ensemble' if condition=='replay' else condition;protect_probe=True;gripper=True")
    s=s.replace("early=condition!='replay' and early_gate(self.object_history)","early=early_gate(self.object_history)")
    s=s.replace("r=old.evaluate(case,t,'e120_ensemble',models,None,choices[folds[str(case)]])", "r=old.evaluate(case,t,method,models,None,choices[folds[str(case)]])")
    s=s.replace("stem=f'{case}_e120_ensemble_frozen_bias';save", "stem=f'{case}_{method}_frozen_bias';reference_stem=f'{case}_e120_ensemble_frozen_bias';save")
    s=s.replace("base=read(BASE/'trials'/f'{stem}.json')", "base=read(BASE/'trials'/f'{reference_stem}.json')")
    s=s.replace("reference=read(REFERENCE/'trials'/f'{stem}.json')", "reference=read(REFERENCE/'trials'/f'{reference_stem}.json')")
    s=s.replace("if condition=='replay':\n            for key", "if method=='e120_ensemble':\n            for key")
    s=s.replace("np.load(REFERENCE/'states'/f'{stem}.npz')", "np.load(REFERENCE/'states'/f'{reference_stem}.npz')")
    s=s.replace("before=self.observe();nominal=", "before=self.observe();estimator_jacobian,_=c.predict();estimator_gram=c.gram.copy();estimator_cross=c.cross.copy();nominal=")
    s=s.replace('logs.append(dict(trigger_reason=', 'logs.append(dict(estimator_jacobian=estimator_jacobian.tolist(),estimator_gram=estimator_gram.tolist(),estimator_cross=estimator_cross.tolist(),estimator_local=bool(c.local),trigger_reason=')
    s=s.replace("exact_replays=5 if condition=='replay' else 0", "exact_replays=len(rows) if method=='e120_ensemble' else 0")
    (base/'frozen_estimator_comparison.py').write_text(s)

if __name__=='__main__':run()
