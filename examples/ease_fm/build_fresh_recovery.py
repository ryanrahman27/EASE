"""Combine the verified E180 controller with the prospective cohort adapter."""
from pathlib import Path

HERE = Path(__file__).parent


def source(name):
    return (HERE / (name + ".py")).read_text()


def build():
    previous = source("unseen_estimator")
    frozen = Path("outputs/ease_readiness_recovery_v1/source/readiness_recovery_logging_fix.py").read_text()
    assert frozen == source("readiness_recovery")
    start, end = "        class ProtectedArm", "        with patch.object("
    previous = previous[:previous.index(start)] + frozen[frozen.index(start):frozen.index(end)] + previous[previous.index(end):]
    previous = previous.replace("unseen_estimator_core as evaluator", "fresh_recovery_core as evaluator")
    previous = previous.replace("ease_unseen_estimator_v1","ease_fresh_recovery_v1").replace("'e120_ensemble','analytical'", "'e120_ensemble','analytical','e120_frozen'")
    previous = previous.replace("    if not replay:\n        for m in CONDITIONS:assert read(ROOT/f'replay_{m}/verification.json')['exact_replays']==5", """    assert not replay
    condition = 'bounded_alignment'
    analytical_metric = False""")
    previous = previous.replace("r.update(condition=condition", "r.update(condition=method").replace("E177 PRIVILEGED',condition", "E181',method")
    previous = previous.replace("root/'states'/f'{stem}.npz'", "root/'states'/f'{case}_{method}_{trial[\"mode\"]}.npz'")
    (HERE/"fresh_recovery.py").write_text(previous)
    core = source("unseen_estimator_core").replace("method=='e120_ensemble'", "method in ('e120_ensemble','e120_frozen')")
    before = "        if method in ('e120_ensemble','e120_frozen'):r,_="
    core = core.replace(before, """        if method == 'e120_frozen':
            evaluate_response = core.evaluate
            def calibrated_response(case, task, name, model, mode, **kwargs):
                return evaluate_response(case, task, name, model, 'frozen_all', **kwargs)
            with patch.object(core, 'evaluate', calibrated_response):
                r, _ = previous.evaluate(case,t,method,models['reference_0'],models['correction_0'],'bounded',save=True)
        elif method=='e120_ensemble':r,_=""")
    (HERE/"fresh_recovery_core.py").write_text(core)
    prep = source("unseen_estimator_prepare").replace("from .unseen_estimator import", "from .fresh_recovery import").replace("1770","1810").replace("1772000","1812000").replace("e177","e181").replace("E177","E181")
    a,b=prep.index("    save(ROOT/'protocol.json'"),prep.index("    for name in set(names)")
    prep=prep[:a]+'''    protocol.update(scope="Fresh procedural six-joint bodies. Three paired conditions per accepted placement.",controller="E180 bounded recovery, including 12-step limit within closing, exact E120 clearance and release protections.", controls="Online E120, online analytical, E120 with both learned Jacobians computed exclusively from the initial 32 calibration transitions. All execute the full probe, observe current state and use identical controller logic and limits.",replay="Frozen E180 controller source copied verbatim. Audit paired calibrations, updates, initial first actions and probe budgets.",videos="Paired videos on the first case per success pattern, all methods, plus first collision episode per method. All executed state traces retained.",analysis="Paired wins/losses, interactions, collisions, per-body success and paired differences with percentile intervals from resampling complete bodies (seed 181, 20000 draws).")
    save(ROOT/'protocol.json',protocol)
    names=('fresh_recovery.py','fresh_recovery_core.py','fresh_recovery_prepare.py','build_fresh_recovery.py','readiness_recovery.py','training_estimator_core.py','bounded_history_correction.py','early_object_plan.py','object_preserving_plan.py','local_probe.py','link_clearance_diagnostic.py','arm_pick_place.py','local_history.py')
''' +prep[b:]
    (HERE/"fresh_recovery_prepare.py").write_text(prep)
    video = source("readiness_recovery_videos").replace("from .readiness_recovery import ROOT","from .fresh_recovery import ROOT").replace("E180","E181").replace("fitting grasp-readiness and alignment recovery","fresh six-joint bodies").replace("Protected E120 baseline, readiness checks and bounded alignment recovery. Fitting cases only, same budget caps.","Fresh six-joint bodies. Online E120, analytical and calibration-only E120 with bounded recovery and matched budgets.").replace("Frozen E120 | privileged geometry protections | fitting only","Fresh body | bounded recovery | privileged geometry protections")
    (HERE/"fresh_recovery_videos.py").write_text(video)


if __name__ == "__main__":
    build()
