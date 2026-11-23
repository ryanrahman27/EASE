"""E177 changes data plumbing only; retain the E176 controller verbatim."""
from pathlib import Path

def run():
    base=Path(__file__).parent
    old=(base/'history_estimator_control.py').read_text();start=old.index('def evaluate(');end=old.index('\ndef run(worker):',start);fn=old[start:end]
    line="    source=DATA/'collection' if case<56 else COLLECT;baseline=read(source/'trials'/f'{case}_on_0_frozen_bias.json');assert r['calibration']==baseline['calibration'] and r['initial_estimator']==baseline['initial_estimator']\n"
    assert line in fn;fn=fn.replace(line,'')
    header="\"\"\"Unchanged E144 estimator logic without fitting-case archive assertions.\"\"\"\nfrom pathlib import Path\nfrom unittest.mock import patch\nimport numpy as np\nimport torch\nfrom .history_estimator_control import broad,history,token_features,PHASES,ROT_SCALE,save\nROOT=Path('outputs/ease_unseen_estimator_v1')\n\n"
    (base/'unseen_estimator_core.py').write_text(header+fn)
    s=(base/'frozen_estimator_comparison.py').read_text().replace('E176','E177')
    s=s.replace('from .early_object_plan import early_gate,observation','from .early_object_plan import early_gate,observation\nfrom . import unseen_estimator_core as evaluator\nfrom .mixed_sim import AXES')
    s=s.replace("REFERENCE=Path('outputs/ease_early_object_recovery_v1/early_guard')","REFERENCE=Path('outputs/ease_frozen_estimator_comparison_v1')").replace("ROOT=Path('outputs/ease_frozen_estimator_comparison_v1')","ROOT=Path('outputs/ease_unseen_estimator_v1')")
    s=s.replace('REPLAYS=(0,26,43,62,67)','REPLAYS=(0,26,43,64,67)')
    start=s.index('def prepare():');end=s.index('\ndef run(condition):',start);s=s[:start]+s[end:]
    start=s.index("    assert condition in");end=s.index('    for case,t in enumerate(tasks):',start)
    s=s[:start]+'''    replay=condition.startswith('replay_');method=condition.removeprefix('replay_');assert method in CONDITIONS;root=ROOT/condition;protect_probe=True;gripper=True
    if not replay:
        for m in CONDITIONS:assert read(ROOT/f'replay_{m}/verification.json')['exact_replays']==5
    _,models,hashes=old.broad.setup();AXES.update(read(ROOT/'axis_catalog.json'));core=old.broad.previous.core;core.ROOT=root;old.broad.previous.ROOT=root;original_targets=core.targets;original_rollout=core.probe_rollout;tasks=read(REFERENCE/'cases.json') if replay else read(ROOT/'cases.json');path=root/'records.json';rows=read(path) if path.exists() else [];done={r['case'] for r in rows}
''' +s[end:]
    s=s.replace("if t['validation'] or len(t['home'])!=6 or case in done or (condition=='replay' and case not in REPLAYS):continue","if case in done or (replay and case not in REPLAYS):continue")
    s=s.replace("with patch.object(old,'ROOT',root),", "with patch.object(evaluator,'ROOT',root),patch.object(old,'ROOT',root),")
    s=s.replace('r=old.evaluate(case,t,method,models,None,choices[folds[str(case)]])','r=evaluator.evaluate(case,t,method,models,None,{})')
    s=s.replace("reference_stem=f'{case}_e120_ensemble_frozen_bias'", "reference_stem=stem")
    old="base=read(BASE/'trials'/f'{reference_stem}.json');assert trial['calibration']==base['calibration'] and trial['initial_estimator']==base['initial_estimator'];"
    assert old in s;s=s.replace(old,'')
    old="        reference=read(REFERENCE/'trials'/f'{reference_stem}.json');states=np.load(root/'states'/f'{stem}.npz')['states']\n        if method=='e120_ensemble':\n            for key in ('trace','phases','success','collision','final_estimator'):assert trial[key]==reference[key],key\n            np.testing.assert_array_equal(states,np.load(REFERENCE/'states'/f'{reference_stem}.npz')['states'])"
    new="        states=np.load(root/'states'/f'{stem}.npz')['states']\n        if replay:\n            reference=read(REFERENCE/method/'trials'/f'{reference_stem}.json')\n            for key in ('trace','phases','success','collision','final_estimator','calibration','initial_estimator'):assert trial[key]==reference[key],key\n            np.testing.assert_array_equal(states,np.load(REFERENCE/method/'states'/f'{reference_stem}.npz')['states'])"
    assert old in s;s=s.replace(old,new)
    s=s.replace("(5 if condition=='replay' else 48)","(5 if replay else len(tasks))").replace("exact_replays=len(rows) if method=='e120_ensemble' else 0,calibration_identical=True", "exact_replays=len(rows) if replay else 0,paired_calibration_audit_pending=not replay")
    s=s.replace('no_held_execution=True','prospective_unseen_execution=not replay')
    s=s[:s.index("if __name__=='__main__':")]+"if __name__=='__main__':\n    import sys\n    run(sys.argv[1])\n"
    (base/'unseen_estimator.py').write_text(s)

if __name__=='__main__':run()
