"""E158 uses the frozen E156 lift rule and one predeclared weight attenuation."""
from pathlib import Path

def run():
    source=Path('outputs/ease_lift_posture_control_v1/source/lift_posture_control.py').read_text();source=source.replace('E156','E158').replace('ease_lift_posture_control_v1','ease_conservative_lift_control_v1');source=source.replace('import numpy as np','import numpy as np\nimport torch\nfrom .conservative_history_weights import ROOT as CONSERVATIVE,ConservativeWeights\nfrom .learned_history_weights import SEEDS')
    source=source.replace('def prepare():\n    ROOT.mkdir','def prepare():\n    assert read(CONSERVATIVE/\'summary.json\')[\'qualifies_for_diagnostic\']\n    ROOT.mkdir')
    needle="shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)";source=source.replace(needle,needle+";shutil.copy2(Path(__file__).with_name('conservative_history_weights.py'),ROOT/'source/conservative_history_weights.py')")
    before='E154 standard learned fixed-uniform reaches77/96 versus uniform74/E12072, but legacy case13 still collides at lift. E155 all-phase prior-map projection rescues13 but regresses7-joint controls6/10. Test the identical projection only during lift, where the motivating link/table collision occurs. All pre-lift actions/states must match the original standard controller exactly.'
    after='E154 standard learned weights improve77/96 vs uniform74/E12072 but regress cases. E156 lift-only projection rescues case13 with uniform, not full learned weights. E157 attenuates learned log weights by fixed.25 and renormalizes mass, retaining lower prediction error. Test this one conservative model under the unchanged lift-only rule; no temperature sweep or new weight training. Compare every pre-lift state with an actual matched conservative standard rollout.'
    assert before in source;source=source.replace(before,after);source=source.replace('E153 learned fixed-uniform ensemble','E157 conservative learned fixed-uniform ensemble');source=source.replace("inputs={str(MODEL/p):fingerprint(MODEL/p) for p in ('summary.json','verification.json','frozen.json')}","inputs={str(CONSERVATIVE/p):fingerprint(CONSERVATIVE/p) for p in ('summary.json','verification.json','frozen.json')}")
    needle="    with patch.object(old,'ROOT',ROOT):neural=old.load_neural()";assert needle in source;source=source.replace(needle,'    neural=load_neural()');source=source.replace("base_root=Path('outputs/ease_fixed_uniform_control_v1/standard') if method=='learned_ensemble'", "base_root=ROOT/'standard' if method=='learned_ensemble'")
    functions="""
def load_neural():
    neural={}
    for f in range(4):
        neural[f]=[]
        for seed in SEEDS:
            m=ConservativeWeights().cuda().eval();m.load_state_dict(torch.load(ROOT/'models'/f'fold_{f}'/f'{seed}.pt',weights_only=True));neural[f].append(m)
    return neural

def baseline(scope):
    assert scope in ('diagnostic','full')
    if scope=='full':assert read(ROOT/'diagnostic_summary.json')['ready_for_full']
    root=ROOT/'standard';root.mkdir(exist_ok=True)
    for folder in ('trials','states','actions','histories'):(root/folder).mkdir(exist_ok=True)
    path=root/'records.json';rows=read(path) if path.exists() else [];done={r['case'] for r in rows};tasks=read(ROOT/'cases.json');folds=read(ROOT/'folds.json');_,models,hashes=old.broad.setup();old.broad.previous.ROOT=root;old.broad.previous.core.ROOT=root;neural=load_neural();selected=DIAGNOSTIC_CASES if scope=='diagnostic' else [i for i,t in enumerate(tasks) if not t['validation']]
    for case in selected:
        if case in done:continue
        with patch.object(old,'ROOT',root):r=old.evaluate(case,tasks[case],'learned_ensemble',models,neural[folds[str(case)]],dict(forgetting=.98,ridge=1e-6))
        rows.append(r);save(path,rows);print('E158 standard',case,r['success'],flush=True)
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h

"""
    source=source.replace("if __name__=='__main__':",functions+"if __name__=='__main__':");source=source.replace("    if sys.argv[1]=='prepare':prepare()\n    else:run(sys.argv[1])","    if sys.argv[1]=='prepare':prepare()\n    elif sys.argv[1]=='baseline':baseline(sys.argv[2])\n    else:run(sys.argv[1])")
    target=Path(__file__).with_name('conservative_lift_control.py');assert not target.exists();target.write_text(source);print('Created fixed conservative E158 diagnostic',flush=True)

if __name__=='__main__':run()
