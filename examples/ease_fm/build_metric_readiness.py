"""E179: prespecified fitting-only 2x2 projection metric / universal handoff test."""
from pathlib import Path

def run():
    base=Path(__file__).parent
    s=(base/'frozen_estimator_comparison.py').read_text()
    s=s.replace('E176','E179').replace("ROOT=Path('outputs/ease_frozen_estimator_comparison_v1')", "ROOT=Path('outputs/ease_metric_readiness_v1')")
    s=s.replace("CONDITIONS=('e120_ensemble','analytical')", "CONDITIONS=('baseline','analytical_metric','universal_readiness','combined')")
    a=s.index('def prepare():');b=s.index('\ndef run(condition):',a)
    s=s[:a]+'''def prepare():
    ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir();shutil.copy2(BASE/'cases.json',ROOT/'cases.json')
    _,_,hashes=old.broad.setup()
    save(ROOT/'protocol.json',dict(experiment='E179',scope='48 existing six-joint fitting cases x4 conditions; five exact preflight replays. No unseen execution or training.',conditions=list(CONDITIONS),nominal='Frozen E120 reference/correction ensembles, bounded residual, calibrated bias and history updates in every condition.',metric='baseline/universal_readiness use original reference J.T J + .015^2 I. analytical_metric/combined use solve(gram,cross).T from identical current E120 histories only for projection metric, at every eligible phase including probes. No analytical nominal action.',readiness='universal_readiness/combined apply existing routed handoff to all cases. At close entry use existing pick_goal or original xy at z=.24; xy error<=8mm,z error<=12mm,axis error<.08,object xy drift<=8mm. Hold close-entry joint posture, keep open if not ready. At lift entry abort/hold/open unless ready and at least five consecutive final two-pad close steps. No retries or extra interactions. Routed behavior unchanged.',budget='Original caps: calibration32,approach80,descend100 or300 only if route accepted,probe32,close40,lift100,transport100,lower100,release40,retreat80,settle40. Original early exit/scoring. Release265mm, probe/gripper/link protection, planning and E120 unchanged.',analysis='Paired success/collision losses against 48/48 fitting baseline, actual interactions, readiness/abort and projection counts. Full baseline exact replay required. No selection on unseen data. Thresholds inherited without fitting retuning.',videos='Baseline and candidates for every discordant case plus fixed cases0,26,43,62,67; all available success/failure categories. Actual saved state replay.',checkpoints=hashes,privileged=True))
    for name in (*CONDITIONS,'replay'):
        root=ROOT/name;root.mkdir()
        for folder in ('trials','states','actions','histories','clearance','planning'):(root/folder).mkdir()
    names=('metric_readiness.py','build_metric_readiness.py','frozen_estimator_comparison.py','early_object_plan.py','object_preserving_plan.py','object_preserving_planner.py','history_estimator_control.py','bounded_history_correction.py','training_estimator_core.py','local_history.py','pick_place_control.py','retention_fresh.py','arm_pick_place.py','local_probe.py','link_clearance_diagnostic.py')
    for name in names:shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py')]})
    save(ROOT/'status.json',dict(status='prepared',held_evaluated=False))
''' +s[b:]
    s=s.replace("method='e120_ensemble' if condition=='replay' else condition", "method='e120_ensemble';analytical_metric=condition in ('analytical_metric','combined');universal=condition in ('universal_readiness','combined')")
    s=s.replace("if self.route is not None:\n                    if self.phase=='close'", "if self.route is not None or universal:\n                    if self.phase=='close'")
    s=s.replace("self.close_anchor=before['q'].copy();gap=before['position']-self.pick_goal;drift=np.linalg.norm(self.data.xpos[self.obj,:2]-self.pick_goal[:2]);", "self.close_anchor=before['q'].copy();grasp_goal=self.pick_goal if self.pick_goal is not None else np.r_[self.xy,.24];gap=before['position']-grasp_goal;drift=np.linalg.norm(self.data.xpos[self.obj,:2]-grasp_goal[:2]);")
    s=s.replace("j,_=c.predict();j=np.asarray(j,dtype=np.float64);metric=", "j,_=c.predict();j=np.linalg.solve(c.gram,c.cross).T if analytical_metric else j;j=np.asarray(j,dtype=np.float64);metric=")
    s=s.replace("if method=='e120_ensemble':", "if condition in ('baseline','replay'):")
    s=s.replace("exact_replays=len(rows) if method=='e120_ensemble' else 0", "exact_replays=len(rows) if condition in ('baseline','replay') else 0")
    (base/'metric_readiness.py').write_text(s)
    v=(base/'frozen_estimator_videos.py').read_text().replace('frozen_estimator_comparison import ROOT','metric_readiness import ROOT').replace('E176','E179').replace('frozen-controller estimator comparison','fitting projection metric and grasp-readiness comparison').replace('E120 versus analytical estimation with the same privileged E175 controller and budget caps. Fitting cases only. Historical E174 failure is labeled separately.','Frozen E120 nominal policy in all four conditions; projection metric and universal handoff ablations. Fitting cases only, same budget caps.').replace("'PRIVILEGED geometry constraint diagnostic' if method.startswith('privileged_') else 'BASELINE E120 + release | No geometry constraint'", "'Frozen E120 | privileged geometry protections | fitting only'")
    (base/'metric_readiness_videos.py').write_text(v)

if __name__=='__main__':run()
