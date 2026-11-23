"""Build the fitting-only E180 runner from the previously audited E179 controller."""
from pathlib import Path
import shutil
import sys

from .metric_readiness import ROOT as PREVIOUS, fingerprint, read, save

HERE = Path(__file__).parent
ROOT = Path("outputs/ease_readiness_recovery_v1")
CONDITIONS = ("checks_only", "bounded_alignment")


def prepare():
    ROOT.mkdir(exist_ok=False)
    (ROOT / "source").mkdir()
    shutil.copy2(PREVIOUS / "cases.json", ROOT / "cases.json")
    for condition in CONDITIONS:
        for folder in ("trials", "states", "actions", "histories", "clearance", "planning"):
            (ROOT / condition / folder).mkdir(parents=True)
    runner = (HERE / "metric_readiness.py").read_text()
    start = runner.index("def prepare():")
    end = runner.index("    _,models,hashes=", runner.index("def run(condition):", start))
    runner = runner[:start] + '''def run(condition):
    assert condition in ('checks_only', 'bounded_alignment')
    root = ROOT / condition
    method = 'e120_ensemble'
    protect_probe = gripper = True
    analytical_metric = False
''' + runner[end:]
    runner = runner.replace(str(PREVIOUS).replace("\\","/"), ROOT.as_posix()).replace("E179", "E180")
    runner = runner.replace("if self.route is not None or universal:", "if self.route is not None:")
    pos = runner.index("                eligible=bool")
    runner = runner[:pos] + '''                aligning = False
                if self.route is None:
                    if self.phase == 'close':
                        if self.last_phase != 'close':
                            self.align_steps = 0
                            self.close_anchor = before['q'].copy()
                        if not self.close_ready:
                            goal = np.r_[self.xy, .24]
                            gap = before['position'] - goal
                            drift = np.linalg.norm(self.data.xpos[self.obj, :2] - goal[:2])
                            axis = np.linalg.norm(before['rotation'][:, 2] - [0, 0, 1])
                            first = self.last_phase != 'close'
                            retry = condition == 'bounded_alignment' and self.align_steps <= 12 and not self.reject_locked
                            if first or retry:
                                self.close_ready = bool(np.linalg.norm(gap[:2]) <= .008 and abs(gap[2]) <= .012 and axis < .08 and drift <= .008)
                            if not self.close_ready:
                                aperture = 0.
                                aligning = retry and self.align_steps < 12 and drift <= .008
                                if aligning:
                                    # E120 already targets the pick pose during close. Leave its
                                    # bounded nominal command and original clearance projection intact.
                                    self.align_steps += 1
                                else:
                                    if not self.reject_locked:
                                        self.close_anchor = before['q'].copy()
                                    self.reject_locked = True
                                    optimization_target = np.clip(self.close_anchor - before['q'] + c.bias, -.04, .04)
                                    handoff_control = True
                    if self.phase == 'lift' and self.last_phase == 'close' and (not self.close_ready or self.close_good < 5):
                        self.aborted = True
                        self.abort_anchor = before['q'].copy()
                    if self.aborted:
                        optimization_target = np.clip(self.abort_anchor - before['q'] + c.bias, -.04, .04)
                        aperture = 0.
                        handoff_control = True
''' + runner[pos:]
    runner = runner.replace("env.object_history=[]", "env.reject_locked=False;env.align_steps=0;env.object_history=[]")
    runner = runner.replace("logs.append(dict(estimator_jacobian=", "logs.append(dict(aligning=aligning,estimator_jacobian=")
    (HERE / "readiness_recovery.py").write_text(runner[:runner.index("if __name__==")] + "if __name__ == '__main__':\n    import sys\n    run(sys.argv[1])\n")
    video = (HERE / "metric_readiness_videos.py").read_text().replace("metric_readiness import ROOT", "readiness_recovery import ROOT").replace("E179", "E180").replace("fitting projection metric and grasp-readiness comparison", "fitting grasp-readiness and alignment recovery").replace("Frozen E120 nominal policy in all four conditions; projection metric and universal handoff ablations.", "Protected E120 baseline, readiness checks and bounded alignment recovery.").replace('<img width="1000" src="../comparison.png">', '')
    (HERE / "readiness_recovery_videos.py").write_text(video)
    for p in (__file__, HERE / "readiness_recovery.py", HERE / "readiness_recovery_videos.py"):
        shutil.copy2(p, ROOT / "source" / Path(p).name)
    save(ROOT / "protocol.json", dict(baseline=str(PREVIOUS / "baseline"), cases=48, conditions=CONDITIONS,
         recovery="Maximum 12 open-gripper E120 pick-pose commands charged to the 40-step close budget. Recheck original readiness each step, latch on acceptance. Lock out on excessive object drift or timeout. Hold/open on rejection, retain original routed behavior. Original five-pad-step lift check and privileged protections.",
         checkpoints=read(PREVIOUS / "protocol.json")["checkpoints"]))
    save(ROOT / "frozen.json", {str(p):fingerprint(p) for p in [ROOT / "protocol.json", ROOT / "cases.json", *(ROOT / "source").iterdir()]})


if __name__ == "__main__":
    prepare()
