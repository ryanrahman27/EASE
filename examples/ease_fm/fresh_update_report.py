"""Aggregate E182 matched physical branches by source, phase and case."""
from pathlib import Path
import shutil
import numpy as np
from .fresh_update_diagnosis import ROOT,SOURCE,CASES,METHODS,read,save,fingerprint


def run():
    phases=[]
    counts=0
    divergences={}
    for case in CASES:
        trajectories=[]
        for m in METHODS:
            rows=read(ROOT/m/f"{case}.json")
            trajectories.append(rows)
            counts+=len(rows)
            for phase in dict.fromkeys(r["phase"] for r in rows):
                group=[r for r in rows if r["phase"]==phase]
                metrics={}
                for i,method in enumerate(METHODS):
                    branch=[r["branches"][i] for r in group]
                    metrics[method]=dict(position_rmse_mm=float(np.sqrt(np.mean([b["prediction_error_mm"]**2 for b in branch]))),
                                         mean_progress_mm=float(np.mean([b["progress_mm"] for b in branch])),
                                         collision_steps=sum(b["collision"] for b in branch))
                phases.append(dict(case=case,source=m,phase=phase,n=len(group),metrics=metrics))
        divergences[case]=next(i for i,(a,b) in enumerate(zip(*trajectories[:2])) if np.max(np.abs(np.array(a["branches"][0]["action"])-b["branches"][1]["action"]))>1e-10)
    save(ROOT/"per_phase.json",phases)
    save(ROOT/"first_online_frozen_command_difference.json",divergences)
    lines=["# E182 | How online history updates changed pick-and-place outcomes", "",
           f"Retrospective analysis of all six E181 placements with a success disagreement among the three estimators. On {counts:,} histories from 18 actual episodes, each estimator was given the same current state, goal and controller context. Online statistics were rebuilt from the source trajectory's measured transitions, including on frozen-policy trajectories. The frozen candidate used its calibration history throughout. We checked the original estimator prediction, protected action, next physical state and forbidden contacts at every step. The E180 recovery rule and all checkpoints were held fixed.", "",
           "## Where the outcomes changed","",
           "| Case | Online E120 | Frozen E120 | Analytical | Observed difference |","|---|---|---|---|---|",
           "| 6 | Success | Failure | Success | Close-entry tool error was 1.8 mm online and 343 mm frozen. Frozen E120 spent all 12 alignment commands and aborted. |",
           "| 38 | Success | Failure | Success | Online E120 and analytical accepted recovery routes and reached close entry within 7.5 mm. Frozen E120 reached close entry 360 mm away and exhausted alignment. |",
           "| 43 | Success | Failure | Success | All three passed readiness and sustained two-pad contact through closing. Only frozen E120 failed to register a lift. |",
           "| 25 | Failure | Success | Success | Online E120 displaced the cube during descent and subsequently failed the object-drift check. |",
           "| 9 | Failure | Failure | Success | Cube drift at close entry was 29.9 mm online and 54.0 mm frozen. Analytical preserved the initial cube position. |",
           "| 35 | Success | Success | Failure | Analytical reached close entry 34.3 mm from the tool goal and exhausted 12 alignment commands. Both E120 variants secured the grasp. |","",
           "## Case 25", "",
           f"The online and frozen E120 executed commands first differed at post-calibration step {divergences[25]} (zero based). The online approach reached the goal. Descent finished with 2.8 mm tool error and 15.6 mm horizontal cube drift. By close entry the drift was 16.9 mm, exceeding the 8 mm recovery limit. No route was accepted, alignment was locked out and the robot aborted without a forbidden contact. Both comparison trajectories kept the cube at its starting XY position through the probe and then scored successful placements.", "",
           "On the same 24 online descent histories, reference-Jacobian position-change RMSE was 2.75 mm online, 3.60 mm frozen and 0.84 mm analytical. The respective protected actions made 3.58, 5.80 and 4.33 mm of average one-step progress toward the tool goal. RMSE uses measured joint displacement and observed tool displacement. It measures local kinematic prediction accuracy. The branch progress measures the effect of applying the alternative protected joint command for 40 ms.", "",
           "At step 24, cube drift was already 5.44 mm. From this exact state, one online command moved the cube another 4.52 mm, the frozen command moved it 6.87 mm and analytical moved it 4.68 mm. The successful comparison episodes approached along different trajectories. Substituting their actions at this contact state also moved the cube. The local branches identify a path-dependent failure and do not establish which earlier multi-step intervention would prevent it.", "",
           "## Matched descent comparison on each online E120 trajectory", "",
           "| Case | Position-change RMSE in mm, online / frozen / analytical | Mean one-step tool progress in mm, same order |","|---|---:|---:|"]
    for p in phases:
        if p["source"]==METHODS[0] and p["phase"]=="descend":
            values=[" / ".join(f"{p['metrics'][m][key]:.2f}" for m in METHODS) for key in ("position_rmse_mm","mean_progress_mm")]
            lines.append(f"| {p['case']} | {values[0]} | {values[1]} |")
    lines.extend(["","These cases were selected from the completed cohort by outcome. The original prospective scores remain 34/48 online E120, 32/48 frozen E120 and 35/48 analytical. The six cases distinguish grasp acquisition, lift retention, and object displacement as separate failure mechanisms. Per-phase metrics cover every source trajectory. Counterfactual branches inherit its route, aperture and guard decisions for that step.", "",
                  "[Per-phase data](per_phase.json) · [Case 25 contact audit](case25_contact.json) · [First action differences](first_online_frozen_command_difference.json)"])
    report="\n".join(lines)+"\n"
    (ROOT/"report.md").write_text(report,encoding="utf8")
    manifest=read(ROOT/"videos/manifest.json")["videos"]
    assert len(manifest)==18 and all(v["all_saved_states_match_trace"] and fingerprint(ROOT/"videos"/v["file"])==v["sha256"] for v in manifest)
    for p,h in {**read(SOURCE/"frozen.json"),**read(SOURCE/"protocol.json")["checkpoints"]}.items():assert fingerprint(p)==h
    for p in Path(__file__).parent.glob("fresh_update*.py"):shutil.copy2(p,ROOT/"source"/p.name)
    save(ROOT/"status.json",dict(status="complete",episodes=18,histories=counts,verified_source_transitions=counts,branches=counts*3,videos=18,recovery_changed=False))
    save(ROOT/"artifact_hashes.json",{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob("*") if p.is_file() and p.name!="artifact_hashes.json"})
    review=Path("outputs/ease_iteration_review_v1")
    section=f"\n## E182 matched-history diagnosis\n\nE181 cases 6, 38, 43, 25, 9 and 35. Verified {counts:,} matched states and 18 success/failure videos. Case 25 displaced the cube before closing. Report in `{ROOT}/report.md`.\n"
    for p in (Path(__file__).with_name("TECHNICAL_REPORT.md"),review/"report.md"):
        if "## E182 matched-history" not in p.read_text(encoding="utf8"):
            with p.open("a",encoding="utf8") as f:f.write(section)
    p=review/"WORKING_STATE.md"
    old=p.read_text(encoding="utf8")
    if not old.startswith("# Latest checkpoint: E182"):p.write_text("# Latest checkpoint: E182 complete\n"+section+"\n---\n"+old,encoding="utf8")
    studies=read(review/"studies.json")
    if not any(s["experiment"]=="E182" for s in studies):
        studies.append(dict(experiment="E182",folder=str(ROOT),report_exists=True,videos=18))
        save(review/"studies.json",studies)
    print(counts,"verified matched histories")


if __name__=="__main__":run()
