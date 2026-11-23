"""Paired, body-clustered E181 outcomes and verified state replay selection."""
import shutil
import sys
from pathlib import Path

import numpy as np

from .fresh_recovery import ROOT, CONDITIONS, read, save, fingerprint

LABELS = ("Online E120", "Online analytical", "Calibration-only E120")
BUDGETS = dict(approach=80, descend=100, probe=32, close=40, lift=100,
               transport=100, lower=100, release=40, retreat=80, settle=40)


def audit():
    tasks = read(ROOT / "cases.json")
    rows = []
    selected = set()
    patterns = set()
    collisions_seen = set()
    for method in CONDITIONS:
        assert read(ROOT / method / "verification.json")["episodes"] == len(tasks)
    for case, task in enumerate(tasks):
        paired = []
        for method in CONDITIONS:
            folder = ROOT / method
            stem = f"{case}_{method}_frozen_bias"
            trial = read(folder / "trials" / f"{stem}.json")
            log = read(folder / "clearance" / f"{stem}.json")
            state_stem = f"{case}_{method}_{trial['mode']}"
            states = np.load(folder / "states" / f"{state_stem}.npz")["states"]
            initial = trial["initial_estimator"]
            if not paired:
                calibration = trial["calibration"]
                initial_stats = initial
                first = log[0]
                xml = fingerprint(folder / "states" / f"{state_stem}.xml")
            assert xml == fingerprint(folder / "states" / f"{state_stem}.xml")
            assert initial == initial_stats and len(calibration) == 32 and trial["calibration"] == calibration
            assert len(states) == len(log) == len(trial["trace"]) == trial["steps"] - 32
            for key in ("full_qpos_before", "full_qvel_before"):
                np.testing.assert_array_equal(first[key],log[0][key])
            frozen = method == "e120_frozen"
            assert trial["response_updates"] == (0 if frozen else len(log))
            if frozen:
                for key in ("nominal_action","final_action","estimator_jacobian"):
                    np.testing.assert_array_equal(first[key],log[0][key])
                assert trial["final_estimator"] == initial
                for l in log:
                    assert l["estimator_gram"] == initial["gram"] and l["estimator_cross"] == initial["cross"]
                    assert l["estimator_jacobian"] == first["estimator_jacobian"]
                with np.load(folder/"histories"/f"{stem}.npz") as history:
                    for key in ("dq_history", "dx_history"):
                        assert np.all(history[key] == history[key][0])
            routed = any(l["route_active"] for l in log)
            caps = dict(BUDGETS, descend=300 if routed else 100)
            phase_steps = {p["phase"]:p["steps"] for p in trial["phases"]}
            assert phase_steps.keys() == caps.keys() and all(0 < phase_steps[k] <= cap for k,cap in caps.items())
            assert phase_steps["probe"] == 32 and phase_steps["close"] == 40
            alignment = sum(l["aligning"] for l in log)
            assert alignment <= 12 and all(l["phase"]=="close" and l["aperture"]==0 for l in log if l["aligning"])
            paired.append(dict(case=case,body=task["body_index"],family=task["arm"]["family"],method=method,
                               success=trial["success"],collision=trial["collision"],steps=trial["steps"],
                               alignment_steps=alignment,aborted=log[-1]["aborted"],routed=routed,updates=trial["response_updates"]))
            if trial["collision"] and method not in collisions_seen:
                selected.add(case)
                collisions_seen.add(method)
        pattern = tuple(r["success"] for r in paired)
        if pattern not in patterns:
            selected.add(case)
            patterns.add(pattern)
        rows.extend(paired)
    save(ROOT/"paired.json",rows)
    body_ids = sorted({r["body"] for r in rows})
    body_scores = np.array([[np.mean([r["success"] for r in rows if r["body"]==b and r["method"]==m])
                            for m in CONDITIONS] for b in body_ids])
    assert all(sum(r["body"] == b for r in rows)==4*len(CONDITIONS) for b in body_ids)
    draws = np.random.default_rng(181).integers(len(body_ids),size=(20000,len(body_ids)))
    summary = {}
    lines = ["# E181 prospective pick-and-place transfer", "",
             f"All {len(body_ids)} fresh six-joint bodies passed initial-posture screening. Four placements per body give {len(tasks)} paired cases in four previously used axis families. Body seeds and physical signatures were checked against archived case manifests before scoring. Each condition uses the E180 bounded recovery rule, privileged geometry and contact protections, and the same interaction caps.", "",
             "| Estimator | Successes | Episodes with collision | Median control steps including calibration | Episodes using alignment |",
             "|---|---:|---:|---:|---:|"]
    for m, label in zip(CONDITIONS,LABELS):
        group=[r for r in rows if r["method"]==m]
        counts=dict(success=sum(r["success"] for r in group),collision=sum(r["collision"] for r in group),
                    median_steps=float(np.median([r["steps"] for r in group])),total_steps=sum(r["steps"] for r in group),
                    alignment=sum(r["alignment_steps"]>0 for r in group))
        summary[m]=counts
        lines.append(f"| {label} | {counts['success']}/{len(tasks)} | {counts['collision']} | {counts['median_steps']:g} | {counts['alignment']} |")
    lines+=["","## Paired comparisons",""]
    for j in (1,2):
        diff=body_scores[:,0]-body_scores[:,j]
        low,high=100*np.percentile(diff[draws].mean(axis=1),[2.5,97.5])
        a=[r for r in rows if r["method"]==CONDITIONS[0]]
        b=[r for r in rows if r["method"]==CONDITIONS[j]]
        wins=[x["case"] for x,y in zip(a,b) if x["success"] and not y["success"]]
        losses=[x["case"] for x,y in zip(a,b) if y["success"] and not x["success"]]
        summary["versus_"+CONDITIONS[j]]=dict(wins=wins,losses=losses,difference_pp=100*diff.mean(),body_bootstrap_95_pp=[low,high])
        lines.append(f"Online E120 versus {LABELS[j]} had {len(wins)} unique successes and {len(losses)} unique losses. Paired difference {100*diff.mean():+.1f} percentage points (body-bootstrap 95% interval {low:+.1f} to {high:+.1f}).")
        lines.append("")
    lines+=["The frozen-history condition uses both E120 networks and current state feedback. Its initial 32 calibration transitions determine its motion estimate for the entire episode. Its 32 pre-close probe actions still count toward the budget. Audits verified identical calibration and initial state across all three methods, identical first actions for the two E120 conditions, zero post-calibration updates for the frozen condition, one update per step for both online conditions, and the original phase limits.", "",
            "Intervals resample 12 complete bodies with replacement, including all four placements of each sampled body. The inference applies to this simulated perturbation distribution and the four sampled axis families.", "",
            "| Body | Axis family | Online E120 | Analytical | Frozen E120 |","|---|---|---:|---:|---:|"]
    for b,scores in zip(body_ids,body_scores):
        family=next(r["family"] for r in rows if r["body"]==b)
        lines.append(f"| {b} | {family} | "+" | ".join(f"{int(score*4)}/4" for score in scores)+" |")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(10, 2.8), layout="constrained")
    counts = (body_scores.T * 4).astype(int)
    axis.imshow(counts, vmin=0, vmax=4, cmap="Blues", aspect="auto")
    for method_id, values in enumerate(counts):
        for body, count in enumerate(values):
            axis.text(body, method_id, str(count), ha="center", va="center",
                      color="white" if count >= 3 else "#162536")
    axis.set_xticks(range(len(body_ids)), body_ids)
    axis.set_yticks(range(3), LABELS)
    axis.set_xlabel("Fresh body (four paired placements per body)")
    axis.set_title("Pick-and-place successes per arm")
    figure.savefig(ROOT/"body_results.png",dpi=160)
    plt.close(figure)
    lines.extend(["", "![Paired task successes by body](body_results.png)"])
    lines+=["","[Video gallery](videos/index.html) · [All paired results](paired.json) · [Cohort and protocol](protocol.json)"]
    (ROOT/"report.md").write_text("\n".join(lines)+"\n")
    save(ROOT/"summary.json",summary)
    videos=[]
    for folder in ("states","trials"):
        (ROOT/folder).mkdir(exist_ok=True)
    for case in sorted(selected):
        for method in CONDITIONS:
            trial=read(ROOT/method/"trials"/f"{case}_{method}_frozen_bias.json")
            stem=f"{case}_{method}_{trial['mode']}"
            save(ROOT/"trials"/f"{stem}.json",trial)
            for ext in ("xml","npz"):
                shutil.copy2(ROOT/method/"states"/f"{stem}.{ext}",ROOT/"states"/f"{stem}.{ext}")
            videos.append(dict(case=case,method=method,mode=trial["mode"]))
    save(ROOT/"video_records.json",videos)
    print(summary,flush=True)


def finish():
    videos=read(ROOT/"videos/manifest.json")["videos"]
    assert len(videos)==len(read(ROOT/"video_records.json"))
    for v in videos:
        assert v["all_saved_states_match_trace"] and fingerprint(ROOT/"videos"/v["file"])==v["sha256"]
    for p,h in {**read(ROOT/"frozen.json"),**read(ROOT/"protocol.json")["checkpoints"]}.items():
        assert fingerprint(p)==h
    report=(ROOT/"report.md").read_text().split("| Body |")[0].strip()
    section="\n"+report.replace("# E181","## E181",1)+f"\n\nArtifacts in `{ROOT}`.\n"
    review=Path("outputs/ease_iteration_review_v1")
    for path in (Path(__file__).with_name("TECHNICAL_REPORT.md"),review/"report.md"):
        if "## E181 prospective" not in path.read_text(encoding="utf-8"):
            with path.open("a",encoding="utf-8") as f:f.write(section)
    p=review/"WORKING_STATE.md"
    old=p.read_text(encoding="utf-8")
    if not old.startswith("# Latest checkpoint: E181"):
        p.write_text("# Latest checkpoint: E181 complete\n"+section+"\n---\n"+old,encoding="utf-8")
    studies=read(review/"studies.json")
    if not any(r["experiment"]=="E181" for r in studies):
        studies.append(dict(experiment="E181",folder=str(ROOT),report_exists=True,videos=len(videos)))
        save(review/"studies.json",studies)
    shutil.copy2(__file__,ROOT/"source"/Path(__file__).name)
    save(ROOT/"status.json",dict(status="complete",episodes=len(read(ROOT/"paired.json")),videos=len(videos)))
    save(ROOT/"artifact_hashes.json",{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob("*") if p.is_file() and p.name!="artifact_hashes.json"})
    print("E181 complete",flush=True)


if __name__ == "__main__":
    finish() if sys.argv[1:]==["finish"] else audit()
