"""Compare actual E180 trajectories with the archived E179 baseline."""
from pathlib import Path
import shutil
import sys

import numpy as np

from .build_readiness_recovery import ROOT, PREVIOUS, CONDITIONS, read, save, fingerprint

METHOD = "e120_ensemble_frozen_bias"


def audit():
    baseline = PREVIOUS / "baseline"
    prior = {r["case"]: r for r in read(PREVIOUS / "details.json")["baseline"]}
    rows = []
    selected = {40, 42}
    for condition in CONDITIONS:
        run = ROOT / condition
        records = read(run / "records.json")
        assert read(run / "verification.json")["episodes"] == len(records) == 48
        assert {r["case"] for r in records} == set(prior)
        for record in records:
            case = record["case"]
            stem = f"{case}_{METHOD}"
            trial = read(run / "trials" / f"{stem}.json")
            original = read(baseline / "trials" / f"{stem}.json")
            logs = read(run / "clearance" / f"{stem}.json")
            states = np.load(run / "states" / f"{stem}.npz")["states"]
            reference = np.load(baseline / "states" / f"{stem}.npz")["states"]
            close = [l for l in logs if l["phase"] == "close"]
            i = next(i for i,l in enumerate(logs) if l["phase"] == "close")
            assert len(close) == 40 and len(states) + 32 == trial["steps"]
            assert trial["calibration"] == original["calibration"]
            np.testing.assert_array_equal(states[:i], reference[:i])
            expected = prior[case]["would_be_ready"]
            assert close[0]["close_ready"] == expected
            alignment = sum(l["aligning"] for l in close)
            assert alignment <= (12 if condition == "bounded_alignment" else 0)
            assert all(l["aperture"] == 0 for l in close if not l["close_ready"])
            ready_index = next((n for n,l in enumerate(close) if l["close_ready"]), None)
            identical = np.array_equal(states, reference) and trial["trace"] == original["trace"]
            if expected and prior[case]["final_two_pad_steps"] >= 5:
                assert identical, (condition, case)
            if not identical:
                selected.add(case)
            rows.append(dict(condition=condition, case=case, ready_entry=expected, alignment_steps=alignment,
                             ready_after=ready_index, pads=close[-1]["close_good"], exact=identical,
                             aborted=logs[-1]["aborted"], success=trial["success"], collision=trial["collision"],
                             steps=trial["steps"], final_object=trial["final_object"]))
    save(ROOT / "paired.json", rows)
    videos = []
    for folder in ("trials", "states"):
        (ROOT / folder).mkdir(exist_ok=True)
    for label, folder in [("baseline", baseline), *((c, ROOT/c) for c in CONDITIONS)]:
        for case in sorted(selected):
            old = f"{case}_{METHOD}"
            new = f"{case}_{label}_frozen_bias"
            trial = read(folder / "trials" / f"{old}.json")
            trial["method"] = label
            save(ROOT / "trials" / f"{new}.json", trial)
            for ext in ["xml","npz"]:
                shutil.copy2(folder / "states" / f"{old}.{ext}", ROOT / "states" / f"{new}.{ext}")
            videos.append(dict(case=case,method=label,mode="frozen_bias"))
    save(ROOT / "video_records.json", videos)


def finish():
    rows = read(ROOT / "paired.json")
    videos = read(ROOT / "videos/manifest.json")["videos"]
    assert len(videos) == len(read(ROOT / "video_records.json"))
    for v in videos:
        assert v["all_saved_states_match_trace"] and fingerprint(ROOT / "videos" / v["file"]) == v["sha256"]
    lines = ["# E180 fitting evaluation of grasp-readiness checks and bounded alignment", "",
             "Each candidate ran 48 complete pick-and-place episodes on the original six-joint fitting placements. The comparator uses the 48 audited E179 baseline trajectories. Every candidate reaches close entry on the exact baseline trajectory. The routed handoff, probe protection, release height, E120 weights and analytical comparison from earlier experiments retain their original definitions.", "",
             "| Condition | Successes | Collision episodes | Complete trajectories identical to E120 |",
             "|---|---:|---:|---:|", "| Archived E120 baseline | 48/48 | 0 | 48 |"]
    for condition in CONDITIONS:
        group = [r for r in rows if r["condition"] == condition]
        lines.append(f"| {condition} | {sum(r['success'] for r in group)}/48 | {sum(r['collision'] for r in group)} | {sum(r['exact'] for r in group)} |")
    lines.extend(["", "Case 40 passes the entry checks and exactly reproduces the original successful trajectory. Case 42 is the only baseline grasp that fails the entry checks. For accepted non-routed grasps, ordinary E120 closing continues. The recovery condition spends up to 12 of the original 40 close steps aligning to the pick pose with the fingers open. It checks position, orientation and object drift before each attempt to close.", ""])
    for r in rows:
        if r["case"] == 42:
            outcome = "succeeded" if r["success"] else "failed"
            lines.append(f"Case 42 {outcome} under {r['condition']}. Alignment commands used = {r['alignment_steps']}. First ready close step (zero based) = {r['ready_after']}. Final consecutive two-pad steps = {r['pads']}. Abort = {r['aborted']}. Collision = {r['collision']}.")
    lines.extend(["", "## Research status", "", "E177 established simulated pick-and-place transfer to 12 newly sampled six-joint bodies across four previously used axis families. Protected E120 and the analytical estimator each completed 42/48 placements. E120 uniquely solved three placements and the analytical estimator uniquely solved three. E180 measures fitting retention and provides one recovery opportunity. A claim that learned adaptation improves transfer still needs a prospective paired advantage across fresh bodies, with body-level uncertainty and a matched control for history adaptation.", "", "[Success and failure videos](videos/index.html) · [All paired outcomes](paired.json) · [Protocol and checkpoint hashes](protocol.json)"])
    report = "\n".join(lines) + "\n"
    (ROOT / "report.md").write_text(report)
    shutil.copy2(__file__, ROOT / "source" / Path(__file__).name)
    review = Path("outputs/ease_iteration_review_v1")
    section = "\n" + report.replace("# E180", "## E180", 1).split("[Success and failure videos]")[0] + f"\nArtifacts in `{ROOT}`.\n"
    for path in [Path(__file__).with_name("TECHNICAL_REPORT.md"), review / "report.md"]:
        with path.open("a", encoding="utf-8") as f:
            f.write(section)
    latest = review / "WORKING_STATE.md"
    latest.write_text("# Latest checkpoint: E180 complete\n" + section + "\n---\n" + latest.read_text(encoding="utf-8"), encoding="utf-8")
    studies = read(review / "studies.json")
    studies.append(dict(experiment="E180",folder=str(ROOT),report_exists=True,videos=len(videos)))
    save(review / "studies.json", studies)
    save(ROOT / "status.json", dict(status="complete",new_episodes=96,archived_baseline=48,videos=len(videos),held_evaluated=False))
    save(ROOT / "artifact_hashes.json", {str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob("*") if p.is_file() and p.name!="artifact_hashes.json"})
    print(report)


if __name__ == "__main__":
    finish() if sys.argv[1:] == ["finish"] else audit()
