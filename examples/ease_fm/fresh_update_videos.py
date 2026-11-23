"""Prepare all 18 actual E181 trajectories for the E182 diagnostic gallery."""
from pathlib import Path
import shutil
from .fresh_update_diagnosis import ROOT, SOURCE, CASES, METHODS, read, save


def render():
    from . import fresh_recovery_videos
    for folder in ("source","trials","states"):
        (ROOT/folder).mkdir(parents=True,exist_ok=True)
    records=[]
    for case in CASES:
        for method in METHODS:
            trial=read(SOURCE/method/"trials"/f"{case}_{method}_frozen_bias.json")
            stem=f"{case}_{method}_{trial['mode']}"
            save(ROOT/"trials"/f"{stem}.json",trial)
            for ext in ("xml","npz"):
                shutil.copy2(SOURCE/method/"states"/f"{stem}.{ext}",ROOT/"states"/f"{stem}.{ext}")
            records.append(dict(case=case,method=method,mode=trial["mode"]))
    save(ROOT/"video_records.json",records)
    fresh_recovery_videos.ROOT=ROOT
    fresh_recovery_videos.run()
    index=ROOT/"videos/index.html"
    index.write_text(index.read_text().replace("E181 / PRIVILEGED fresh six-joint bodies","E182 matched-history diagnosis | recorded E181 episodes"))


if __name__=="__main__":
    render()
