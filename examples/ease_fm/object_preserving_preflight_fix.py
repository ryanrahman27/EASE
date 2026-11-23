"""Archive/fix phase bookkeeping caught during replay, before intervention outcomes."""
from pathlib import Path
import shutil
from .abstaining_task_value_gate import read,save,fingerprint
from .build_object_preserving import run as build
ROOT=Path('outputs/ease_object_preserving_v1')

def run():
    assert not any((ROOT/c/'records.json').exists() for c in ('replay','object_route','guarded_handoff'))
    archive=ROOT/'phase_bookkeeping_preflight';archive.mkdir()
    for p in (ROOT/'protocol.json',ROOT/'frozen.json',ROOT/'source/object_preserving.py'):shutil.copy2(p,archive/p.name)
    build()
    for name in ('object_preserving.py','build_object_preserving.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    protocol=read(ROOT/'protocol.json');protocol['preflight_correction']='Replay stopped during initial calibration: generated phase bookkeeping was mistakenly inserted into the contacts callback. Restricted it to completion of controller steps. No intervention episode or completed replay existed. Original source/protocol/hash manifest archived; thresholds, plans and scoring unchanged.';save(ROOT/'protocol.json',protocol)
    old=read(ROOT/'frozen.json');save(ROOT/'frozen.json',{p:fingerprint(p) for p in old})

if __name__=='__main__':run()
