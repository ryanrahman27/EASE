"""Adapt the audited E124 reporting/audit pipeline to the fixed E125 expansion."""
from pathlib import Path

def build():
    s=Path(__file__).with_name('task_value_gate_artifacts.py').read_text()
    s=s.replace('task_value_gate','broad_task_value_gate').replace('E124','E125')
    s=s.replace('assert len(rows)==240','assert len(rows)==312')
    s=s.replace("('reference','unbounded','candidate','analytical')","('reference','unbounded','previousgate','candidate','analytical')")
    s=s.replace("for baseline in ('reference','unbounded'):","for baseline in ('reference','unbounded','previousgate'):")
    s=s.replace("s=s.replace(needle,extra+needle).replace('Outcome", "s=s.replace('len(rows)==240','len(rows)==312');s=s.replace(needle,extra+needle).replace('Outcome")
    s=s.replace('Fitting-only paired task outcomes and learned gate','Broader fitting placements and learned task-value gate')
    s=s.replace('candidate = E125 learned task-value gate.','previousgate = E124 gate; candidate = E125 broadened-data gate.')
    start=s.index('        f"Expanded to all16');end=s.index("        'One shared MLP",start)
    s=s[:start]+'''        f"Expanded fitting placements16 to48 by adding32 randomized variants. All original24 placements, including8 held, remain unchanged. Across144 fitting checkpoint/placement pairs,144 on references and1296 off-pulse continuations are saved:480 reused E124 episodes and960 new episodes. All1296 intervention snapshots/prefixes checked. {training['rows']} clean paired labels enter fitting. Held rows in fitting:0.",'',
        'Placement seed125000: two variants per original fitting placement; object XY uniform in[0.275,0.425] x[-0.10,0.10] meters. Destination preserves the source displacement plus independent +/-20mm jitter. Same home and new calibration seeds. Coverage was widened after inspecting placement geometry and before any new outcomes were collected; the unused local-jitter catalog is archived in design_history/. Exact held placements and labels remain excluded, but negative-Y spatial support is now included in fitting. This changes the spatial extrapolation setting. No IK/success screening or held-outcome-driven threshold/model selection.', '',
'''+s[end:]
    s=s.replace('this is48 fitting checkpoint/placement trajectories','this is144 fitting checkpoint/placement trajectories')
    s=s.replace("('candidate','Learned task-value gate')","('previousgate','E124 gate'),('candidate','E125 broader data')")
    s=s.replace('(i-1)*.23','(i-1.5)*.20').replace('],.21,label=label','],.18,label=label').replace('set_ylim(0,110)','set_ylim(0,125)').replace('legend(ncol=3','legend(ncol=4')
    s=s.replace('Revision 112 ·','Revision TEMP ·').replace('Revision 111 ·','Revision 112 ·').replace('Revision TEMP ·','Revision 113 ·')
    s=s.replace('480 fitting collection episodes (410 new)','1440 fitting collection episodes (960 new)')
    s=s.replace('collection_episodes=480,new_collection_episodes=410','collection_episodes=1440,new_collection_episodes=960')
    start=s.index("        'Keep E120 as the baseline;");end=s.index("        '| Held case/seed",start)
    s=s[:start]+'''        f"E120 remains unchanged. Retained {s['gains_retained']}/6 original E120 gains and recovered {s['regressions_recovered']}/4 original regressions. Strict retention check (including E118, E120 and E124 successful episodes): {s['retention_gate_passed']}. See the full paired changes before interpreting aggregate improvement.", '',
        'Compare the fixed original fitting and held placements against both E120 and the smaller-data E124 gate. The architecture, training seed, update count and threshold are unchanged; normalization is refit using expanded fitting data only. Sigmoid scores remain uncalibrated. No unseen embodiment claims are made.', '',
'''+s[end:]
    s=s.replace('candidate = learned task-value gate.','previousgate = E124 gate; candidate = E125 broader-data gate.')
    Path(__file__).with_name('broad_task_value_gate_artifacts.py').write_text(s)

if __name__=='__main__':build()
