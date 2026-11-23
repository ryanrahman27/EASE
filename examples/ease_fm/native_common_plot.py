"""Static phase-retention figure for the E63 technical report."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(output):
    root=Path(output);rows=json.loads((root/'records.json').read_text())
    methods=list(dict.fromkeys(r['method'] for r in rows));phases=[p['phase'] for p in rows[0]['phases']]
    fig,axes=plt.subplots(1,2,figsize=(14,7),layout='constrained')
    for ax,robot in zip(axes,['piper','panda']):
        counts=np.array([[sum(p['passes'] for r in rows if r['method']==method and r['robot']==robot for p in r['phases'] if p['phase']==phase) for phase in phases] for method in methods])
        ax.imshow(counts,vmin=0,vmax=6,cmap='YlGnBu',aspect='auto')
        ax.set_xticks(range(9),phases,rotation=45,ha='right');ax.set_yticks(range(10),methods)
        ax.set_title(f'{robot.capitalize()}: phase holds passed out of 6 cases')
        for i in range(10):
            for j in range(9):ax.text(j,i,str(counts[i,j]),ha='center',va='center',color='white' if counts[i,j]>=4 else 'black')
    fig.suptitle('E63: complete common pose schedule — geometry contacts disabled\nFinal eight samples within 6 mm and 0.08 tool-Z error; frozen models; no early termination',fontsize=13)
    fig.savefig(root/'phase_holds.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(**vars(p.parse_args()))
