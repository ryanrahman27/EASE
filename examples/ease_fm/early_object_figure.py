"""Publication-style count plot from audited E175 summaries."""
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .early_object import ROOT
from .abstaining_task_value_gate import read

def run():
    s=read(ROOT/'summary.json');values=[s['references']['E120_release'],s['references']['E174_guarded'],s['conditions']['early_guard']['outcomes']]
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,key,title in zip(axes,('success','collisions','collision_free_failures'),('Tasks completed','Forbidden-contact episodes','Collision-free failures')):
        ns=[v[key] for v in values];ax.bar(range(3),ns,color=['#84909d','#3b76ae','#29896c'],width=.65)
        ax.set_xticks(range(3),['E120 + release','E174','Earlier recovery']);ax.set_title(title,fontsize=11);ax.set_ylim(0,52);ax.set_yticks([0,12,24,36,48]);ax.set_ylabel('Episodes / 48');ax.spines[['top','right']].set_visible(False)
        for i,n in enumerate(ns):ax.text(i,n+.8,str(n),ha='center',fontsize=11)
    fig.suptitle('E175: earlier object-safe recovery | privileged fitting evaluation',fontsize=13);fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=180);plt.close(fig);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)

if __name__=='__main__':run()
