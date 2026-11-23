"""Plot predeclared placement coverage without reading any outcomes."""
import numpy as np
from .expanded_structured_pairs import ROOT
from .abstaining_task_value_gate import read

def run():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tasks=read(ROOT/'cases.json');fresh=read(ROOT/'fresh_cases.json');bodies=sorted({t['arm']['seed'] for t in tasks});fig,axes=plt.subplots(2,2,figsize=(10,8),sharex=True,sharey=True)
    for ax,body in zip(axes.flat,bodies):
        old=[t for t in tasks[:56] if t['arm']['seed']==body and not t['validation']];new=[t for t in tasks[56:] if t['arm']['seed']==body];test=[t for t in fresh if t['arm']['seed']==body]
        for rr,label,marker,color in ((old,'Existing fitting','o','#555555'),(new,'Additional fitting','+','#2679b2'),(test,'Reserved fresh test','s','#d58432')):
            xy=np.asarray([t['xy'] for t in rr]);ax.scatter(xy[:,0],xy[:,1],label=label,marker=marker,c=color,s=45,alpha=.8)
        ax.set_title(f'Body {body} / {len(new[0]["home"])} joints');ax.set_xlabel('Object X (m)');ax.set_ylabel('Object Y (m)');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=8);fig.suptitle('E138 placement coverage — geometry only, no outcome filtering');fig.tight_layout();fig.savefig(ROOT/'coverage.png',dpi=150);plt.close(fig)

if __name__=='__main__':run()
