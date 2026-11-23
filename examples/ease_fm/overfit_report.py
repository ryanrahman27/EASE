"""Render descriptive tiny-subset results; no uncertainty or transfer claims."""
import json
from pathlib import Path
import sys
import numpy as np


def main():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.path.insert(0,str(Path('.recovery-deps').resolve()))
        import matplotlib.pyplot as plt
    rows=[]
    for label,folder in [('Original demonstrations','ease_overfit_v1'),('With shared recovery','ease_overfit_retrained_v1')]:
        for kind in ['flow','direct_chunk','direct_mlp']:
            root=Path('outputs')/folder/kind
            d=json.loads((root/'evaluation_5000.json').read_text())
            rows.append({'data':label,'model':kind,'successes':sum(r['success'] for r in d['records']),
                'trials':len(d['records']),'teacher_rmse_rad':d['teacher']['rmse_rad'],
                'parameters':json.loads((root/'training.json').read_text())['parameters']})
    out=Path('outputs/ease_overfit_retrained_v1')
    (out/'comparison.json').write_text(json.dumps(rows,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    x=np.arange(3); width=.34
    for index,label in enumerate(['Original demonstrations','With shared recovery']):
        selected=[r for r in rows if r['data']==label]
        pos=x+(index-.5)*width
        bars=axes[0].bar(pos,[100*r['successes']/r['trials'] for r in selected],width,label=label,color=['#72869c','#167d8d'][index])
        for bar,r in zip(bars,selected):
            axes[0].text(bar.get_x()+width/2,bar.get_height()+2,f"{r['successes']}/{r['trials']}",ha='center',fontsize=9)
        axes[1].bar(pos,[1000*r['teacher_rmse_rad'] for r in selected],width,color=['#72869c','#167d8d'][index])
    for ax in axes:
        ax.set_xticks(x,['Flow','Direct chunk','Direct MLP']); ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylim(0,115); axes[0].set_ylabel('Training rollout success (%)')
    axes[1].set_ylabel('Original-state action RMSE (mrad)')
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,-.16),frameon=False,fontsize=9)
    fig.suptitle('Recovery-state coverage improves tiny-subset control',fontsize=13)
    fig.text(.5,.02,'One six-joint arm, four goals, seven training episodes; seed 0, 5,000 updates.\nFlow: three sampling repeats per episode. Descriptive results; no held-out evaluation.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.17,1,.94])
    fig.savefig(out/'comparison.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
