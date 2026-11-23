"""E153: learn relative weights while preserving the strongest uniform kernel."""
import shutil
from pathlib import Path
from unittest.mock import patch
import torch
from . import calibration_prior_learning as training
from .anchored_history_weights import AnchoredWeights
from .learned_history_weights import token_features
from .abstaining_task_value_gate import read,save,fingerprint
ROOT=Path('outputs/ease_fixed_uniform_learning_v1')

class FixedUniformWeights(AnchoredWeights):
    def __init__(self):super().__init__(.98,1e-6)
    def forward(self,q,x,mask,unused_prior,f=None):return super().forward(q,x,mask,f)

def features(q,x,mask,unused_prior):return token_features(q,x,mask)

def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('models','source'):(ROOT/folder).mkdir()
    for name in ('dataset.npz','cases.json'):shutil.copy2(training.ROOT/name,ROOT/name)
    for source in (__file__,training.__file__,Path(__file__).with_name('anchored_history_weights.py'),Path(__file__).with_name('learned_history_weights.py')):shutil.copy2(source,ROOT/'source'/Path(source).name)
    save(ROOT/'protocol.json',dict(experiment='E153',motivation='Uniform32 is the strongest nonprivileged fixed-controller reference so far, despite worse prediction error. Earlier weighted models changed kernel/ridge or introduced a calibration prior. Isolate learned relative weighting while retaining exactly the uniform baseline recency.98 and zero-centered ridge1e-6.',data='Exactly E1466706 clean every-fourth measured transitions from96 fitting placements, four original ancestry-family folds, same placement-balanced sampling and training-only normalization. Adaptive development data, not independent final tests. No added directional-probe supervision.',model='Original8-feature32-wide2-layer history Transformer; positive weights with bounded logits and normalized recency mass. Fixed.98recency and1e-6 zero-centered ridge in float64. No calibration-prior term or feature. The dataset prior field is ignored by this model and its reference. Initial output exactly uniform32. No body ID, phase, object/contact or future query enters model features.',training='Same E146 three fixed seeds,1500updates,AdamW3e-4,wd1e-4,batch128,clip1,normalized measured-next-motionMSE+.001 normalizedlogweight-square. Reuse the training implementation through explicit adapters; no best seed/checkpoint selection. Average all three Jacobians.',screen='At least5% lower ensemble position MSE than uniform32, neither joint count worse, scaled-rotation MSE<=1.05x. Passing allows full fitting control with the already fixed E151 posture rule and identical uniform/E120 references. It does not open fresh tests automatically. E120 unchanged.',inputs={str(training.ROOT/p):fingerprint(training.ROOT/p) for p in ('dataset.npz','verification.json')}))

def train():
    original_solve=training.solve
    def zero_solve(q,x,mask,unused_prior,logw):return original_solve(q,x,mask,torch.zeros_like(unused_prior),logw)
    with patch.object(training,'ROOT',ROOT),patch.object(training,'PriorWeights',FixedUniformWeights),patch.object(training,'features',features),patch.object(training,'solve',zero_solve):training.train()

if __name__=='__main__':
    import sys
    {'prepare':prepare,'train':train}[sys.argv[1]]()
