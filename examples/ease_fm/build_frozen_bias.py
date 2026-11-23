"""Preserve E114 and generate its single-factor E115 extension."""
from pathlib import Path
b=Path(__file__).parent
s=(b/'adaptation_ablation.py').read_text().split('\ndef run():')[0]
s=s.replace('ease_adaptation_ablation_v1','ease_frozen_bias_v1').replace("MODES=('online','frozen_response','frozen_all')","MODES=('online','frozen_response','frozen_all','frozen_bias')")
s=s.replace("if mode=='online':super().update(before,after,command);updates+=1", "if mode in ('online','frozen_bias'):\n                saved_bias=self.bias.copy();super().update(before,after,command);updates+=1\n                if mode=='frozen_bias':\n                    self.bias=saved_bias\n                    np.testing.assert_array_equal(self.bias,initial['bias'])")
s=s.replace("if mode=='online':c.local=True", "if mode in ('online','frozen_bias'):c.local=True")
s=s.replace("if mode!='online':", "if mode in ('frozen_response','frozen_all'):")
s=s.replace("if mode=='frozen_all':assert", "if mode in ('frozen_all','frozen_bias'):assert")
s=s.replace("if mode=='online' else 0", "if mode in ('online','frozen_bias') else 0")
(b/'frozen_bias_core.py').write_text(s)
