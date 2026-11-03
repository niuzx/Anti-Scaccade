"""Generated random inputs only: no clinical samples or performance claims."""
from pathlib import Path
import sys,json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];EXPERIMENT=ROOT/'experiments/balanced'
sys.path[:0]=[str(EXPERIMENT),str(EXPERIMENT/'model_source')]
from run_revision import splits,metric
from latest_model import SubscoreModel
cfg=json.loads((ROOT/'configs/balanced.json').read_text());cfg.update(target_length=16,L_smell=16)
torch.set_num_threads(2);torch.manual_seed(7)
for fold in splits(20):
 assert not set(fold['train'])&set(fold['test'])
 assert not set(fold['fit'])&set(fold['validation'])
assert metric(np.array([1.,2.,3.]),np.array([1.,2.,3.]))['MAE']==0
x=(torch.randn(2,40,16,12),torch.randn(2,40,16,8),torch.randn(2,16),torch.randn(2,980))
for mode in ['olfactory_only','eye_only','multimodal_no_wavelet','multimodal']:
 model=SubscoreModel(mode,6,cfg).eval();prediction=model(x)
 assert prediction.shape==(2,6) and torch.isfinite(prediction).all()
 changed=[v.clone() for v in x]
 excluded={'olfactory_only':[0,1,3],'eye_only':[2],'multimodal_no_wavelet':[3],'multimodal':[]}[mode]
 for i in excluded:changed[i]=torch.randn_like(changed[i])*100
 if excluded:assert torch.equal(prediction,model(tuple(changed))),mode
 prediction.square().mean().backward()
 assert any(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
 print('PASS',mode)
print('All CPU smoke checks passed; this is not a training reproduction.')
