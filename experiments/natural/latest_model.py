"""Modality ablations of the server's wavelet model; immutable source snapshots."""
from pathlib import Path
import json
from types import SimpleNamespace, MethodType
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent

def fast_processor(self, x, lengths=None):
    # Only already-pooled tensors use the fast path. No learned computation is removed.
    if isinstance(x, torch.Tensor) and lengths is None and x.shape[2] == self.target_length:
        pooled = x
    else:
        pooled = self._pool_trials_to_fixed_len(x, lengths)
    b, t, length, channels = pooled.shape
    z = self.pos_encoding(self.input_proj(pooled.reshape(b*t, length, channels)))
    for block in self.self_attn_layers:
        z = block(z)
    z = z.reshape(b, t, length, self.d_model)
    pairs = self.rolling_transformer(z[:, :-1].reshape(b*(t-1), length, self.d_model),
                                     z[:, 1:].reshape(b*(t-1), length, self.d_model))
    return self.pooling(pairs.transpose(1, 2)).squeeze(-1).reshape(b, t-1, self.d_model)

def optimize_processors(core):
    for name in ('eye_processor', 'angle_processor'):
        if hasattr(core, name):
            obj = getattr(core, name)
            obj.forward = MethodType(fast_processor, obj)

def head(dim, classes, cfg):
    h = cfg['mlp_hidden']; drop = cfg['dropout']
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,h), nn.GELU(), nn.Dropout(drop),
                         nn.Linear(h,max(16,h//2)), nn.GELU(), nn.Dropout(drop),
                         nn.Linear(max(16,h//2),classes))

class SubscoreModel(nn.Module):
    def __init__(self, mode, classes, cfg):
        super().__init__()
        from train_classification_multitask_addsmell_wavelet_0409 import SingleTaskWaveletFusionModel
        from train_classification_multitask_addsmell_0409 import SingleTaskClassificationModel
        self.mode = mode
        kwargs = dict(input_dim_eye=12,input_dim_angle=8,d_model=cfg['d_model'],nhead=cfg['nhead'],
                      num_layers=cfg['num_layers'],dropout=cfg['dropout'],mlp_hidden=cfg['mlp_hidden'],
                      num_classes=classes,target_length=cfg['target_length'],
                      num_bica_blocks=cfg['num_bica_blocks'],L_smell=cfg['L_smell'],share_backbone=True)
        if mode in ('multimodal','eye_only'):
            self.core = SingleTaskWaveletFusionModel(wavelet_dim=cfg['wavelet_dim'],
                         wavelet_embed_dim=cfg['wavelet_embed_dim'],**kwargs)
        else:
            self.core = SingleTaskClassificationModel(**kwargs)
        c = self.core
        if mode == 'eye_only':
            for name in ['smell_conv','smell_temporal','smell_pe','bica','smell_concat_proj']:
                delattr(c,name)
            c.output_mlp = head(c.flat_in_dim+cfg['wavelet_embed_dim'],classes,cfg)
        elif mode == 'olfactory_only':
            for name in ['eye_processor','angle_processor','fuse_ln_q','fuse_ln_kv','fuse_attn','bica']:
                delattr(c,name)
        optimize_processors(c)

    def forward(self, x):
        e,a,s,w = x; c = self.core
        if self.mode == 'multimodal':
            return c((e,a,s,w))
        if self.mode == 'multimodal_no_wavelet':
            return c((e,a,s))
        if self.mode == 'eye_only':
            ep=c.eye_processor(e); ap=c.angle_processor(a)
            kv=c.fuse_ln_kv(ep)
            z,_=c.fuse_attn(c.fuse_ln_q(ap),kv,kv,need_weights=False)
            return c.output_mlp(torch.cat([z.flatten(1),c.wavelet_encoder(w)],1))
        # No eye-derived inputs or wavelet branch exist in this model.
        z=c._encode_smell(s)
        z=F.adaptive_avg_pool1d(z.transpose(1,2),39).transpose(1,2).flatten(1)
        return c.output_mlp(torch.cat([z,c.smell_concat_proj(s)],1))

def wavelet_inputs(rows, cfg):
    from audit_data import sha
    from train_classification_multitask_addsmell_wavelet_0409 import extract_subject_wavelet_features
    audit=json.loads((ROOT/'audit/audit_summary.json').read_text())
    key=sha(ROOT/'audit/cohort.json')+audit['npy_sha256']+sha(__file__)+sha(ROOT/'model_source/train_classification_multitask_addsmell_wavelet_0409.py')
    cache=ROOT/'cache_wavelet.npz'
    if cache.exists():
        z=np.load(cache,allow_pickle=False)
        if str(z['key'])==key:
            return torch.from_numpy(z['features'])
    args=SimpleNamespace(**cfg['wavelet'])
    features=[]; names=None
    for i,row in enumerate(rows):
        modalities=[]
        for mod in ('eye','angle'):
            modalities.append([torch.from_numpy(np.load(Path(row['folder'])/f'trial{t}_{mod}.npy').astype('float32')) for t in range(1,41)])
        values,current_names=extract_subject_wavelet_features(*modalities,args)
        if names is None: names=current_names
        assert names==current_names and np.isfinite(values).all()
        features.append(values)
        if i%20==0: print(f'wavelet features {i+1}/{len(rows)}',flush=True)
    array=np.stack(features)
    assert array.shape==(len(rows),cfg['wavelet_dim'])
    np.savez_compressed(cache,features=array,key=key,feature_names=np.array(names))
    return torch.from_numpy(array)
