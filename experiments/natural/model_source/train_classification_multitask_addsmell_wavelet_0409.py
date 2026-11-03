# -*- coding: utf-8 -*-
"""
Wavelet-enhanced MoCA subtask classification.

This script keeps the original eye/angle Transformer backbone and adds a
patient-level wavelet feature branch before the final classification head.
It is intended to be compared with train_classification_multitask_addsmell_0409_balanced.py.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pywt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

import train_classification_multitask_addsmell_0409 as base
import train_classification_multitask_addsmell_0409_balanced as bal


def robust_resample_zscore(x, target_len: int):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if x.size == 1:
        return np.zeros(target_len, dtype=np.float32)

    src = np.linspace(0.0, 1.0, x.size, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
    x = np.interp(dst, src, x).astype(np.float32)

    median = float(np.median(x))
    mad = float(np.median(np.abs(x - median))) * 1.4826
    if mad < 1e-6:
        mad = float(np.std(x))
    if mad < 1e-6:
        return np.zeros(target_len, dtype=np.float32)
    return np.clip((x - median) / mad, -8.0, 8.0).astype(np.float32)


def spectral_entropy(power_by_freq):
    p = np.asarray(power_by_freq, dtype=np.float64)
    total = float(np.sum(p))
    if total <= 1e-12:
        return 0.0
    p = p / total
    return float(-np.sum(p * np.log(p + 1e-12)) / math.log(len(p) + 1e-12))


def extract_dwt_trial_features(x, wavelet: str, level: int):
    max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    use_level = max(1, min(level, max_level))
    coeffs = pywt.wavedec(x, wavelet, level=use_level)
    tags = [f"A{use_level}"] + [f"D{i}" for i in range(use_level, 0, -1)]
    energies = np.asarray([float(np.sum(c * c)) for c in coeffs], dtype=np.float64)
    total_energy = float(np.sum(energies) + 1e-12)

    feats = {
        "basic_mean": float(np.mean(x)),
        "basic_std": float(np.std(x)),
        "basic_mean_abs": float(np.mean(np.abs(x))),
        "basic_rms": float(np.sqrt(np.mean(x * x) + 1e-12)),
        "basic_q95_abs": float(np.quantile(np.abs(x), 0.95)),
    }
    signs = np.signbit(x)
    feats["basic_zero_cross_rate"] = float(np.mean(signs[1:] != signs[:-1])) if len(x) > 1 else 0.0

    for tag, coeff, energy in zip(tags, coeffs, energies):
        feats[f"dwt_energy_ratio_{tag}"] = float(energy / total_energy)
        feats[f"dwt_std_{tag}"] = float(np.std(coeff))
        feats[f"dwt_mean_abs_{tag}"] = float(np.mean(np.abs(coeff)))
    return feats


def extract_cwt_subject_features(x, fs: float, wavelet: str, n_freqs: int, f_min: float, f_max: float):
    freqs = np.linspace(f_max, f_min, n_freqs, dtype=np.float32)
    scales = pywt.frequency2scale(wavelet, freqs / fs)
    coeffs, _ = pywt.cwt(x, scales, wavelet, sampling_period=1.0 / fs)
    power = np.abs(coeffs) ** 2
    freq_power = np.mean(power, axis=1)
    total = float(np.sum(freq_power) + 1e-12)

    low = freq_power[freqs <= 4.0]
    mid = freq_power[(freqs > 4.0) & (freqs <= 8.0)]
    high = freq_power[freqs > 8.0]
    return {
        "cwt_log_energy": float(np.log1p(total)),
        "cwt_spectral_entropy": spectral_entropy(freq_power),
        "cwt_dominant_freq": float(freqs[int(np.argmax(freq_power))]),
        "cwt_low_ratio": float(np.sum(low) / total) if low.size else 0.0,
        "cwt_mid_ratio": float(np.sum(mid) / total) if mid.size else 0.0,
        "cwt_high_ratio": float(np.sum(high) / total) if high.size else 0.0,
        "cwt_time_energy_std": float(np.std(np.mean(power, axis=0))),
    }


def extract_subject_wavelet_features(subject_eye, subject_angle, args):
    feature_values = []
    feature_names = []

    def add_feature(prefix, key, value):
        feature_names.append(f"{prefix}_{key}")
        feature_values.append(float(value))

    modalities = [("eye", subject_eye), ("angle", subject_angle)]
    for modality_name, trials in modalities:
        n_channels = int(trials[0].shape[1])
        for channel_idx in range(n_channels):
            prefix = f"{modality_name}_ch{channel_idx:02d}"
            resampled_trials = [
                robust_resample_zscore(trial[:, channel_idx].detach().cpu().numpy(), args.wavelet_len)
                for trial in trials
            ]

            dwt_rows = [
                extract_dwt_trial_features(x, args.dwt_wavelet, args.dwt_level)
                for x in resampled_trials
            ]
            dwt_keys = list(dwt_rows[0].keys())
            dwt_matrix = np.asarray([[row[k] for k in dwt_keys] for row in dwt_rows], dtype=np.float32)
            for col_idx, key in enumerate(dwt_keys):
                add_feature(prefix, f"{key}_trial_mean", np.mean(dwt_matrix[:, col_idx]))
                add_feature(prefix, f"{key}_trial_std", np.std(dwt_matrix[:, col_idx]))

            if args.use_cwt:
                representative = np.mean(np.stack(resampled_trials, axis=0), axis=0)
                cwt_feats = extract_cwt_subject_features(
                    representative,
                    fs=args.fs,
                    wavelet=args.cwt_wavelet,
                    n_freqs=args.cwt_freqs,
                    f_min=args.cwt_freq_min,
                    f_max=args.cwt_freq_max,
                )
                for key, value in cwt_feats.items():
                    add_feature(prefix, key, value)

    return np.asarray(feature_values, dtype=np.float32), feature_names


def build_or_load_wavelet_cache(X_eye_all, X_angle_all, subject_names, args):
    cache_path = Path(args.wavelet_cache)
    config = {
        "wavelet_len": args.wavelet_len,
        "dwt_wavelet": args.dwt_wavelet,
        "dwt_level": args.dwt_level,
        "use_cwt": args.use_cwt,
        "cwt_wavelet": args.cwt_wavelet,
        "cwt_freqs": args.cwt_freqs,
        "cwt_freq_min": args.cwt_freq_min,
        "cwt_freq_max": args.cwt_freq_max,
        "fs": args.fs,
        "version": 2,
    }
    config_json = json.dumps(config, sort_keys=True, ensure_ascii=False)

    if cache_path.exists() and not args.rebuild_wavelet_cache:
        try:
            cached = np.load(cache_path, allow_pickle=True)
            cached_subjects = [str(x) for x in cached["subject_names"].tolist()]
            cached_config = str(cached["config"].tolist())
            if cached_subjects == list(subject_names) and cached_config == config_json:
                print(f"读取小波特征缓存: {cache_path}")
                return (
                    cached["features"].astype(np.float32),
                    [str(x) for x in cached["feature_names"].tolist()],
                )
            print("小波缓存与当前数据或参数不一致，将重新提取。")
        except Exception as exc:
            print(
                f"小波缓存无法读取（{type(exc).__name__}: {exc}），"
                "将删除旧缓存并重新提取（常见于 NumPy 版本不一致）。"
            )
            try:
                cache_path.unlink()
            except OSError:
                pass
    elif args.rebuild_wavelet_cache and cache_path.exists():
        print(f"--rebuild-wavelet-cache: 忽略已有缓存 {cache_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    all_features = []
    feature_names = None
    print("开始提取患者级小波特征...")
    for idx, (eye_trials, angle_trials, subject) in enumerate(zip(X_eye_all, X_angle_all, subject_names), start=1):
        features, names = extract_subject_wavelet_features(eye_trials, angle_trials, args)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise RuntimeError(f"小波特征名不一致: {subject}")
        all_features.append(features)
        if idx == 1 or idx % 20 == 0 or idx == len(subject_names):
            print(f"  wavelet {idx:03d}/{len(subject_names)}: {subject}")

    feature_matrix = np.vstack(all_features).astype(np.float32)
    np.savez_compressed(
        cache_path,
        features=feature_matrix,
        subject_names=np.asarray(subject_names, dtype=object),
        feature_names=np.asarray(feature_names, dtype=object),
        config=np.asarray(config_json, dtype=object),
    )
    print(f"小波特征已缓存: {cache_path} | shape={feature_matrix.shape}")
    return feature_matrix, feature_names


class WaveletEyeFeatureDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        x_eye,
        x_angle,
        y,
        smell,
        wavelet_features,
        eye_mean,
        eye_std,
        angle_mean,
        angle_std,
        clip_z=6.0,
        normalize_eye_angle=True,
    ):
        self.x_eye = x_eye
        self.x_angle = x_angle
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.smell = torch.as_tensor(smell, dtype=torch.float32)
        self.wavelet_features = torch.as_tensor(wavelet_features, dtype=torch.float32)
        self.eye_mean = eye_mean
        self.eye_std = eye_std
        self.angle_mean = angle_mean
        self.angle_std = angle_std
        self.clip_z = clip_z
        self.normalize_eye_angle = normalize_eye_angle

    def __len__(self):
        return len(self.x_eye)

    def _norm_trials(self, trials, mean, std):
        out = []
        for trial in trials:
            x = torch.nan_to_num(trial.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if self.normalize_eye_angle:
                x = (x - mean) / std
                if self.clip_z is not None and self.clip_z > 0:
                    x = torch.clamp(x, -self.clip_z, self.clip_z)
            out.append(x)
        return out

    def __getitem__(self, idx):
        eye = self._norm_trials(self.x_eye[idx], self.eye_mean, self.eye_std)
        angle = self._norm_trials(self.x_angle[idx], self.angle_mean, self.angle_std)
        return (eye, angle, self.smell[idx], self.wavelet_features[idx]), self.y[idx]


def collate_wavelet_trials(batch):
    xs_eye, xs_angle, smells, wavelets, ys = [], [], [], [], []
    for x, y in batch:
        xs_eye.append(x[0])
        xs_angle.append(x[1])
        smells.append(x[2])
        wavelets.append(x[3])
        ys.append(y)
    return (
        xs_eye,
        xs_angle,
        torch.stack(smells, dim=0),
        torch.stack(wavelets, dim=0),
    ), torch.stack(ys, dim=0)


def move_wavelet_x_to_device(x, device):
    xs_eye, xs_angle, smells, wavelets = x
    eye_out = [[trial.to(device) for trial in subject] for subject in xs_eye]
    angle_out = [[trial.to(device) for trial in subject] for subject in xs_angle]
    return eye_out, angle_out, smells.to(device), wavelets.to(device)


class SingleTaskWaveletFusionModel(base.SingleTaskClassificationModel):
    def __init__(self, wavelet_dim: int, wavelet_embed_dim: int = 64, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wavelet_dim = int(wavelet_dim)
        self.wavelet_embed_dim = int(wavelet_embed_dim)
        self.wavelet_encoder = nn.Sequential(
            nn.LayerNorm(self.wavelet_dim),
            nn.Linear(self.wavelet_dim, max(64, self.wavelet_embed_dim * 2)),
            nn.GELU(),
            nn.Dropout(kwargs.get("dropout", 0.2)),
            nn.Linear(max(64, self.wavelet_embed_dim * 2), self.wavelet_embed_dim),
            nn.GELU(),
        )
        mlp_hidden = kwargs.get("mlp_hidden", 192)
        num_classes = kwargs.get("num_classes")
        final_in_dim = self.flat_in_dim + self.smell_concat_dim + self.wavelet_embed_dim
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(final_in_dim),
            nn.Linear(final_in_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(kwargs.get("dropout", 0.2)),
            nn.Linear(mlp_hidden, max(16, mlp_hidden // 2)),
            nn.GELU(),
            nn.Dropout(kwargs.get("dropout", 0.2)),
            nn.Linear(max(16, mlp_hidden // 2), num_classes),
        )

    def forward(self, x, lengths=None):
        xs_eye, xs_angle, smells, wavelets = x

        eye_pairs = self.eye_processor(xs_eye, lengths)
        angle_pairs = self.angle_processor(xs_angle, lengths)

        q = self.fuse_ln_q(angle_pairs)
        k = self.fuse_ln_kv(eye_pairs)
        v = self.fuse_ln_kv(eye_pairs)
        fused_ea, _ = self.fuse_attn(q, k, v, need_weights=False)

        smell_encoded = self._encode_smell(smells)
        for blk in self.bica:
            smell_encoded, fused_ea = blk(smell_encoded, fused_ea)

        fused_flat = fused_ea.reshape(fused_ea.size(0), -1)
        smells = smells.to(device=fused_flat.device, dtype=fused_flat.dtype)
        if smells.dim() == 1:
            smells = smells.unsqueeze(0)
        elif smells.dim() > 2:
            smells = smells.view(smells.size(0), -1)
        smell_for_concat = self.smell_concat_proj(smells)

        wavelets = wavelets.to(device=fused_flat.device, dtype=fused_flat.dtype)
        wavelet_embed = self.wavelet_encoder(wavelets)
        fused_with_features = torch.cat([fused_flat, smell_for_concat, wavelet_embed], dim=1)
        return self.output_mlp(fused_with_features)


def compute_metrics(y_true, y_pred, num_classes: int):
    labels = sorted(set(np.asarray(y_true, dtype=int).tolist()) | set(np.asarray(y_pred, dtype=int).tolist()))
    if not labels:
        labels = list(range(num_classes))
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal_acc),
        "score": float(0.5 * macro_f1 + 0.5 * bal_acc),
    }


def run_epoch(model, loader, loss_fn, device, task_idx: int, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_n = 0
    y_true, y_pred = [], []
    for xb, yb in loader:
        yb = yb.to(device).long()
        target = yb[:, task_idx]
        xb = move_wavelet_x_to_device(xb, device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, target)
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_n = int(target.size(0))
        total_loss += float(loss.detach().cpu()) * batch_n
        total_n += batch_n
        y_true.extend(target.detach().cpu().numpy().tolist())
        y_pred.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())

    return total_loss / max(1, total_n), compute_metrics(y_true, y_pred, model.output_mlp[-1].out_features), np.asarray(y_true), np.asarray(y_pred)


def count_dict(values):
    return {int(k): int(v) for k, v in sorted(Counter(np.asarray(values, dtype=int)).items())}


def standardize_wavelet_train_val(train_features, val_features):
    mean = np.nanmean(train_features, axis=0).astype(np.float32)
    std = np.nanstd(train_features, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    train_z = np.nan_to_num((train_features - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    val_z = np.nan_to_num((val_features - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(train_z, -6.0, 6.0).astype(np.float32), np.clip(val_z, -6.0, 6.0).astype(np.float32), mean, std


def train_one_task(task_idx, data, wavelet_matrix, wavelet_names, args, device):
    X_eye_all, X_angle_all, y_all, subject_names, smell_all = data
    y_task_all = np.asarray(y_all[:, task_idx], dtype=int)
    num_classes = base.NUM_CLASSES[task_idx]
    task_name = base.SUB_NAMES[task_idx]

    valid_mask = (y_task_all >= 0) & (y_task_all < num_classes)
    valid_indices = np.where(valid_mask)[0]
    y_task_valid = y_task_all[valid_indices]
    train_rel, val_rel, split_mode = bal.stratified_or_guarded_split(y_task_valid, args.test_size, args.seed + task_idx)
    train_idx = valid_indices[train_rel]
    val_idx = valid_indices[val_rel]

    X_train_eye = bal.take_by_indices(X_eye_all, train_idx)
    X_train_angle = bal.take_by_indices(X_angle_all, train_idx)
    X_train_smell = bal.take_by_indices(smell_all, train_idx)
    y_train = bal.take_by_indices(y_all, train_idx)

    X_val_eye = bal.take_by_indices(X_eye_all, val_idx)
    X_val_angle = bal.take_by_indices(X_angle_all, val_idx)
    X_val_smell = bal.take_by_indices(smell_all, val_idx)
    y_val = bal.take_by_indices(y_all, val_idx)

    wavelet_train, wavelet_val, wavelet_mean, wavelet_std = standardize_wavelet_train_val(
        wavelet_matrix[train_idx], wavelet_matrix[val_idx]
    )

    if args.normalize:
        eye_mean, eye_std = bal.compute_channel_stats(X_train_eye)
        angle_mean, angle_std = bal.compute_channel_stats(X_train_angle)
    else:
        eye_mean = torch.zeros(X_eye_all[0][0].shape[1], dtype=torch.float32)
        eye_std = torch.ones_like(eye_mean)
        angle_mean = torch.zeros(X_angle_all[0][0].shape[1], dtype=torch.float32)
        angle_std = torch.ones_like(angle_mean)

    print("\n" + "=" * 78)
    print(f"小波融合任务 {task_idx}: {task_name} | {num_classes} 类 | split={split_mode}")
    print(f"训练集: {len(train_idx)} | 验证集: {len(val_idx)} | wavelet_dim={wavelet_matrix.shape[1]}")
    print(f"全量分布: {count_dict(y_task_all)}")
    print(f"训练分布: {count_dict(y_train[:, task_idx])}")
    print(f"验证分布: {count_dict(y_val[:, task_idx])}")

    train_ds = WaveletEyeFeatureDataset(
        X_train_eye, X_train_angle, y_train, X_train_smell, wavelet_train,
        eye_mean, eye_std, angle_mean, angle_std, clip_z=args.clip_z, normalize_eye_angle=args.normalize
    )
    val_ds = WaveletEyeFeatureDataset(
        X_val_eye, X_val_angle, y_val, X_val_smell, wavelet_val,
        eye_mean, eye_std, angle_mean, angle_std, clip_z=args.clip_z, normalize_eye_angle=args.normalize
    )
    sampler, class_weights_np = bal.make_weighted_sampler(y_train[:, task_idx], num_classes, args.seed + task_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_wavelet_trials, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_wavelet_trials, num_workers=0)

    model = SingleTaskWaveletFusionModel(
        wavelet_dim=wavelet_matrix.shape[1],
        wavelet_embed_dim=args.wavelet_embed_dim,
        input_dim_eye=int(X_eye_all[0][0].shape[1]),
        input_dim_angle=int(X_angle_all[0][0].shape[1]),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        mlp_hidden=args.mlp_hidden,
        num_classes=num_classes,
        target_length=args.target_len,
        share_backbone=True,
        num_bica_blocks=args.num_bica_blocks,
        L_smell=args.l_smell,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience, min_lr=args.min_lr
    )
    loss_fn = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights_np, dtype=torch.float32, device=device))

    print(f"class_weights: {[round(float(x), 4) for x in class_weights_np.tolist()]}")
    print(f"device={device} | target_len={args.target_len} | d_model={args.d_model} | wavelet_embed={args.wavelet_embed_dim}")

    best_score = -1.0
    best_state = None
    best_epoch = -1
    wait = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, _ = run_epoch(model, train_loader, loss_fn, device, task_idx, optimizer=optimizer)
        with torch.no_grad():
            val_loss, val_metrics, val_true, val_pred = run_epoch(model, val_loader, loss_fn, device, task_idx, optimizer=None)
        scheduler.step(val_metrics["score"])

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | lr={row['lr']:.2e} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macroF1={val_metrics['macro_f1']:.4f} | "
            f"val_balAcc={val_metrics['balanced_accuracy']:.4f}"
        )

        if val_metrics["score"] > best_score + args.min_delta:
            best_score = val_metrics["score"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            best_true = val_true.copy()
            best_pred = val_pred.copy()
            wait = 0
            print(f"  New best wavelet score={best_score:.4f} @ epoch {epoch}")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  Early stop: {args.patience} epochs without metric improvement.")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    else:
        best_true = val_true.copy()
        best_pred = val_pred.copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"task{task_idx}_{task_name}"
    ckpt_path = out_dir / f"{safe_name}_best_wavelet.pth"
    hist_path = out_dir / f"{safe_name}_history.csv"
    cm_path = out_dir / f"{safe_name}_confusion.csv"
    meta_path = out_dir / f"{safe_name}_meta.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_score": best_score,
            "task_idx": task_idx,
            "task_name": task_name,
            "num_classes": num_classes,
            "class_weights": class_weights_np.tolist(),
            "train_indices": train_idx.tolist(),
            "val_indices": val_idx.tolist(),
            "subject_names_train": [subject_names[int(i)] for i in train_idx],
            "subject_names_val": [subject_names[int(i)] for i in val_idx],
            "wavelet_feature_names": wavelet_names,
            "wavelet_mean": wavelet_mean.tolist(),
            "wavelet_std": wavelet_std.tolist(),
            "eye_mean": eye_mean.tolist(),
            "eye_std": eye_std.tolist(),
            "angle_mean": angle_mean.tolist(),
            "angle_std": angle_std.tolist(),
            "args": vars(args),
        },
        ckpt_path,
    )

    pd.DataFrame(history).to_csv(hist_path, index=False, encoding="utf-8-sig")
    cm = confusion_matrix(best_true, best_pred, labels=list(range(num_classes)))
    pd.DataFrame(cm, index=[f"true_{i}" for i in range(num_classes)], columns=[f"pred_{i}" for i in range(num_classes)]).to_csv(
        cm_path, encoding="utf-8-sig"
    )
    final_metrics = compute_metrics(best_true, best_pred, num_classes)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_idx": task_idx,
                "task_name": task_name,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "best_metrics": final_metrics,
                "wavelet_dim": int(wavelet_matrix.shape[1]),
                "split_mode": split_mode,
                "all_distribution": count_dict(y_task_all),
                "train_distribution": count_dict(y_train[:, task_idx]),
                "val_distribution": count_dict(y_val[:, task_idx]),
                "checkpoint": str(ckpt_path),
                "history_csv": str(hist_path),
                "confusion_csv": str(cm_path),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"已保存小波融合模型: {ckpt_path}")
    return {"task_idx": task_idx, "task_name": task_name, "best_epoch": best_epoch, "best_score": best_score, **final_metrics}


def parse_args():
    parser = argparse.ArgumentParser(description="Wavelet-enhanced MoCA subtask classification.")
    parser.add_argument("--base-dir", default="./Dataset/eye1_pro_new")
    parser.add_argument("--moca-path", default="./Dataset/Moca_new.xlsx")
    parser.add_argument("--smell-path", default="./Dataset/smellfinal.xls")
    parser.add_argument("--out-dir", default="./model_wavelet")
    parser.add_argument("--wavelet-cache", default="./cache/wavelet_eye1_pro_new_v2.npz")
    parser.add_argument("--rebuild-wavelet-cache", action="store_true")
    parser.add_argument("--tasks", nargs="+", type=int, default=list(range(7)))

    parser.add_argument("--wavelet-len", type=int, default=256)
    parser.add_argument("--dwt-wavelet", default="db4")
    parser.add_argument("--dwt-level", type=int, default=4)
    parser.add_argument("--use-cwt", action="store_true", default=True)
    parser.add_argument("--no-cwt", dest="use_cwt", action="store_false")
    parser.add_argument("--cwt-wavelet", default="morl")
    parser.add_argument("--cwt-freqs", type=int, default=12)
    parser.add_argument("--cwt-freq-min", type=float, default=0.5)
    parser.add_argument("--cwt-freq-max", type=float, default=15.0)
    parser.add_argument("--fs", type=float, default=60.0)
    parser.add_argument("--wavelet-embed-dim", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--target-len", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-bica-blocks", type=int, default=1)
    parser.add_argument("--l-smell", type=int, default=64)
    parser.add_argument("--mlp-hidden", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false")
    parser.add_argument("--clip-z", type=float, default=6.0)
    parser.set_defaults(normalize=True)
    return parser.parse_args()


def main():
    args = parse_args()
    bal.set_seed(args.seed)
    device = bal.choose_device(args.device)

    print("=" * 78)
    print("Wavelet-enhanced balanced training")
    print(f"推荐解释器: {bal.PYTHON_ENV}")
    print("=" * 78)
    data = base.load_eyedata1_new(args.base_dir, args.moca_path, args.smell_path)
    X_eye_all, X_angle_all, y_all, subject_names, smell_all = data
    print(f"数据量: subjects={len(subject_names)}, y={np.asarray(y_all).shape}, smell={np.asarray(smell_all).shape}")

    wavelet_matrix, wavelet_names = build_or_load_wavelet_cache(X_eye_all, X_angle_all, subject_names, args)
    print(f"小波特征矩阵: {wavelet_matrix.shape}")

    results = []
    for task_idx in args.tasks:
        if task_idx < 0 or task_idx >= len(base.SUB_NAMES):
            raise ValueError(f"非法任务编号: {task_idx}")
        results.append(train_one_task(task_idx, data, wavelet_matrix, wavelet_names, args, device))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary_wavelet.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n全部指定小波融合任务训练完成")
    print(f"汇总结果: {summary_path}")


if __name__ == "__main__":
    main()
