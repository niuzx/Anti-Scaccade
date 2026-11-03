# -*- coding: utf-8 -*-
"""
Improved training entry for train_classification_multitask_addsmell_0409.py.

Main changes:
1. Per-task stratified validation split.
2. WeightedRandomSampler instead of manual downsampling.
3. Class-weighted CrossEntropyLoss.
4. Early stopping by macro-F1 / balanced accuracy instead of validation loss only.
5. Device guard for RTX 30xx + old CUDA builds.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, WeightedRandomSampler

import train_classification_multitask_addsmell_0409 as base


PYTHON_ENV = r"C:\ProgramData\anaconda3\envs\dti_env\python.exe"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def take_by_indices(values, indices):
    indices = np.asarray(indices, dtype=int)
    if isinstance(values, np.ndarray):
        return values[indices]
    return [values[int(i)] for i in indices]


def count_dict(values):
    return {int(k): int(v) for k, v in sorted(Counter(np.asarray(values, dtype=int)).items())}


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)

    if not torch.cuda.is_available():
        return torch.device("cpu")

    device_name = torch.cuda.get_device_name(0)
    print(f"检测到 CUDA 设备: {device_name}，使用 cuda:0")
    return torch.device("cuda:0")


def stratified_or_guarded_split(y_task, test_size: float, seed: int):
    """Keep singleton classes in train, then stratify the remaining classes."""
    y_task = np.asarray(y_task, dtype=int)
    all_indices = np.arange(len(y_task))
    counts = Counter(y_task.tolist())
    rare_classes = {cls for cls, n in counts.items() if n < 2}

    if rare_classes:
        rare_mask = np.isin(y_task, list(rare_classes))
        rare_indices = all_indices[rare_mask]
        regular_indices = all_indices[~rare_mask]
        regular_y = y_task[regular_indices]
        regular_counts = Counter(regular_y.tolist())
        if len(regular_counts) >= 2 and min(regular_counts.values()) >= 2:
            splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            train_rel, val_rel = next(splitter.split(np.zeros(len(regular_y)), regular_y))
            train_idx = np.concatenate([regular_indices[train_rel], rare_indices])
            val_idx = regular_indices[val_rel]
            return np.sort(train_idx), np.sort(val_idx), f"stratified_keep_rare_train:{sorted(rare_classes)}"

    if len(counts) >= 2 and min(counts.values()) >= 2:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros(len(y_task)), y_task))
        return np.sort(train_idx), np.sort(val_idx), "stratified"

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(all_indices)
    n_val = max(1, int(round(len(all_indices) * test_size)))
    val_idx = shuffled[:n_val]
    train_idx = shuffled[n_val:]
    return np.sort(train_idx), np.sort(val_idx), "random_fallback"


def make_class_weights(y_task, num_classes: int, power: float = 0.5, max_weight: float = 10.0):
    y_task = np.asarray(y_task, dtype=int)
    counts = np.bincount(y_task, minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    if present.any():
        n_present = float(present.sum())
        total = float(counts[present].sum())
        weights[present] = (total / (n_present * counts[present])) ** power
        weights[present] = np.clip(weights[present], 0.05, max_weight)
    return counts, weights


def make_weighted_sampler(y_task, num_classes: int, seed: int):
    _, class_weights = make_class_weights(y_task, num_classes)
    sample_weights = class_weights[np.asarray(y_task, dtype=int)]
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    return sampler, class_weights


def compute_channel_stats(subject_trials):
    sums = None
    sums2 = None
    n_points = 0
    for subject in subject_trials:
        for trial in subject:
            x = torch.nan_to_num(trial.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
            if x.numel() == 0:
                continue
            if sums is None:
                channels = x.size(-1)
                sums = torch.zeros(channels, dtype=torch.float64)
                sums2 = torch.zeros(channels, dtype=torch.float64)
            xd = x.to(dtype=torch.float64)
            sums += xd.sum(dim=0)
            sums2 += (xd * xd).sum(dim=0)
            n_points += int(x.size(0))

    if sums is None or n_points == 0:
        raise ValueError("无法计算归一化统计量：训练数据为空")

    mean = sums / float(n_points)
    var = torch.clamp(sums2 / float(n_points) - mean * mean, min=1e-6)
    std = torch.sqrt(var)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean.float(), std.float()


class NormalizedEyeFeatureDatasetWithSmell(torch.utils.data.Dataset):
    def __init__(
        self,
        x1_eye,
        x1_angle,
        y,
        smell_features,
        eye_mean,
        eye_std,
        angle_mean,
        angle_std,
        clip_z=None,
    ):
        self.x1_eye = x1_eye
        self.x1_angle = x1_angle
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.smell_features = torch.as_tensor(smell_features, dtype=torch.float32)
        self.eye_mean = eye_mean
        self.eye_std = eye_std
        self.angle_mean = angle_mean
        self.angle_std = angle_std
        self.clip_z = clip_z

    def __len__(self):
        return len(self.x1_eye)

    def _normalize_trials(self, trials, mean, std):
        out = []
        for trial in trials:
            x = torch.nan_to_num(trial.float(), nan=0.0, posinf=0.0, neginf=0.0)
            x = (x - mean) / std
            if self.clip_z is not None and self.clip_z > 0:
                x = torch.clamp(x, -self.clip_z, self.clip_z)
            out.append(x)
        return out

    def __getitem__(self, idx):
        eye = self._normalize_trials(self.x1_eye[idx], self.eye_mean, self.eye_std)
        angle = self._normalize_trials(self.x1_angle[idx], self.angle_mean, self.angle_std)
        return (eye, angle), self.y[idx], self.smell_features[idx]


def compute_metrics(y_true, y_pred, num_classes: int):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    metric_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if not metric_labels:
        metric_labels = list(range(num_classes))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=metric_labels, average="macro", zero_division=0)
    try:
        bal_acc = balanced_accuracy_score(y_true, y_pred)
    except Exception:
        bal_acc = 0.0
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal_acc),
        "score": float(0.5 * macro_f1 + 0.5 * bal_acc),
    }


def run_one_epoch(model, loader, loss_fn, device, task_idx: int, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_n = 0
    preds = []
    trues = []

    for xb, yb in loader:
        yb = yb.to(device).long()
        y_task = yb[:, task_idx]
        xb = base._move_x_to_device(xb, device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(xb)
        loss = loss_fn(logits, y_task)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        batch_n = int(y_task.size(0))
        total_loss += float(loss.detach().cpu()) * batch_n
        total_n += batch_n
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().numpy().tolist())
        trues.extend(y_task.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, total_n)
    metrics = compute_metrics(trues, preds, model.output_mlp[-1].out_features)
    return avg_loss, metrics, np.asarray(trues, dtype=int), np.asarray(preds, dtype=int)


def train_one_task(task_idx: int, data, args, device: torch.device):
    X_eye_all, X_angle_all, y_all, subject_names, smell_all = data
    y_task_all = np.asarray(y_all[:, task_idx], dtype=int)
    num_classes = base.NUM_CLASSES[task_idx]
    task_name = base.SUB_NAMES[task_idx]

    valid_mask = (y_task_all >= 0) & (y_task_all < num_classes)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < len(y_task_all):
        print(f"任务 {task_idx} {task_name}: 过滤 {len(y_task_all) - len(valid_indices)} 个越界标签样本")

    y_task_valid = y_task_all[valid_indices]
    train_rel, val_rel, split_mode = stratified_or_guarded_split(y_task_valid, args.test_size, args.seed + task_idx)
    train_idx = valid_indices[train_rel]
    val_idx = valid_indices[val_rel]

    X_train = (
        take_by_indices(X_eye_all, train_idx),
        take_by_indices(X_angle_all, train_idx),
        take_by_indices(smell_all, train_idx),
    )
    X_val = (
        take_by_indices(X_eye_all, val_idx),
        take_by_indices(X_angle_all, val_idx),
        take_by_indices(smell_all, val_idx),
    )
    y_train = take_by_indices(y_all, train_idx)
    y_val = take_by_indices(y_all, val_idx)

    print("\n" + "=" * 78)
    print(f"任务 {task_idx}: {task_name} | {num_classes} 类 | split={split_mode}")
    print(f"训练集: {len(train_idx)} | 验证集: {len(val_idx)}")
    print(f"全量分布: {count_dict(y_task_all)}")
    print(f"训练分布: {count_dict(y_train[:, task_idx])}")
    print(f"验证分布: {count_dict(y_val[:, task_idx])}")

    norm_state = None
    if args.normalize:
        eye_mean, eye_std = compute_channel_stats(X_train[0])
        angle_mean, angle_std = compute_channel_stats(X_train[1])
        norm_state = {
            "eye_mean": eye_mean.tolist(),
            "eye_std": eye_std.tolist(),
            "angle_mean": angle_mean.tolist(),
            "angle_std": angle_std.tolist(),
            "clip_z": args.clip_z,
        }
        print("已启用训练集通道标准化")
        print(f"eye std range: {float(eye_std.min()):.4f} - {float(eye_std.max()):.4f}")
        print(f"angle std range: {float(angle_std.min()):.4f} - {float(angle_std.max()):.4f}")
        train_ds = NormalizedEyeFeatureDatasetWithSmell(
            X_train[0], X_train[1], y_train, X_train[2],
            eye_mean, eye_std, angle_mean, angle_std, clip_z=args.clip_z
        )
        val_ds = NormalizedEyeFeatureDatasetWithSmell(
            X_val[0], X_val[1], y_val, X_val[2],
            eye_mean, eye_std, angle_mean, angle_std, clip_z=args.clip_z
        )
    else:
        train_ds = base.EyeFeatureDatasetWithSmell(X_train[0], X_train[1], y_train, X_train[2])
        val_ds = base.EyeFeatureDatasetWithSmell(X_val[0], X_val[1], y_val, X_val[2])
    sampler, class_weights_np = make_weighted_sampler(y_train[:, task_idx], num_classes, args.seed + task_idx)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=base.collate_trials,
        drop_last=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=base.collate_trials,
        drop_last=False,
        num_workers=0,
    )

    input_dim_eye = int(X_eye_all[valid_indices[0]][0].shape[1])
    input_dim_angle = int(X_angle_all[valid_indices[0]][0].shape[1])
    model = base.SingleTaskClassificationModel(
        input_dim_eye=input_dim_eye,
        input_dim_angle=input_dim_angle,
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
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, args.lr_patience),
        min_lr=args.min_lr,
    )
    class_weights = torch.as_tensor(class_weights_np, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    print(f"class_weights: {[round(float(x), 4) for x in class_weights_np.tolist()]}")
    print(
        f"device={device} | target_len={args.target_len} | d_model={args.d_model} | "
        f"nhead={args.nhead} | batch_size={args.batch_size}"
    )

    best_score = -1.0
    best_state = None
    best_epoch = -1
    wait = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics, _, _ = run_one_epoch(
            model, train_loader, loss_fn, device, task_idx, optimizer=optimizer
        )
        with torch.no_grad():
            val_loss, val_metrics, val_true, val_pred = run_one_epoch(
                model, val_loader, loss_fn, device, task_idx, optimizer=None
            )

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
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr={row['lr']:.2e} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macroF1={val_metrics['macro_f1']:.4f} | "
            f"val_balAcc={val_metrics['balanced_accuracy']:.4f}"
        )

        improved = val_metrics["score"] > best_score + args.min_delta
        if improved:
            best_score = val_metrics["score"]
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_val_true = val_true.copy()
            best_val_pred = val_pred.copy()
            print(f"  New best: score={best_score:.4f} @ epoch {epoch}")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  Early stop: {args.patience} epochs without metric improvement.")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    else:
        best_val_true = val_true.copy()
        best_val_pred = val_pred.copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"task{task_idx}_{task_name}"
    ckpt_path = out_dir / f"{safe_name}_best_balanced.pth"
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
            "normalization": norm_state,
            "args": vars(args),
        },
        ckpt_path,
    )

    import pandas as pd

    pd.DataFrame(history).to_csv(hist_path, index=False, encoding="utf-8-sig")
    cm = confusion_matrix(best_val_true, best_val_pred, labels=list(range(num_classes)))
    pd.DataFrame(cm, index=[f"true_{i}" for i in range(num_classes)], columns=[f"pred_{i}" for i in range(num_classes)]).to_csv(
        cm_path, encoding="utf-8-sig"
    )
    final_metrics = compute_metrics(best_val_true, best_val_pred, num_classes)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_idx": task_idx,
                "task_name": task_name,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "best_metrics": final_metrics,
                "split_mode": split_mode,
                "all_distribution": count_dict(y_task_all),
                "train_distribution": count_dict(y_train[:, task_idx]),
                "val_distribution": count_dict(y_val[:, task_idx]),
                "checkpoint": str(ckpt_path),
                "history_csv": str(hist_path),
                "confusion_csv": str(cm_path),
                "normalization": norm_state,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"已保存模型: {ckpt_path}")
    print(f"已保存训练曲线CSV: {hist_path}")
    print(f"已保存混淆矩阵CSV: {cm_path}")
    return {
        "task_idx": task_idx,
        "task_name": task_name,
        "best_epoch": best_epoch,
        "best_score": best_score,
        **final_metrics,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Balanced MoCA subtask classification training.")
    parser.add_argument("--base-dir", default="./Dataset/eye1_pro_new")
    parser.add_argument("--moca-path", default="./Dataset/Moca_new.xlsx")
    parser.add_argument("--smell-path", default="./Dataset/smellfinal.xls")
    parser.add_argument("--out-dir", default="./model_balanced")
    parser.add_argument("--tasks", nargs="+", type=int, default=list(range(7)), help="Task ids, e.g. --tasks 0 5")
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
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...")
    parser.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable train-set channel normalization.")
    parser.add_argument("--clip-z", type=float, default=6.0, help="Clip normalized eye/angle features to [-clip_z, clip_z].")
    parser.set_defaults(normalize=True)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    print("=" * 78)
    print("Balanced training for MoCA subtask classification")
    print(f"推荐解释器: {PYTHON_ENV}")
    print("=" * 78)
    data = base.load_eyedata1_new(args.base_dir, args.moca_path, args.smell_path)
    X_eye_all, X_angle_all, y_all, subject_names, smell_all = data
    print(f"数据量: subjects={len(subject_names)}, y={np.asarray(y_all).shape}, smell={np.asarray(smell_all).shape}")
    print(f"嗅觉非零样本: {int((np.abs(smell_all).sum(axis=1) > 0).sum())}/{len(smell_all)}")

    results = []
    for task_idx in args.tasks:
        if task_idx < 0 or task_idx >= len(base.SUB_NAMES):
            raise ValueError(f"非法任务编号: {task_idx}")
        results.append(train_one_task(task_idx, data, args, device))

    summary_path = Path(args.out_dir) / "summary_balanced.csv"
    import pandas as pd

    pd.DataFrame(results).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print("\n全部指定任务训练完成")
    print(f"汇总结果: {summary_path}")


if __name__ == "__main__":
    main()
