import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from torch.utils.data import WeightedRandomSampler
import math
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import os
from pathlib import Path
import random
from pypinyin import lazy_pinyin, Style
from model_share_transformer import EyeFeatureDataset, SelfAttentionBlock, RollingTransformer

# ========================= 训练：RollingTransformer 单任务独立分类模型（No-PE + 定制抽样） =========================

SUB_NAMES = ["视空间总分","命名总分","注意总分","语言总分","抽象总分","延迟回忆总分","定向总分"]
NUM_CLASSES = [6, 4, 7, 4, 3, 6, 7]
BATCH_SIZES = [6, 4, 7, 4, 3, 6, 7]

def chinese_to_pinyin(text):
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    chinese_chars = ''.join([c for c in text if '\u4e00' <= c <= '\u9fff'])
    if not chinese_chars:
        return ""
    pinyin = lazy_pinyin(chinese_chars, style=Style.NORMAL)
    return ''.join(pinyin).lower()

def train_test_split(X, y, test_size=0.2, random_state=42):
    np.random.seed(random_state); random.seed(random_state)
    n_samples = len(X); n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    if isinstance(X, list):
        X_train = [X[i] for i in train_indices]; X_test = [X[i] for i in test_indices]
    else:
        X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test

# ===================== 读取数据（顺序对齐） =====================
def safe_to_float(val, default=0.0):
    """安全地将值转换为float"""
    if pd.isna(val):
        return default
    if isinstance(val, (int, float)):
        return float(val)
    val_str = str(val).strip()
    if not val_str or val_str.lower() in ['nan', 'none', '']:
        return default
    import re
    match = re.search(r'[\d.]+', val_str)
    return float(match.group()) if match else default

def load_eyedata1_new(base_dir="./Dataset/eye1_pro_new", moca_path="./Dataset/Moca_new.xlsx", smell_path="./Dataset/smellfinal.xls"):
    base_path = Path(base_dir)
    moca_data = pd.read_excel(moca_path, sheet_name="Sheet1", engine="openpyxl")
    
    # 读取smell数据
    try:
        smell_data = pd.read_excel(smell_path, engine="openpyxl")
        print(f"成功读取smell数据: {smell_data.shape}")
    except:
        try:
            smell_data = pd.read_excel(smell_path, engine="xlrd")
            print(f"成功读取smell数据: {smell_data.shape}")
        except Exception as e:
            print(f"警告: 无法读取smell数据: {e}")
            smell_data = None
    
    print(f"\n正在读取数据，共 {len([f for f in base_path.iterdir() if f.is_dir()])} 个受试者文件夹")
    print(f"Moca文件行数: {len(moca_data)}")
    if smell_data is not None:
        print(f"Smell文件行数: {len(smell_data)}")
    
    # ========== 通过姓名列匹配数据 ==========
    # 1. 找到Moca数据的姓名列（通常是第2列，索引1）
    moca_name_col_idx = 1  # 默认第2列
    moca_names = []
    for idx in range(len(moca_data)):
        if moca_name_col_idx < len(moca_data.columns):
            name = str(moca_data.iloc[idx, moca_name_col_idx]).strip()
            moca_names.append(name)
        else:
            moca_names.append("")
    
    # 创建Moca姓名到索引的映射（使用拼音匹配）
    moca_pinyin_map = {}
    for idx, name in enumerate(moca_names):
        if pd.notna(name) and name != "nan" and name != "":
            pinyin = chinese_to_pinyin(name)
            if pinyin:
                # 如果出现重复拼音，保留第一个
                if pinyin not in moca_pinyin_map:
                    moca_pinyin_map[pinyin] = idx
    
    print(f"成功读取 {len(moca_pinyin_map)} 个Moca姓名（去重后）")
    
    # 2. 找到Smell数据的姓名列
    smell_name_col = None
    smell_names = None
    if smell_data is not None:
        col_names = list(smell_data.columns)
        # 尝试找到姓名列
        for col in col_names:
            col_str = str(col).strip()
            if '姓名' in col_str or '名字' in col_str or 'name' in col_str.lower():
                smell_name_col = col
                break
        
        # 如果没找到，尝试第一列或第二列
        if smell_name_col is None:
            if len(col_names) > 0:
                smell_name_col = col_names[0]
                print(f"未找到明确的姓名列，使用第一列: {smell_name_col}")
        
        if smell_name_col is None:
            print("⚠️ 警告: 无法找到smell数据的姓名列")
            smell_data = None
        else:
            smell_names = smell_data[smell_name_col].astype(str).str.strip()
            print(f"成功读取 {len(smell_names)} 个Smell姓名")
    
    # 3. 找到识别1~识别16列的索引
    smell_cols = []
    if smell_data is not None:
        col_names = list(smell_data.columns)
        target_cols = ['识别1', '识别2', '识别3', '识别4', '识别5', '识别6', '识别7', '识别8', 
                      '识别9', '识别10', '识别11', '识别12', '识别13', '识别14', '识别15', '识别16']
        col_name_map = {str(col).strip(): i for i, col in enumerate(col_names)}
        col_name_map_lower = {str(col).strip().lower(): i for i, col in enumerate(col_names)}
        
        for target_col in target_cols:
            if target_col in col_name_map:
                smell_cols.append(col_name_map[target_col])
            elif target_col.lower() in col_name_map_lower:
                smell_cols.append(col_name_map_lower[target_col.lower()])
        
        # 如果没找到完整的列，尝试正则匹配
        if len(smell_cols) < 16:
            smell_cols = []
            import re
            for i, col in enumerate(col_names):
                col_str = str(col).strip()
                if '识别' in col_str:
                    match = re.search(r'识别.*?(\d+)', col_str)
                    if match:
                        num = int(match.group(1))
                        if 1 <= num <= 16:
                            smell_cols.append((num, i))
            if smell_cols:
                smell_cols.sort(key=lambda x: x[0])
                smell_cols = [idx for _, idx in smell_cols][:16]
        
        if len(smell_cols) < 16:
            print(f"⚠️ 警告: 未找到完整的识别1~识别16列（找到{len(smell_cols)}列），将使用前16列或填充0")
            if len(smell_cols) == 0:
                smell_cols = list(range(min(16, len(col_names))))
        else:
            print(f"成功找到识别1~识别16列（共{len(smell_cols)}列）")
    
    # ========== 读取眼动数据并匹配 ==========
    X_eye, X_angle, y, subject_names, smell_features_list = [], [], [], [], []
    all_folders = sorted([f for f in base_path.iterdir() if f.is_dir()], key=lambda x: x.name)
    target_col_indices = [9, 13, 18, 23, 26, 32, 39]
    
    matched_count = 0
    unmatched_folders = []
    data_idx = 0  # 用于按顺序匹配的索引
    name_matched_count = 0  # 通过姓名匹配成功的数量
    sequential_matched_count = 0  # 通过顺序匹配的数量
    
    # 调试：打印前几个文件夹名和Moca姓名
    print(f"\n调试信息：")
    print(f"前5个文件夹名: {[f.name for f in all_folders[:5]]}")
    print(f"前5个Moca姓名: {moca_names[:5]}")
    print(f"前5个Moca姓名拼音: {[chinese_to_pinyin(name) for name in moca_names[:5]]}")
    
    for folder in all_folders:
        subject_name = folder.name  # 文件夹名称（可能是姓名或拼音）
        trials_eye, trials_angle = [], []
        valid_trials = 0
        for i in range(1, 41):
            eye_f = folder / f"trial{i}_eye.npy"; ang_f = folder / f"trial{i}_angle.npy"
            if eye_f.exists() and ang_f.exists():
                trials_eye.append(torch.tensor(np.load(eye_f), dtype=torch.float32))
                trials_angle.append(torch.tensor(np.load(ang_f), dtype=torch.float32))
                valid_trials += 1
        if valid_trials == 0:
            continue  # 跳过无效文件夹
        
        if len(trials_eye) > 0:
            C_eye = trials_eye[0].shape[1]
            C_angle = trials_angle[0].shape[1] if len(trials_angle) > 0 else C_eye
        else:
            C_eye, C_angle = 10, 10
        while len(trials_eye) < 40:
            trials_eye.append(torch.zeros(1, C_eye)); trials_angle.append(torch.zeros(1, C_angle))
        
        # ========== 通过姓名匹配Moca和Smell数据 ==========
        # 从文件夹名中提取拼音部分（处理 person_XXX_Name 格式）
        folder_name_clean = subject_name.strip()
        # 如果文件夹名包含下划线，尝试提取最后一部分（姓名部分）
        if '_' in folder_name_clean:
            parts = folder_name_clean.split('_')
            # 取最后一部分作为可能的姓名
            name_part = parts[-1]
            # 将驼峰格式转换为小写（HuangChaoE -> huangchaoe）
            # 直接转换为小写即可，因为驼峰格式转小写就是连续的拼音
            import re
            name_part_clean = re.sub(r'[^a-zA-Z]', '', name_part).lower()
        else:
            import re
            name_part_clean = re.sub(r'[^a-zA-Z]', '', folder_name_clean).lower()
        
        # 尝试匹配Moca数据
        folder_pinyin = chinese_to_pinyin(subject_name)  # 如果文件夹名是中文
        moca_idx = None
        
        # 策略1: 从文件夹名提取的拼音部分在moca_pinyin_map中
        if name_part_clean and name_part_clean in moca_pinyin_map:
            moca_idx = moca_pinyin_map[name_part_clean]
            name_matched_count += 1
        
        # 策略2: 文件夹名的拼音在moca_pinyin_map中（如果文件夹名是中文）
        if moca_idx is None and folder_pinyin and folder_pinyin in moca_pinyin_map:
            moca_idx = moca_pinyin_map[folder_pinyin]
            name_matched_count += 1
        
        # 策略3: 文件夹名直接等于Moca姓名
        if moca_idx is None:
            for idx, moca_name in enumerate(moca_names):
                if subject_name.strip() == str(moca_name).strip():
                    moca_idx = idx
                    name_matched_count += 1
                    break
        
        # 策略4: 从文件夹名提取的拼音等于Moca姓名的拼音
        if moca_idx is None:
            for idx, moca_name in enumerate(moca_names):
                if pd.notna(moca_name) and moca_name != "nan" and moca_name != "":
                    moca_pinyin = chinese_to_pinyin(str(moca_name))
                    if name_part_clean and moca_pinyin and name_part_clean == moca_pinyin:
                        moca_idx = idx
                        name_matched_count += 1
                        break
        
        # 策略5: 文件夹名的拼音等于Moca姓名的拼音（如果文件夹名是中文）
        if moca_idx is None:
            for idx, moca_name in enumerate(moca_names):
                if pd.notna(moca_name) and moca_name != "nan" and moca_name != "":
                    moca_pinyin = chinese_to_pinyin(str(moca_name))
                    if folder_pinyin and moca_pinyin and folder_pinyin == moca_pinyin:
                        moca_idx = idx
                        name_matched_count += 1
                        break
        
        # 策略6: 文件夹名包含Moca姓名，或Moca姓名包含文件夹名
        if moca_idx is None:
            for idx, moca_name in enumerate(moca_names):
                if pd.notna(moca_name) and moca_name != "nan" and moca_name != "":
                    moca_name_str = str(moca_name).strip()
                    folder_name_str = subject_name.strip()
                    if folder_name_str in moca_name_str or moca_name_str in folder_name_str:
                        moca_idx = idx
                        name_matched_count += 1
                        break
        
        # 策略7: 如果所有策略都失败，尝试按顺序匹配（作为备选方案）
        if moca_idx is None and data_idx < len(moca_data):
            moca_idx = data_idx
            sequential_matched_count += 1
            # 只在调试模式下输出警告（前10个）
            if data_idx < 10:
                print(f"  警告: 文件夹 '{subject_name}' (提取拼音: '{name_part_clean}') 无法通过姓名匹配，使用顺序匹配 (索引 {data_idx})")
        
        if moca_idx is None:
            unmatched_folders.append(subject_name)
            continue  # 跳过无法匹配Moca数据的文件夹
        
        data_idx += 1  # 只有在成功匹配后才增加索引
        
        # 读取Moca分数
        row = moca_data.iloc[moca_idx]
        scores = []
        for col_idx in target_col_indices:
            val = row.iloc[col_idx] if col_idx < len(row) else 0.0
            scores.append(float(val) if pd.notna(val) else 0.0)
        
        # 尝试匹配Smell数据
        smell_idx = None
        smell_feat = None
        if smell_data is not None and smell_name_col is not None and smell_names is not None:
            for idx, smell_name in enumerate(smell_names):
                if pd.notna(smell_name) and smell_name != "nan" and smell_name != "":
                    # 尝试直接匹配或拼音匹配
                    if (subject_name == smell_name or 
                        chinese_to_pinyin(subject_name) == chinese_to_pinyin(smell_name) or
                        (moca_idx < len(moca_names) and smell_name == moca_names[moca_idx])):
                        smell_idx = idx
                        break
            
            if smell_idx is not None and len(smell_cols) > 0:
                s_row = smell_data.iloc[smell_idx]
                s_feat = [safe_to_float(s_row.iloc[i]) if i < len(s_row) else 0.0 for i in smell_cols]
                smell_feat = np.array(s_feat, dtype=np.float32)
                matched_count += 1
            else:
                smell_feat = np.zeros(16, dtype=np.float32)
        else:
            smell_feat = np.zeros(16, dtype=np.float32)
        
        # 添加数据
        X_eye.append(trials_eye)
        X_angle.append(trials_angle)
        y.append(scores)
        subject_names.append(subject_name)
        smell_features_list.append(smell_feat)
    
    print(f"\n数据匹配结果:")
    print(f"  成功匹配的样本数: {len(X_eye)}")
    print(f"  其中通过姓名匹配的: {name_matched_count}")
    print(f"  其中通过顺序匹配的: {sequential_matched_count}")
    print(f"  其中匹配到Smell数据的: {matched_count}")
    if unmatched_folders:
        print(f"  未匹配到Moca数据的文件夹数: {len(unmatched_folders)}")
        if len(unmatched_folders) <= 10:
            print(f"  未匹配的文件夹: {unmatched_folders}")
    
    return X_eye, X_angle, np.array(y, dtype=np.float32), subject_names, np.array(smell_features_list, dtype=np.float32)

# ---- 数据集类（支持嗅觉） ----
class EyeFeatureDatasetWithSmell(Dataset):
    def __init__(self, x1_eye, x1_angle, y, smell_features=None):
        self.x1_eye = x1_eye
        self.x1_angle = x1_angle
        if isinstance(y, np.ndarray):
            self.y = torch.tensor(y, dtype=torch.float32)
        else:
            self.y = torch.tensor(y, dtype=torch.float32)
        
        if smell_features is not None:
            if isinstance(smell_features, np.ndarray):
                self.smell_features = torch.tensor(smell_features, dtype=torch.float32)
            else:
                self.smell_features = torch.tensor(smell_features, dtype=torch.float32)
        else:
            self.smell_features = torch.zeros(len(self.y), 16, dtype=torch.float32)

    def __len__(self):
        return len(self.x1_eye)

    def __getitem__(self, idx):
        return (self.x1_eye[idx], self.x1_angle[idx]), self.y[idx], self.smell_features[idx]

# ---- DataLoader 的 collate：保持 x 为"list-of-trials"，y 做 batch 堆叠，支持smell ----
def collate_trials(batch):
    xs, ys, smells = zip(*batch)
    xs_eye = [x[0] for x in xs]
    xs_angle = [x[1] for x in xs]
    ys = torch.stack(ys, dim=0)
    smells = torch.stack(smells, dim=0)  # (B, 16)
    return (xs_eye, xs_angle, smells), ys

def _move_x_to_device(x, device):
    """把批次 x（包含eye、angle和smell数据）搬到 device。"""
    if isinstance(x, tuple) and len(x) == 3:
        xs_eye, xs_angle, smells = x
        xs_eye_out, xs_angle_out = [], []
        for sample_eye in xs_eye:
            if isinstance(sample_eye, torch.Tensor):
                xs_eye_out.append(sample_eye.to(device))
            else:
                xs_eye_out.append([t.to(device) for t in sample_eye])
        for sample_angle in xs_angle:
            if isinstance(sample_angle, torch.Tensor):
                xs_angle_out.append(sample_angle.to(device))
            else:
                xs_angle_out.append([t.to(device) for t in sample_angle])
        smells_out = smells.to(device)
        return (xs_eye_out, xs_angle_out, smells_out)
    elif isinstance(x, tuple) and len(x) == 2:
        # 兼容旧格式（无smell）
        xs_eye, xs_angle = x
        xs_eye_out, xs_angle_out = [], []
        for sample_eye in xs_eye:
            if isinstance(sample_eye, torch.Tensor):
                xs_eye_out.append(sample_eye.to(device))
            else:
                xs_eye_out.append([t.to(device) for t in sample_eye])
        for sample_angle in xs_angle:
            if isinstance(sample_angle, torch.Tensor):
                xs_angle_out.append(sample_angle.to(device))
            else:
                xs_angle_out.append([t.to(device) for t in sample_angle])
        return (xs_eye_out, xs_angle_out)
    else:
        raise TypeError("Unsupported x type for device move.")

# ===================== 无位置编码的单路径处理器 =====================
class NoPositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

# ===================== 多尺度时间卷积压缩模块 =====================
class MultiScaleTemporalCompressor(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3)
        self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # (B,T,F) -> (B,F,T)
        x = x.transpose(1, 2)
        out = (self.conv3(x) + self.conv5(x) + self.conv7(x)) / 3.0
        res = self.res_conv(x)
        out = self.norm(out + res)
        out = self.act(out)
        out = self.dropout(out)
        # (B,C,T) -> (B,T,C)
        return out.transpose(1, 2)

# ===================== 时间长度自适应对齐 =====================
def time_adaptive_pool(x, target_L):
    # x: (B, T, C) -> (B, C, T) -> pool -> (B, C, target_L) -> (B, target_L, C)
    return F.adaptive_avg_pool1d(x.transpose(1, 2), target_L).transpose(1, 2)

# ===================== 双向交叉注意力块 =====================
class BiCrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead=8, ff_dim=256, dropout=0.2):
        super().__init__()
        # Smell <- Eye
        self.attn_s2e = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                              dropout=dropout, batch_first=True)
        self.ln_s1 = nn.LayerNorm(d_model)
        self.ff_s = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.ln_s2 = nn.LayerNorm(d_model)

        # Eye <- Smell
        self.attn_e2s = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead,
                                              dropout=dropout, batch_first=True)
        self.ln_e1 = nn.LayerNorm(d_model)
        self.ff_e = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.ln_e2 = nn.LayerNorm(d_model)

    def forward(self, smell, eye):
        """
        smell: (B, L_q, d)
        eye:   (B, L_kv, d)
        返回：更新后的 (smell, eye)
        """
        # 并行计算两个方向的交叉注意力
        s_attn, _ = self.attn_s2e(query=smell, key=eye, value=eye)   # Smell从Eye取信息
        e_attn, _ = self.attn_e2s(query=eye,   key=smell, value=smell)  # Eye从Smell取信息

        smell = self.ln_s1(smell + s_attn)
        eye   = self.ln_e1(eye   + e_attn)

        smell = self.ln_s2(smell + self.ff_s(smell))
        eye   = self.ln_e2(eye   + self.ff_e(eye))

        return smell, eye

# ===================== 分类模型（No-PE + 共享骨干 + 嗅觉融合） =====================
class SingleTaskClassificationModel(nn.Module):
    def __init__(self,
                 input_dim_eye=12,
                 input_dim_angle=8,
                 d_model=128,
                 nhead=8,
                 num_layers=2,
                 dropout=0.1,
                 mlp_hidden=256,
                 num_classes=7,
                 target_length=1024,
                 share_backbone=True,
                 num_bica_blocks=2,
                 L_smell=256):
        super().__init__()
        # 使用 model_share_transformer 的共享 backbone：共享 self-attn 与 rolling；禁用位置编码
        from model_share_transformer import SinglePathProcessor
        shared_pos = NoPositionalEncoding() if share_backbone else None
        shared_stack = nn.ModuleList([
            SelfAttentionBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ]) if share_backbone else None
        shared_roll = RollingTransformer(d_model, nhead, dropout) if share_backbone else None

        self.eye_processor = SinglePathProcessor(
            input_dim_eye, d_model, nhead, num_layers, dropout, target_length,
            shared_pos=shared_pos, shared_self_stack=shared_stack, shared_rolling=shared_roll
        )
        self.angle_processor = SinglePathProcessor(
            input_dim_angle, d_model, nhead, num_layers, dropout, target_length,
            shared_pos=shared_pos, shared_self_stack=shared_stack, shared_rolling=shared_roll
        )

        # 在 pair token 维度上做跨模态融合（与 model_share_transformer 对齐）
        self.fuse_ln_q  = nn.LayerNorm(d_model)
        self.fuse_ln_kv = nn.LayerNorm(d_model)
        self.fuse_attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # ---- 嗅觉编码模块（参考BiCA_MLP） ----
        self.smell_dim = 16
        self.L_smell = L_smell
        self.smell_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128), nn.ReLU()
        )
        self.smell_temporal = MultiScaleTemporalCompressor(in_channels=128, out_channels=d_model, stride=1, dropout=dropout)
        self.smell_pe = NoPositionalEncoding()  # 使用无位置编码（与模型整体一致）

        # ---- 双向交叉注意力堆叠（融合嗅觉和眼动特征） ----
        self.bica = nn.ModuleList([
            BiCrossAttentionBlock(d_model=d_model, nhead=nhead, ff_dim=mlp_hidden, dropout=dropout)
            for _ in range(num_bica_blocks)
        ])

        # 融合后的特征维度：39*C（pair tokens）
        flat_in_dim = (40 - 1) * d_model  # 39*C
        self.flat_in_dim = flat_in_dim

        # ---- 分类前再次拼接原始嗅觉向量 ----
        # 如果16维嗅觉向量与展平后的 fused_ea 长度差距过大，则先映射到 fused_ea 的 1/4 长度后再拼接
        self.smell_concat_dim = max(self.smell_dim, flat_in_dim // 39)
        if self.smell_concat_dim == self.smell_dim:
            self.smell_concat_proj = nn.Identity()
        else:
            self.smell_concat_proj = nn.Sequential(
                nn.LayerNorm(self.smell_dim),
                nn.Linear(self.smell_dim, self.smell_concat_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        final_in_dim = flat_in_dim + self.smell_concat_dim
        self.output_mlp = nn.Sequential(
            nn.LayerNorm(final_in_dim),
            nn.Linear(final_in_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden // 2, num_classes)
        )

    def _encode_smell(self, smell):
        """
        编码嗅觉特征
        smell: (B, 16) -> (B, L_smell, d_model)
        """
        if smell.dim() == 2:           # (B, 16) -> (B, 16, 1)
            smell = smell.unsqueeze(-1)
        # (B, 16, 1) -> (B, 1, 16) -> conv -> (B, 128, 16) -> (B, 16, 128)
        s = self.smell_conv(smell.transpose(1, 2)).transpose(1, 2)
        s = self.smell_temporal(s)                      # (B, 16', d_model)
        s = time_adaptive_pool(s, self.L_smell)        # (B, L_smell, d_model)
        s = self.smell_pe(s)
        return s

    def forward(self, x, lengths=None):
        if len(x) == 3:
            xs_eye, xs_angle, smells = x
        else:
            xs_eye, xs_angle = x
            smells = None

        eye_pairs = self.eye_processor(xs_eye, lengths)       # (B,39,C)
        angle_pairs = self.angle_processor(xs_angle, lengths) # (B,39,C)

        # 第一步：单向融合 eye 和 angle（以 angle 为 Q，eye 为 K/V）
        q = self.fuse_ln_q(angle_pairs)
        k = self.fuse_ln_kv(eye_pairs)
        v = self.fuse_ln_kv(eye_pairs)
        fused_ea, _ = self.fuse_attn(q, k, v, need_weights=False)  # (B,39,C)

        # 第二步：如果有嗅觉数据，进行双向交叉注意力融合
        if smells is not None:
            smell_encoded = self._encode_smell(smells)  # (B, L_smell, d_model)
            # 双向交叉注意力融合
            for blk in self.bica:
                smell_encoded, fused_ea = blk(smell_encoded, fused_ea)
            # 使用融合后的眼动特征
            fused = fused_ea
        else:
            fused = fused_ea

        # 展平后，再将输入的原始嗅觉16维向量（必要时映射到 fused_ea 的 1/4 长度）拼接后分类
        fused_flat = fused.reshape(fused.size(0), -1)  # (B, 39*C)

        if smells is None:
            smells = torch.zeros(
                fused_flat.size(0),
                self.smell_dim,
                device=fused_flat.device,
                dtype=fused_flat.dtype
            )
        else:
            smells = smells.to(device=fused_flat.device, dtype=fused_flat.dtype)
            if smells.dim() == 1:
                smells = smells.unsqueeze(0)
            elif smells.dim() > 2:
                smells = smells.view(smells.size(0), -1)

        smell_for_concat = self.smell_concat_proj(smells)
        fused_with_smell = torch.cat([fused_flat, smell_for_concat], dim=1)
        return self.output_mlp(fused_with_smell)

# ===================== 评估（单任务） =====================
def evaluate_single_task(model, loader, device, task_idx):
    model.eval(); all_preds, all_trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            yb = yb.to(device).long(); yb_task = yb[:, task_idx]
            xb = _move_x_to_device(xb, device)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            all_preds.append(pred.cpu()); all_trues.append(yb_task.cpu())
    return torch.cat(all_preds).numpy(), torch.cat(all_trues).numpy()

# ===================== 下抽样策略 =====================
def build_downsampled_indices(y_task: np.ndarray, task_idx: int, rng: np.random.RandomState) -> np.ndarray:
    """根据任务与规则构建下抽样后的样本索引集合（基于训练集顺序的索引）"""
    y_flat = y_task.astype(int)
    indices_per_class = {c: np.where(y_flat == c)[0] for c in np.unique(y_flat)}
    selected = []
    def take(cls, k):
        cls_idx = indices_per_class.get(cls, np.array([], dtype=int))
        if k is None:  # 不变
            selected.extend(cls_idx.tolist())
        else:  # 指定数量（下抽样）
            if len(cls_idx) <= k:
                selected.extend(cls_idx.tolist())
            else:
                chosen = rng.choice(cls_idx, size=k, replace=False)
                selected.extend(chosen.tolist())

    # 规则实现
    if task_idx == 0:  # 视空间总分：1~4不变，5取50（0默认不变）
        for c in [1,2,3,4,0]: take(c, None)
        take(5, 50)
    elif task_idx == 1:  # 命名总分：1~2不变，3取30（0默认不变）
        take(1, None); take(2, None); take(0, None); take(3, 30)
    elif task_idx == 2:  # 注意总分：0,3,4,5不变，6取30（1,2默认不变）
        for c in [0,3,4,5,1,2]: take(c, None)
        take(6, 30)
    elif task_idx == 3:  # 语言总分：0~1不变，2取20，3取20
        take(0, None); take(1, None); take(2, 20); take(3, 20)
    elif task_idx == 5:  # 延迟回忆：不变化
        for c in sorted(indices_per_class.keys()): take(c, None)
    elif task_idx == 6:  # 定向总分：4~5不变，6取8（0~3默认不变）
        for c in [0,1,2,3,4,5]: take(c, None)
        take(6, 8)
    else:  # 其他（抽象总分等）：不变化
        for c in sorted(indices_per_class.keys()): take(c, None)

    selected = np.array(sorted(set(selected)), dtype=int)
    return selected

# ===================== 训练单任务 =====================
def train_single_task(task_idx, X_train, y_train, X_val, y_val,
                       target_len=1024,
                       input_dim_eye=12,
                       input_dim_angle=8,
                       d_model=128,
                       nhead=8,
                       num_layers=2,
                       mlp_hidden=256,
                       dropout=0.1,
                       batch_size=4,
                       num_epochs=180,
                       lr=6e-4,
                       device="cuda:0",
                       patience=15,
                       min_delta=1e-4):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    task_name = SUB_NAMES[task_idx]; num_classes = NUM_CLASSES[task_idx]
    print(f"\n{'='*70}\n开始训练任务 {task_idx}: {task_name} ({num_classes}类, batch_size={batch_size})\n{'='*70}")

    y_train_task = y_train[:, task_idx]
    y_val_task = y_val[:, task_idx]

    # 获取嗅觉数据
    if isinstance(X_train, tuple) and len(X_train) == 3:
        X_train_eye, X_train_angle, X_train_smell = X_train
        X_val_eye, X_val_angle, X_val_smell = X_val
    elif isinstance(X_train, tuple) and len(X_train) == 2:
        X_train_eye, X_train_angle = X_train
        X_val_eye, X_val_angle = X_val
        X_train_smell = X_val_smell = None
    else:
        X_train_eye = X_train_angle = X_train
        X_val_eye = X_val_angle = X_val
        X_train_smell = X_val_smell = None

    # 构建下抽样后的训练子集索引
    rng = np.random.RandomState(42)
    selected_indices = build_downsampled_indices(y_train_task, task_idx, rng)

    base_train_ds = EyeFeatureDatasetWithSmell(X_train_eye, X_train_angle, y_train, X_train_smell)
    train_ds = Subset(base_train_ds, selected_indices.tolist())
    val_ds = EyeFeatureDatasetWithSmell(X_val_eye, X_val_angle, y_val, X_val_smell)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_trials, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_trials, drop_last=False)

    model = SingleTaskClassificationModel(
        input_dim_eye=input_dim_eye,
        input_dim_angle=input_dim_angle,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        mlp_hidden=mlp_hidden,
        num_classes=num_classes,
        target_length=target_len,
        share_backbone=True
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = EpochBasedLambdaLR(optimizer, lr_lambda_fn=lr_lambda)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float('inf'); best_state = None; best_epoch = -1; wait = 0
    train_hist, val_hist = [], []
    print(f"Device: {device} | target_len={target_len}, d_model={d_model}, num_classes={num_classes}")

    for epoch in range(num_epochs):
        model.train(); epoch_loss = 0.0; n = 0
        for xb, yb in train_loader:
            yb = yb.to(device).long(); yb_task = yb[:, task_idx]
            xb = _move_x_to_device(xb, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb_task)
            loss.backward(); optimizer.step()
            bs = yb_task.size(0); epoch_loss += loss.item() * bs; n += bs
        train_loss = epoch_loss / max(1, n)

        model.eval(); val_loss = 0.0; val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                yb = yb.to(device).long(); yb_task = yb[:, task_idx]
                xb = _move_x_to_device(xb, device)
                logits = model(xb)
                loss = loss_fn(logits, yb_task)
                bs = yb_task.size(0); val_loss += loss.item() * bs; val_n += bs
        val_loss = val_loss / max(1, val_n)
        train_hist.append(train_loss); val_hist.append(val_loss)

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:03d}/{num_epochs} | LR={lr_now:.6f} | Train={train_loss:.6f} | Val={val_loss:.6f}")
        scheduler.step(epoch=epoch)  # 传入当前epoch数

        improved = (val_loss + 1e-12) < (best_val - min_delta)
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch; wait = 0
            print(f"  🎯 New best @ epoch {epoch+1}: Val={val_loss:.6f}")
        else:
            wait += 1
            if wait >= patience:
                print(f"  🛑 Early stop (no improve {patience} epochs). Best @ epoch {best_epoch+1}, Val={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    os.makedirs("./model", exist_ok=True)
    model_path = f"./model/task{task_idx}_{SUB_NAMES[task_idx]}_best_npe.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "best_epoch": best_epoch,
        "best_val": best_val,
        "train_hist": train_hist,
        "val_hist": val_hist,
        "task_idx": task_idx,
        "task_name": SUB_NAMES[task_idx],
        "num_classes": num_classes
    }, model_path)
    print(f"✅ 已保存最佳模型 -> {model_path}")
    return model

# ===================== 学习率调度器 =====================
initial_lr = 1e-4
warmup_epochs = 5
total_epochs = 180

class EpochBasedLambdaLR:
    """基于epoch的学习率调度器（包装LambdaLR以支持epoch-based调度）"""
    def __init__(self, optimizer, lr_lambda_fn, last_epoch=-1):
        self.optimizer = optimizer
        self.lr_lambda_fn = lr_lambda_fn
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        lr_multiplier = self.lr_lambda_fn(epoch)
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_multiplier
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

def lr_lambda(current_epoch):
    """学习率调度函数，接收epoch数（从0开始）"""
    if current_epoch < warmup_epochs:
        return 0.1 + 0.9 * (float(current_epoch) / warmup_epochs)
    else:
        progress = float(current_epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

if __name__ == "__main__":
    print("=" * 70)
    print("开始训练 Single Task Classification 模型（No-PE + 定制抽样）")
    print("=" * 70)
    print("\nBatch Size配置:")
    for i in range(7):
        print(f"  任务{i} ({SUB_NAMES[i]}): {BATCH_SIZES[i]} (对应{NUM_CLASSES[i]}类)")
    print("=" * 70)

    print("\n正在加载 eye1_pro_new 数据...")
    X_eye_all, X_angle_all, y_all, subject_names, smell_all = load_eyedata1_new(
        "./Dataset/eye1_pro_new", 
        "./Dataset/Moca_new.xlsx",
        "./Dataset/smellfinal.xls"
    )
    if len(X_eye_all) == 0:
        print("数据加载失败：没有有效数据，退出")
        raise SystemExit(1)
    if len(X_eye_all) != len(smell_all):
        print(f"⚠️ 警告: 数据数量不一致！X_eye_all={len(X_eye_all)}, smell_all={len(smell_all)}")
        # 截断到最小长度
        min_len = min(len(X_eye_all), len(smell_all))
        X_eye_all = X_eye_all[:min_len]
        X_angle_all = X_angle_all[:min_len]
        y_all = y_all[:min_len]
        smell_all = smell_all[:min_len]
        print(f"已截断到最小长度: {min_len}")
    print(f"\n数据形状: y_all.shape = {y_all.shape}, smell_all.shape = {smell_all.shape}")

    print("\n划分训练集和验证集...")
    X_eye_train_list, X_eye_val_list, y_train, y_val = train_test_split(X_eye_all, y_all, test_size=0.2, random_state=42)
    X_angle_train_list, X_angle_val_list, _, _ = train_test_split(X_angle_all, y_all, test_size=0.2, random_state=42)
    smell_train_list, smell_val_list, _, _ = train_test_split(smell_all, y_all, test_size=0.2, random_state=42)

    X_train = (X_eye_train_list, X_angle_train_list, smell_train_list)
    X_val = (X_eye_val_list, X_angle_val_list, smell_val_list)

    print("\n开始训练...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    models = []
    for task_idx in range(7):
        batch_size = BATCH_SIZES[task_idx]
        model = train_single_task(
            task_idx=task_idx,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            target_len=512,
            input_dim_eye=12,
            input_dim_angle=8,
            d_model=8,
            nhead=4,
            num_layers=2,
            mlp_hidden=256,
            dropout=0.1,
            batch_size=batch_size,
            num_epochs=180,
            lr=1e-4,
            device=device,
            patience=15,
            min_delta=1e-4
        )
        models.append(model)
    print("\n所有任务训练完成！\n")


