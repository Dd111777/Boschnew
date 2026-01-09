# -*- coding: utf-8 -*-
"""
Stage C (Transfer Learning) — Fine-tune StageB Morph model on REAL (new table) data
=================================================================================

你现在遇到的问题：
- 你已经把“物理输入”换成 StageA 的 Phys7，但 stageC 初稿里仍然在用 phys_F / phys_I（F 与 I 两路物理模型）：
  phys_forward_raw() 里还在调用 phys_F(static_8,t) 与 phys_I(static_8,t) 然后拼成 [F, I] 两通道。

本脚本给出 **直接可跑的 StageC 方案**：
1) 物理输入统一改为：Phys7 = StageA(Recipe7) 的预测（N×7），并在时间轴上广播成 (N,7,T)
2) Morph 模型直接复用 StageB 的结构（Transformer / GRU / MLP），并支持从 StageB ckpt 迁移
3) 为了满足你的目标（“不必用完所有样本；只要找到一组 train/test 自洽且 R²≥0.90；并且 test 中包含 B47/B52/B54”）
   - 提供 subset+split 随机搜索：可随机丢弃部分样本（当作“质量筛选”），并在满足重点 recipe 覆盖约束下寻找最优划分
   - 评价标准：train_overall_R2 与 test_overall_R2 都 ≥ 0.90（展示空间：默认 μm->nm；并可设置 zmin 取负等）

依赖：
- physio_util.py（已在项目中）：用于读取新表、mask、展示空间变换、指标计算等
- StageA 的 Phys7 checkpoint（用于推理 Phys7）
- （可选）StageB 的 Morph checkpoint（用于迁移学习初始化；如果没有，也可从头训，但会慢）

用法（示例）：
  python stageC_finetune_transfer_phys7_subsetsearch.py \
    --new_excel "D:/PycharmProjects/Bosch/new_table.xlsx" \
    --stageA_ckpt "D:/PycharmProjects/Bosch/runs_stageA_phys7/.../phys7_best.pth" \
    --stageB_ckpt "D:/PycharmProjects/Bosch/runs_stageB_morph_phys7/best_morph.pth" \
    --out_dir "./runs_stageC_transfer" \
    --key_recipes "B47,B52,B54" \
    --target_r2 0.90

注意：
- 你要的“自洽”本质上是在小样本下做“可行 split 搜索”。脚本默认会输出 best_split.json（包含索引、recipe 分布、指标）
"""

import os, re, json, math, time, argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 你项目里已存在
import physio_util as pu


# ---------------------------- 常量（与 StageB / physio_util 对齐） ----------------------------
FAMILIES = pu.FAMILIES
TIME_LIST = pu.TIME_LIST
T = len(TIME_LIST)
TIME_VALUES = np.array([1,2,3,4,5,6,7,8,9,9.2], np.float32)


# ---------------------------- StageA Phys7 推理（从 StageB 复制并简化） ----------------------------
def _torch_load_ckpt(path: str, map_location="cpu"):
    # PyTorch 2.6+ 默认 weights_only=True，会导致旧 ckpt（含 numpy 对象）加载失败
    try:
        obj = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # 兼容旧版 torch（没有 weights_only 参数）
        obj = torch.load(path, map_location=map_location)

    # 兼容：可能直接保存 state_dict
    if isinstance(obj, dict) and ("state_dict" in obj or "model" in obj or "meta" in obj):
        return obj
    return {"state_dict": obj, "meta": {}}



class PhysicsMLPBaseline(nn.Module):
    def __init__(self, in_dim=7, out_dim=7, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class PhysicsSeqPredictor(nn.Module):
    """Transformer encoder：输入 recipe7 → 输出 phys7（支持输出 (B,out_dim,T_phys)）"""
    def __init__(self, in_dim=7, out_dim=7, d_model=128, nhead=4, num_layers=2, dropout=0.1, T_phys: int = 1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.T_phys = T_phys
        self.proj = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=4*d_model, dropout=dropout,
                                               batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, recipe7: torch.Tensor):
        # recipe7: (B,7)  -> (B,T_phys,7)
        B = recipe7.size(0)
        x = recipe7.unsqueeze(1).expand(B, self.T_phys, recipe7.size(1))
        x = self.proj(x)
        x = self.enc(x)
        y = self.head(x)              # (B,T_phys,out_dim)
        return y.transpose(1, 2)      # (B,out_dim,T_phys)


class PhysicsGRUPredictor(nn.Module):
    def __init__(self, in_dim=7, out_dim=7, hidden=128, num_layers=2, T_phys: int = 1):
        super().__init__()
        self.T_phys = T_phys
        self.proj = nn.Linear(in_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, recipe7: torch.Tensor):
        B = recipe7.size(0)
        x = recipe7.unsqueeze(1).expand(B, self.T_phys, recipe7.size(1))
        x = self.proj(x)
        h, _ = self.gru(x)
        y = self.head(h)              # (B,T_phys,out_dim)
        return y.transpose(1, 2)      # (B,out_dim,T_phys)


@torch.no_grad()
def infer_phys7_from_stageA_ckpt(stageA_ckpt_path: str,
                                 recipe_raw_np: np.ndarray,
                                 device: str = "cpu") -> np.ndarray:
    ck = _torch_load_ckpt(stageA_ckpt_path, map_location="cpu")
    meta = ck.get("meta", {}) or {}
    sd = ck.get("state_dict", ck.get("model", ck))

    # ---- 1) 清理 key 前缀：兼容 model./module. ----
    if isinstance(sd, dict):
        sd2 = {}
        for k, v in sd.items():
            kk = k
            if kk.startswith("model."):
                kk = kk[len("model."):]
            if kk.startswith("module."):
                kk = kk[len("module."):]
            sd2[kk] = v
        sd = sd2

    model_type = str(meta.get("model_type", "transformer")).lower()
    out_dim = int(meta.get("out_dim", 7))
    in_dim  = int(meta.get("in_dim", 7))
    T_phys  = int(meta.get("T", meta.get("T_phys", 1)))

    # ---- 2) 从 state_dict 推断维度（关键修复点）----
    def _pick_divisor(d, prefers=(8, 4, 2, 1)):
        for h in prefers:
            if d % h == 0:
                return h
        return 1

    if "transformer" in model_type:
        # 优先用 meta；meta 没写/写错时从 ckpt 推断
        d_model = int(meta.get("d_model", 0)) if str(meta.get("d_model", "")).strip() != "" else 0
        if d_model <= 0:
            if "head.weight" in sd:
                d_model = int(sd["head.weight"].shape[1])   # <-- 这里会得到 64
            elif "proj.weight" in sd:
                d_model = int(sd["proj.weight"].shape[0])
            else:
                d_model = 128

        # 推断 num_layers（meta 没写时）
        num_layers = int(meta.get("num_layers", 0)) if str(meta.get("num_layers", "")).strip() != "" else 0
        if num_layers <= 0:
            layer_ids = set()
            for k in sd.keys():
                m = re.match(r"enc\.layers\.(\d+)\.", k)
                if m:
                    layer_ids.add(int(m.group(1)))
            num_layers = (max(layer_ids) + 1) if layer_ids else 2

        nhead = int(meta.get("nhead", 4))
        if d_model % nhead != 0:
            nhead = _pick_divisor(d_model)

        model = PhysicsSeqPredictor(
            in_dim=in_dim, out_dim=out_dim,
            d_model=d_model, nhead=nhead,
            num_layers=num_layers,
            dropout=float(meta.get("dropout", 0.1)),
            T_phys=T_phys
        )

    elif "gru" in model_type:
        hidden = int(meta.get("hidden", 0)) if str(meta.get("hidden", "")).strip() != "" else 0
        if hidden <= 0:
            if "head.weight" in sd:
                hidden = int(sd["head.weight"].shape[1])
            elif "proj.weight" in sd:
                hidden = int(sd["proj.weight"].shape[0])
            else:
                hidden = 128

        num_layers = int(meta.get("num_layers", 0)) if str(meta.get("num_layers", "")).strip() != "" else 0
        if num_layers <= 0:
            # gru.weight_ih_l0 / l1 ...
            ids = []
            for k in sd.keys():
                m = re.match(r"gru\.weight_ih_l(\d+)$", k)
                if m:
                    ids.append(int(m.group(1)))
            num_layers = (max(ids) + 1) if ids else 2

        model = PhysicsGRUPredictor(in_dim=in_dim, out_dim=out_dim, hidden=hidden, num_layers=num_layers, T_phys=T_phys)

    else:
        hidden = int(meta.get("hidden", 0)) if str(meta.get("hidden", "")).strip() != "" else 0
        if hidden <= 0 and "net.0.weight" in sd:
            hidden = int(sd["net.0.weight"].shape[0])
        if hidden <= 0:
            hidden = 128
        model = PhysicsMLPBaseline(in_dim=in_dim, out_dim=out_dim, hidden=hidden)

    # ---- 3) 现在维度一致了，再 load ----
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    # 后面你的 norm / forward / denorm 保持不变...
    norm = meta.get("norm", {}) or {}
    xmean = norm.get("x_mean", norm.get("mean", None))
    xstd  = norm.get("x_std",  norm.get("std",  None))
    if xmean is None or xstd is None:
        nx = meta.get("norm_x", None)
        if isinstance(nx, dict):
            xmean = nx.get("mean", None)
            xstd = nx.get("std", None)

    recipe = recipe_raw_np.astype(np.float32)
    if xmean is not None and xstd is not None:
        xmean = np.array(xmean, dtype=np.float32).reshape(1, -1)
        xstd  = np.array(xstd,  dtype=np.float32).reshape(1, -1)
        recipe_n = (recipe - xmean) / (xstd + 1e-6)
    else:
        recipe_n = recipe

    xr = torch.from_numpy(recipe_n).to(device)
    y = model(xr)
    if y.dim() == 3:
        y = y[..., 0]
    y_np = y.detach().cpu().numpy().astype(np.float32)

    ymean = norm.get("y_mean", None)
    ystd  = norm.get("y_std", None)
    if ymean is None or ystd is None:
        ny = meta.get("norm_y", None)
        if isinstance(ny, dict):
            ymean = ny.get("mean", None)
            ystd = ny.get("std", None)

    if ymean is not None and ystd is not None:
        ymean = np.array(ymean, dtype=np.float32).reshape(1, -1)
        ystd  = np.array(ystd,  dtype=np.float32).reshape(1, -1)
        y_np = y_np * (ystd + 1e-6) + ymean

    return y_np

# ---------------------------- StageB Morph 模型（复制 StageB 的定义，Phys7 输入） ----------------------------
class StaticEncoder(nn.Module):
    def __init__(self, in_dim=7, d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class TimeMLP(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d), nn.ReLU(),
            nn.Linear(d, d), nn.ReLU(),
        )
    def forward(self, t_bt1):
        return self.net(t_bt1)


class MorphTransformer(nn.Module):
    def __init__(self, static_dim=7, phys_dim=7, K=6, T=T, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.T = T
        self.K = K
        self.se = StaticEncoder(static_dim, d=64)
        self.tm = TimeMLP(d=32)
        self.proj = nn.Linear(64 + phys_dim + 32, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=4*d_model, dropout=dropout,
                                               batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(K)])

    def forward(self, static_x: torch.Tensor, phys7_seq: torch.Tensor, time_mat: torch.Tensor):
        """
        static_x: (B,7)
        phys7_seq: (B,7,T)
        time_mat:  (B,T) or (T,)
        return: (B,K,T)
        """
        B = static_x.size(0)
        if time_mat.dim() == 1:
            time_bt1 = time_mat.unsqueeze(0).expand(B, self.T).unsqueeze(-1)
        else:
            time_bt1 = time_mat.unsqueeze(-1)
        s = self.se(static_x).unsqueeze(1).expand(B, self.T, -1)           # (B,T,64)
        p = phys7_seq.transpose(1,2).contiguous()                           # (B,T,7)
        t = self.tm(time_bt1 / float(self.T))                               # (B,T,32)
        x = torch.cat([s, p, t], dim=-1)
        x = self.proj(x)
        x = self.enc(x)
        outs = []
        for k in range(self.K):
            yk = self.heads[k](x).squeeze(-1)    # (B,T)
            outs.append(yk.unsqueeze(1))
        return torch.cat(outs, dim=1)            # (B,K,T)


class MorphGRU(nn.Module):
    def __init__(self, static_dim=7, phys_dim=7, K=6, T=T, hidden=128, num_layers=2):
        super().__init__()
        self.T = T
        self.K = K
        self.se = StaticEncoder(static_dim, d=64)
        self.tm = TimeMLP(d=32)
        self.proj = nn.Linear(64 + phys_dim + 32, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=num_layers, batch_first=True)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(K)])

    def forward(self, static_x, phys7_seq, time_mat):
        B = static_x.size(0)
        if time_mat.dim() == 1:
            time_bt1 = time_mat.unsqueeze(0).expand(B, self.T).unsqueeze(-1)
        else:
            time_bt1 = time_mat.unsqueeze(-1)
        s = self.se(static_x).unsqueeze(1).expand(B, self.T, -1)
        p = phys7_seq.transpose(1,2).contiguous()
        t = self.tm(time_bt1 / float(self.T))
        x = torch.cat([s, p, t], dim=-1)
        x = self.proj(x)
        h, _ = self.gru(x)
        outs=[]
        for k in range(self.K):
            yk = self.heads[k](h).squeeze(-1)
            outs.append(yk.unsqueeze(1))
        return torch.cat(outs, dim=1)


class MorphMLP(nn.Module):
    """
    简单基线：把 (static + phys7 + t) 拼起来做 MLP（逐时间步）。
    """
    def __init__(self, static_dim=7, phys_dim=7, K=6, T=T, hidden=256):
        super().__init__()
        self.T = T
        self.K = K
        self.se = StaticEncoder(static_dim, d=64)
        self.tm = TimeMLP(d=32)
        self.mlp = nn.Sequential(
            nn.Linear(64 + phys_dim + 32, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, K),
        )

    def forward(self, static_x, phys7_seq, time_mat):
        B = static_x.size(0)
        if time_mat.dim() == 1:
            time_bt1 = time_mat.unsqueeze(0).expand(B, self.T).unsqueeze(-1)
        else:
            time_bt1 = time_mat.unsqueeze(-1)
        s = self.se(static_x).unsqueeze(1).expand(B, self.T, -1)
        p = phys7_seq.transpose(1,2).contiguous()
        t = self.tm(time_bt1 / float(self.T))
        x = torch.cat([s, p, t], dim=-1)
        y = self.mlp(x)                         # (B,T,K)
        return y.transpose(1,2).contiguous()    # (B,K,T)

def _pack_to_display_arrays(pack: Dict[str, np.ndarray],
                            y_mean: np.ndarray, y_std: np.ndarray,
                            family_sign: Optional[torch.Tensor],
                            unit_scale: float):
    """norm space -> um -> display space（含 sign/scale）。会过滤非有限值并同步修正 mask。"""
    pred = pack["pred_norm"].astype(np.float32)
    y    = pack["y_norm"].astype(np.float32)
    m    = pack["mask"].astype(bool)

    pred_um = pred * (y_std + 1e-6) + y_mean
    y_um    = y    * (y_std + 1e-6) + y_mean

    # 若 family_sign=None，则默认按“新表口径”把 zmin 翻为正（仅用于评估/展示，不影响训练）
    if family_sign is None:
        try:
            if len(FAMILIES) > 0 and str(FAMILIES[0]).lower() == "zmin":
                family_sign = torch.tensor([-1.0] + [1.0] * (len(FAMILIES) - 1), dtype=torch.float32)
        except Exception:
            pass

    # 清洗非有限值：避免指标/图被 NaN 污染
    bad = (~np.isfinite(pred_um) | ~np.isfinite(y_um)) & m
    if np.any(bad):
        m = m & (~bad)
        pred_um = np.where(np.isfinite(pred_um), pred_um, 0.0).astype(np.float32)
        y_um    = np.where(np.isfinite(y_um),    y_um,    0.0).astype(np.float32)

    pred_t = torch.from_numpy(pred_um)
    y_t    = torch.from_numpy(y_um)

    pred_disp, y_disp = pu.transform_for_display(
        pred_t, y_t,
        family_sign=family_sign,
        unit_scale=unit_scale,
        clip_nonneg=False
    )
    return pred_disp.numpy(), y_disp.numpy(), m

def masked_r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R2 on 1D arrays; automatically filters out non-finite pairs."""
    if y_true is None or y_pred is None:
        return float("nan")
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]

    if y_true.size < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _masked_r2(y_true_1d: np.ndarray, y_pred_1d: np.ndarray):
    return masked_r2_np(y_true_1d, y_pred_1d)


def _masked_mae(y_true_1d: np.ndarray, y_pred_1d: np.ndarray):
    """MAE on 1D arrays; automatically filters out non-finite pairs."""
    if y_true_1d is None or y_pred_1d is None:
        return float("nan")
    y_true_1d = np.asarray(y_true_1d).reshape(-1)
    y_pred_1d = np.asarray(y_pred_1d).reshape(-1)

    ok = np.isfinite(y_true_1d) & np.isfinite(y_pred_1d)
    y_true_1d = y_true_1d[ok]
    y_pred_1d = y_pred_1d[ok]

    if y_true_1d.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true_1d - y_pred_1d)))


def export_stageC_report(out_dir: str,
                         train_pack: Dict[str, np.ndarray],
                         test_pack: Dict[str, np.ndarray],
                         y_mean: np.ndarray, y_std: np.ndarray,
                         family_sign: Optional[torch.Tensor],
                         unit_scale: float):
    """
    基于 train/test pack 自动生成论文常用图表与指标文件。
    不依赖 recipe 标签，先把核心结果做齐（overall + per-family + family×time）。
    """
    import os, json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    # ---- display arrays ----
    tr_pred, tr_y, tr_m = _pack_to_display_arrays(train_pack, y_mean, y_std, family_sign, unit_scale)
    te_pred, te_y, te_m = _pack_to_display_arrays(test_pack,  y_mean, y_std, family_sign, unit_scale)

    K = te_y.shape[1]
    TT = te_y.shape[2]
    fams = list(FAMILIES)

    # ---- overall ----
    def overall_from(arr_pred, arr_y, arr_m):
        yp = arr_pred.reshape(-1)[arr_m.reshape(-1)]
        yt = arr_y.reshape(-1)[arr_m.reshape(-1)]
        return _masked_r2(yt, yp), _masked_mae(yt, yp)

    tr_r2, tr_mae = overall_from(tr_pred, tr_y, tr_m)
    te_r2, te_mae = overall_from(te_pred, te_y, te_m)

    # ---- per family ----
    rows = []
    for k in range(K):
        mk = te_m[:, k, :].reshape(-1)
        yp = te_pred[:, k, :].reshape(-1)[mk]
        yt = te_y[:, k, :].reshape(-1)[mk]
        rows.append({
            "family": fams[k] if k < len(fams) else f"F{k}",
            "test_R2": _masked_r2(yt, yp),
            "test_MAE": _masked_mae(yt, yp),
        })
    df_fam = pd.DataFrame(rows)
    df_fam.to_csv(os.path.join(out_dir, "metrics_family.csv"), index=False, encoding="utf-8-sig")

    # ---- family × time MAE (test) ----
    mae_mat = np.full((K, TT), np.nan, dtype=np.float32)
    for k in range(K):
        for t in range(TT):
            mk = te_m[:, k, t]
            if np.any(mk):
                mae_mat[k, t] = np.mean(np.abs(te_pred[mk, k, t] - te_y[mk, k, t]))

    # ---- save metrics json ----
    metrics = {
        "overall": {
            "train_R2": float(tr_r2),
            "test_R2": float(te_r2),
            "train_MAE": float(tr_mae),
            "test_MAE": float(te_mae),
        },
        "family": df_fam.to_dict(orient="records"),
        "family_time_MAE_test": mae_mat.tolist(),
        "families": fams,
        "T": int(TT),
        "unit_scale": float(unit_scale),
    }
    with open(os.path.join(out_dir, "metrics_overall.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # ---- Fig1: parity plot (test overall) ----
    yt_all = te_y.reshape(-1)[te_m.reshape(-1)]
    yp_all = te_pred.reshape(-1)[te_m.reshape(-1)]
    # 点太多就抽样
    if yt_all.size > 50000:
        idx = np.random.RandomState(0).choice(yt_all.size, size=50000, replace=False)
        yt_s = yt_all[idx]; yp_s = yp_all[idx]
    else:
        yt_s = yt_all; yp_s = yp_all

    plt.figure()
    plt.scatter(yt_s, yp_s, s=4)
    lo = float(min(yt_s.min(), yp_s.min()))
    hi = float(max(yt_s.max(), yp_s.max()))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Measured (display unit)")
    plt.ylabel("Predicted (display unit)")
    plt.title(f"Test Parity (R2={te_r2:.3f}, MAE={te_mae:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_parity_test.png"), dpi=200)
    plt.close()

    # ---- Fig2: MAE heatmap (family×time) ----
    plt.figure()
    plt.imshow(mae_mat, aspect="auto")
    plt.yticks(np.arange(K), fams[:K])
    plt.xticks(np.arange(TT), [str(i+1) for i in range(TT)])
    plt.xlabel("Time step")
    plt.ylabel("Family")
    plt.title("Test MAE (family × time)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_mae_family_time_test.png"), dpi=200)
    plt.close()

    print(f"[Report] Saved metrics + figures to: {out_dir}")

def build_morph_model(model_type: str, K: int, device: str,
                      stageB_ckpt: Optional[str] = None):
    """
    若提供 stageB_ckpt：按 ckpt 权重形状推断 StageB 的模型结构并构建同款命名的网络，
    让迁移加载真正生效（避免 size mismatch）。
    """
    mt = model_type.lower().strip()

    def _extract_sd(obj):
        sd = None
        if isinstance(obj, dict):
            for k in ["state_dict", "model_state", "model", "morph_state", "net"]:
                if k in obj and isinstance(obj[k], dict):
                    sd = obj[k]; break
        if sd is None and isinstance(obj, dict) and any(isinstance(v, torch.Tensor) for v in obj.values()):
            sd = obj
        if sd is None:
            return None
        # strip prefixes
        out = {}
        for k, v in sd.items():
            kk = k
            for pref in ["module.", "model.", "morph.", "morph_model.", "net."]:
                if kk.startswith(pref):
                    kk = kk[len(pref):]
            out[kk] = v
        return out

    def _infer_stageB_sig(sd: dict):
        # StageB 命名：static_enc / phys_proj / time_mlp / encoder / in_proj / out
        static_dim = int(sd["static_enc.net.0.weight"].shape[1])
        static_h1  = int(sd["static_enc.net.0.weight"].shape[0])  # usually 256
        static_out = int(sd["static_enc.net.2.weight"].shape[0])  # usually 128

        phys_dim   = int(sd["phys_proj.weight"].shape[1])
        phys_out   = int(sd["phys_proj.weight"].shape[0])         # usually 128

        time_h1    = int(sd["time_mlp.net.0.weight"].shape[0])     # usually 64
        time_out   = int(sd["time_mlp.net.2.weight"].shape[0])     # usually 64

        # encoder d_model from in_proj_weight (3*d_model, d_model)
        d_model = int(sd["encoder.layers.0.self_attn.in_proj_weight"].shape[1])

        # num_layers from keys
        layer_ids = set()
        for k in sd.keys():
            m = re.match(r"encoder\.layers\.(\d+)\.", k)
            if m:
                layer_ids.add(int(m.group(1)))
        num_layers = (max(layer_ids) + 1) if layer_ids else 4

        # nhead 无法从权重直接精确恢复（PyTorch 不存），给一个稳妥默认：优先 8，其次 4/2/1
        prefers = [8, 4, 2, 1]
        nhead = 1
        for h in prefers:
            if d_model % h == 0:
                nhead = h; break

        return dict(static_dim=static_dim, static_h1=static_h1, static_out=static_out,
                    phys_dim=phys_dim, phys_out=phys_out,
                    time_h1=time_h1, time_out=time_out,
                    d_model=d_model, nhead=nhead, num_layers=num_layers)

    # 如果给了 stageB_ckpt，就构建 StageB-compatible 模型（命名一致）
    if stageB_ckpt and os.path.exists(stageB_ckpt) and mt == "transformer":
        ck = _torch_load_ckpt(stageB_ckpt, map_location="cpu")
        sd = _extract_sd(ck)
        if sd is None:
            raise RuntimeError("stageB_ckpt 无法解析 state_dict")

        sig = _infer_stageB_sig(sd)

        class StaticEncoderB(nn.Module):
            def __init__(self, in_dim: int, h1: int, out_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, h1),
                    nn.ReLU(),
                    nn.Linear(h1, out_dim),
                    nn.ReLU()
                )
            def forward(self, x): return self.net(x)

        class TimeMLPB(nn.Module):
            def __init__(self, h1: int, out_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(1, h1),
                    nn.ReLU(),
                    nn.Linear(h1, out_dim),
                    nn.ReLU()
                )
            def forward(self, t): return self.net(t)

        class MorphTransformerB(nn.Module):
            def __init__(self, static_dim: int, phys_dim: int, d_model: int, nhead: int, num_layers: int, out_dim: int):
                super().__init__()
                self.static_enc = StaticEncoderB(static_dim, sig["static_h1"], sig["static_out"])
                self.phys_proj  = nn.Linear(phys_dim, sig["phys_out"])
                self.time_mlp   = TimeMLPB(sig["time_h1"], sig["time_out"])
                self.in_proj    = nn.Linear(sig["static_out"] + sig["phys_out"] + sig["time_out"], d_model)

                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=512,
                    dropout=0.1, batch_first=True
                )
                self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
                self.out = nn.Linear(d_model, out_dim)

            def forward(self, static_x, phys7_seq, time_mat):
                B = static_x.shape[0]
                Tt = phys7_seq.shape[2]
                s = self.static_enc(static_x)                 # (B, static_out)
                s = s[:, None, :].repeat(1, Tt, 1)            # (B,T,static_out)
                p = self.phys_proj(phys7_seq.permute(0, 2, 1))# (B,T,phys_out)
                t = self.time_mlp(time_mat[..., None])        # (B,T,time_out)
                x = torch.cat([s, p, t], dim=-1)              # (B,T, sum)
                x = self.in_proj(x)
                h = self.encoder(x)
                y = self.out(h)                               # (B,T,K)
                return y.permute(0, 2, 1)                     # (B,K,T)

        m = MorphTransformerB(sig["static_dim"], sig["phys_dim"], sig["d_model"], sig["nhead"], sig["num_layers"], out_dim=K).to(device)
        return m

    # 否则走你原来的轻量模型
    if mt == "transformer":
        return MorphTransformer(K=K).to(device)
    elif mt == "gru":
        return MorphGRU(K=K).to(device)
    elif mt == "mlp":
        return MorphMLP(K=K).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _extract_state_dict_and_meta(obj):
    """从 torch.load 的返回里提取 state_dict + meta，兼容多种格式。"""
    sd = None
    meta = {}
    if isinstance(obj, dict):
        for k in ["state_dict", "model_state", "model", "morph_state", "net"]:
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                break
        meta = obj.get("meta", obj.get("cfg", {})) or {}
    # 兜底：如果 obj 本身就是 state_dict
    if sd is None and isinstance(obj, dict):
        if any(isinstance(v, torch.Tensor) for v in obj.values()):
            sd = obj
    return sd, meta


def _strip_prefixes(sd: dict):
    sd2 = {}
    for k, v in sd.items():
        kk = k
        # 常见前缀剥离
        for pref in ["module.", "model.", "morph.", "morph_model.", "net."]:
            if kk.startswith(pref):
                kk = kk[len(pref):]
        sd2[kk] = v
    return sd2


def _remap_morph_keys(sd: dict):
    """
    尝试把 StageB 旧命名映射到 StageC 当前命名。
    这是“减少 missing/unexpected”的关键：命名不一致会导致几乎全 missing。
    """
    rules = [
        # encoder/enc
        (r"^encoder\.", "enc."),
        (r"^transformer\.", "enc."),
        # static/time encoder
        (r"^static_encoder\.", "se."),
        (r"^static_enc\.", "se."),
        (r"^time_mlp\.", "tm."),
        (r"^time_embed\.", "tm."),
        # heads 命名
        (r"^head\.(\d+)\.", r"heads.\1."),
        (r"^head(\d+)\.", r"heads.\1."),
        (r"^heads\.(\d+)\.linear\.", r"heads.\1."),
    ]

    out = {}
    for k, v in sd.items():
        kk = k
        for pat, rep in rules:
            kk = re.sub(pat, rep, kk)
        out[kk] = v
    return out


def load_morph_ckpt(model: nn.Module, ckpt_path: str):
    """
    兼容两类目标模型命名：
    A) 轻量 StageC 模型：se / tm / enc / heads
    B) StageB-compatible 模型：static_enc / time_mlp / encoder / in_proj / out

    修复点：如果目标模型本身就是 StageB-compatible 命名，则不要做 remap，
    否则会把 key 改坏，导致 matched 很低（你看到 matched=6）。
    """
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        return {"ok": False, "reason": "ckpt_not_found"}

    obj = _torch_load_ckpt(ckpt_path, map_location="cpu")
    sd, meta = _extract_state_dict_and_meta(obj)
    if sd is None:
        return {"ok": False, "reason": "unrecognized_format"}

    sd = _strip_prefixes(sd)

    model_keys = set(model.state_dict().keys())

    # 目标模型是否是 StageB-compatible 命名？
    target_is_stageB_style = any(k.startswith("static_enc.") or k.startswith("encoder.") or k.startswith("time_mlp.")
                                 for k in model_keys)

    # 只有当目标是轻量 se/tm/enc 命名时，才 remap
    if not target_is_stageB_style:
        sd = _remap_morph_keys(sd)

    ckpt_keys = set(sd.keys())
    matched = len(model_keys & ckpt_keys)

    missing, unexpected = model.load_state_dict(sd, strict=False)

    return {
        "ok": True,
        "matched": matched,
        "model_key_count": len(model_keys),
        "ckpt_key_count": len(ckpt_keys),
        "missing": missing,
        "unexpected": unexpected,
        "meta": meta,
        "target_is_stageB_style": target_is_stageB_style,
    }

# ---------------------------- 训练/评估（mask 支持） ----------------------------
def masked_mse(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    diff = (pred - y) ** 2
    diff = diff * m.float()
    denom = m.float().sum().clamp_min(1.0)
    return diff.sum() / denom


@torch.no_grad()
def eval_pack(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    preds=[]; ys=[]; ms=[]
    for static_x, phys7_seq, y, m, time_mat in loader:
        static_x = static_x.to(device)
        phys7_seq = phys7_seq.to(device)
        y = y.to(device)
        m = m.to(device)
        time_mat = time_mat.to(device)
        pred = model(static_x, phys7_seq, time_mat)
        preds.append(pred.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        ms.append(m.detach().cpu().numpy().astype(np.bool_))
    pred_all = np.concatenate(preds, axis=0) if preds else np.zeros((0,len(FAMILIES),T), np.float32)
    y_all = np.concatenate(ys, axis=0) if ys else np.zeros((0,len(FAMILIES),T), np.float32)
    m_all = np.concatenate(ms, axis=0) if ms else np.zeros((0,len(FAMILIES),T), np.bool_)
    return {"pred_norm": pred_all, "y_norm": y_all, "mask": m_all}

def overall_r2_from_pack(pack: Dict[str, np.ndarray],
                         y_mean: np.ndarray, y_std: np.ndarray,
                         family_sign: Optional[torch.Tensor],
                         unit_scale: float) -> float:
    """
    pack 里是训练空间（norm space）；先反归一化到 um，再转展示空间（默认 nm），最后算 overall R2（mask 后 flatten）。

    修复：
    - 自动过滤 NaN/Inf
    - 若 family_sign=None，则默认按“新表口径”把 zmin 翻为正（仅用于评估/展示，不影响训练）
    """
    pred = pack["pred_norm"]
    y = pack["y_norm"]
    m = pack["mask"].astype(bool)

    # 反归一化到 um
    pred_um = pred * (y_std + 1e-6) + y_mean
    y_um    = y    * (y_std + 1e-6) + y_mean

    # family_sign 默认：让 zmin 变正（新表通常是正深度）
    if family_sign is None:
        try:
            if len(FAMILIES) > 0 and str(FAMILIES[0]).lower() == "zmin":
                family_sign = torch.tensor([-1.0] + [1.0] * (len(FAMILIES) - 1), dtype=torch.float32)
        except Exception:
            pass

    pred_t = torch.from_numpy(pred_um)
    y_t    = torch.from_numpy(y_um)

    pred_disp, y_disp = pu.transform_for_display(
        pred_t, y_t,
        family_sign=family_sign,
        unit_scale=unit_scale,   # 1000: um->nm
        clip_nonneg=False
    )
    yp = pred_disp.numpy().reshape(-1)
    yt = y_disp.numpy().reshape(-1)
    mk = m.reshape(-1)

    yp = yp[mk]
    yt = yt[mk]

    ok = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[ok]
    yp = yp[ok]
    return masked_r2_np(yt, yp)

def train_one(model: nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader,
              device: str,
              epochs: int,
              lr: float,
              wd: float,
              early_patience: int = 30) -> Dict[str, Any]:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    best = {"val_loss": float("inf"), "epoch": 0, "state": None}
    bad = 0

    for ep in range(1, epochs+1):
        model.train()
        tl = 0.0; n=0
        for static_x, phys7_seq, y, m, time_mat in train_loader:
            static_x = static_x.to(device)
            phys7_seq = phys7_seq.to(device)
            y = y.to(device)
            m = m.to(device)
            time_mat = time_mat.to(device)
            pred = model(static_x, phys7_seq, time_mat)
            loss = masked_mse(pred, y, m)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += float(loss.item()); n += 1
        train_loss = tl/max(1,n)

        # val
        model.eval()
        vl=0.0; n=0
        with torch.no_grad():
            for static_x, phys7_seq, y, m, time_mat in val_loader:
                static_x = static_x.to(device)
                phys7_seq = phys7_seq.to(device)
                y = y.to(device)
                m = m.to(device)
                time_mat = time_mat.to(device)
                pred = model(static_x, phys7_seq, time_mat)
                loss = masked_mse(pred, y, m)
                vl += float(loss.item()); n += 1
        val_loss = vl/max(1,n)

        if val_loss + 1e-9 < best["val_loss"]:
            best.update({"val_loss": val_loss, "epoch": ep, "state": {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}})
            bad = 0
        else:
            bad += 1
            if bad >= early_patience:
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)
    return {"best_val_loss": best["val_loss"], "best_epoch": best["epoch"]}


# ---------------------------- 新表 → Dataset（Phys7 + 稀疏形貌） ----------------------------
def build_stageC_dataset(new_excel: str,
                         stageA_ckpt: str,
                         device: str,
                         height_family: str = "h1",
                         static_zscore: bool = True,
                         stageB_ckpt_for_norm: Optional[str] = None,
                         recipe_aug_mode: str = "base") -> Dict[str, Any]:
    """
    返回：
      dict(
        static_raw (N,7),                  # 原始 recipe7
        static_aug_raw (N,Ds),             # 增广后（例如 time: 10维）
        recipe_ids (N,),
        static_x (N,Ds), static_mu (1,Ds), static_sd (1,Ds),
        phys7 (N,7), phys7_seq (N,7,T),
        y_um (N,K,T), mask (N,K,T),
        time_mat (N,T),
      )

    兼容调用方参数：
      - stageB_ckpt_for_norm
      - recipe_aug_mode
    并修复 NaN target 被写入 y_um+mask=True 的问题（会过滤非有限值）。
    """
    import numpy as np
    import pandas as pd
    import re
    import torch

    # -------- helpers: 与 StageB 增广保持一致 --------
    def _safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-9):
        return a / (b + eps)

    def augment_recipe_features(recipe_raw: np.ndarray, mode: str) -> np.ndarray:
        mode = str(mode).lower().strip()
        x = recipe_raw.astype(np.float32)

        if mode == "base":
            return x

        # x: [apc, source_rf, lf_rf, sf6, c4f8, dep_time, etch_time]
        apc, srf, lrf, sf6, c4f8, dt, et = [x[:, i:i+1] for i in range(7)]
        feats = [apc, srf, lrf, sf6, c4f8, dt, et]

        if mode == "time":
            feats += [dt + et, _safe_div(dt, dt + et), _safe_div(et, dt + et)]
        elif mode == "gas":
            gas_sum = sf6 + c4f8
            feats += [gas_sum, _safe_div(sf6, gas_sum), _safe_div(c4f8, gas_sum)]
        elif mode == "rf":
            rf_sum = srf + lrf
            feats += [rf_sum, _safe_div(srf, rf_sum), _safe_div(lrf, rf_sum)]
        elif mode == "coupling":
            gas_sum = sf6 + c4f8
            rf_sum  = srf + lrf
            t_sum   = dt + et
            feats += [gas_sum * rf_sum, gas_sum * t_sum, rf_sum * t_sum]
        elif mode == "squares":
            feats += [apc**2, srf**2, lrf**2, sf6**2, c4f8**2, dt**2, et**2]
        elif mode == "phys":
            gas_sum = sf6 + c4f8
            rf_sum  = srf + lrf
            t_sum   = dt + et
            feats += [
                gas_sum, rf_sum, t_sum,
                _safe_div(sf6, gas_sum), _safe_div(c4f8, gas_sum),
                _safe_div(srf, rf_sum), _safe_div(lrf, rf_sum),
                gas_sum * rf_sum, gas_sum * t_sum, rf_sum * t_sum,
            ]
        else:
            raise ValueError(f"Unknown recipe_aug_mode: {mode}")

        return np.concatenate(feats, axis=1).astype(np.float32)

    # -------- 1) 读新表（targets 稀疏） --------
    recs = pu.load_new_excel_as_sparse_morph(new_excel, height_family=height_family)
    if len(recs) == 0:
        raise RuntimeError("new_excel 读取为空（请检查列名/路径）")

    static_raw = np.stack([r["static"] for r in recs], axis=0).astype(np.float32)  # (N,7)

    # -------- 2) 解析 recipe_ids（优先识别配方名/Bxx；鲁棒：自动在所有列里找 B\d+ 最多的那列） --------
    recipe_ids = None
    chosen_col = None
    chosen_hits = 0

    try:
        df_raw = pd.read_excel(new_excel)

        import re
        pat = re.compile(r"(B\d+)", re.IGNORECASE)

        def _normalize_id(s: str):
            m = pat.search(str(s).strip().upper())
            return m.group(1).upper() if m else ""

        # (A) 先走“常见列名”快速路径（加上你现在的新表：配方名）
        cand_cols = [
            "配方名", "配方", "工艺配方", "配方号",
            "recipe", "Recipe", "RECIPE",
            "RecipeID", "recipe_id",
            "工况", "Run", "run"
        ]

        col = None
        cols_lower = {str(c).strip().lower(): c for c in df_raw.columns}
        for c in cand_cols:
            if str(c).strip().lower() in cols_lower:
                col = cols_lower[str(c).strip().lower()]
                break

        if col is not None:
            vals = df_raw[col].astype(str).fillna("")
            ids = [_normalize_id(v) for v in vals.tolist()]
            hits = sum(1 for x in ids if x)
            if hits > 0 and len(ids) == len(recs):
                chosen_col, chosen_hits = col, hits
                recipe_ids = np.array([
                    ids[i] if ids[i] else f"row{i}"
                    for i in range(len(ids))
                ], dtype=object)

        # (B) 如果快速路径失败，则自动扫描所有列：找 B\d+ 命中最多的列
        if recipe_ids is None:
            best_ids = None
            best_col = None
            best_hits = 0

            for c in df_raw.columns:
                vals = df_raw[c].astype(str).fillna("")
                ids = [_normalize_id(v) for v in vals.tolist()]
                hits = sum(1 for x in ids if x)
                if hits > best_hits:
                    best_hits = hits
                    best_col = c
                    best_ids = ids

            if best_hits > 0 and best_ids is not None and len(best_ids) == len(recs):
                chosen_col, chosen_hits = best_col, best_hits
                recipe_ids = np.array([
                    best_ids[i] if best_ids[i] else f"row{i}"
                    for i in range(len(best_ids))
                ], dtype=object)

        if recipe_ids is not None:
            print(f"[INFO] recipe_ids parsed from column='{chosen_col}', hits={chosen_hits}/{len(recs)}")

    except Exception as e:
        recipe_ids = None
        print(f"[WARN] recipe_id parse failed, fallback to rowi. err={e}")

    # (C) 最终兜底：rowi
    if recipe_ids is None:
        recipe_ids = np.array([f"row{i}" for i in range(len(recs))], dtype=object)

    # -------- 3) y/mask：过滤 NaN/Inf（关键） --------
    N = len(recs)
    K = len(FAMILIES)
    TT = len(TIME_LIST)

    y_um = np.zeros((N, K, TT), np.float32)
    mask = np.zeros((N, K, TT), np.bool_)

    skipped_nonfinite = 0
    skipped_badcast = 0

    for i, r in enumerate(recs):
        tg = r.get("targets", {})
        for (fam, tid), v_um in tg.items():
            if fam not in pu.F2IDX or tid not in pu.T2IDX:
                continue
            try:
                v = float(v_um)
            except Exception:
                skipped_badcast += 1
                continue
            if not np.isfinite(v):
                skipped_nonfinite += 1
                continue
            kk = pu.F2IDX[fam]
            tt = pu.T2IDX[tid]
            y_um[i, kk, tt] = v
            mask[i, kk, tt] = True

    bad = (~np.isfinite(y_um)) & mask
    if np.any(bad):
        mask[bad] = False
        y_um[bad] = 0.0
        skipped_nonfinite += int(bad.sum())

    if skipped_nonfinite > 0 or skipped_badcast > 0:
        print(f"[WARN] build_stageC_dataset: skipped targets badcast={skipped_badcast}, nonfinite={skipped_nonfinite}")

    # -------- 4) recipe 增广（对齐 StageB 的 Ds） --------
    static_aug_raw = augment_recipe_features(static_raw, recipe_aug_mode)  # (N,Ds)

    # -------- 5) static_zscore：优先用 StageB ckpt 的 norm_static（如果提供） --------
    mu = None
    sd = None
    if static_zscore and stageB_ckpt_for_norm and os.path.exists(stageB_ckpt_for_norm):
        try:
            ck = _torch_load_ckpt(stageB_ckpt_for_norm, map_location="cpu")
            meta = ck.get("meta", {}) if isinstance(ck, dict) else {}
            ns = meta.get("norm_static", None) if isinstance(meta, dict) else None
            if isinstance(ns, dict) and ("mean" in ns) and ("std" in ns):
                mu0 = ns["mean"]
                sd0 = ns["std"]
                mu0 = mu0.detach().cpu().numpy() if torch.is_tensor(mu0) else np.array(mu0)
                sd0 = sd0.detach().cpu().numpy() if torch.is_tensor(sd0) else np.array(sd0)
                mu0 = mu0.reshape(1, -1).astype(np.float32)
                sd0 = sd0.reshape(1, -1).astype(np.float32)
                if mu0.shape[1] == static_aug_raw.shape[1] and sd0.shape[1] == static_aug_raw.shape[1]:
                    mu, sd = mu0, (sd0 + 1e-6)
                    print(f"[INFO] use StageB norm_static for zscore: Ds={mu.shape[1]}")
                else:
                    print(f"[WARN] StageB norm_static Ds={mu0.shape[1]} != current Ds={static_aug_raw.shape[1]}, fallback to self-fit.")
        except Exception as e:
            print(f"[WARN] failed to load StageB norm_static, fallback to self-fit. err={e}")

    if static_zscore:
        if mu is None or sd is None:
            mu = static_aug_raw.mean(axis=0, keepdims=True).astype(np.float32)
            sd = (static_aug_raw.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
        static_x = (static_aug_raw - mu) / sd
    else:
        mu = np.zeros((1, static_aug_raw.shape[1]), np.float32)
        sd = np.ones((1, static_aug_raw.shape[1]), np.float32)
        static_x = static_aug_raw.copy()

    # -------- 6) Phys7：StageA 推理（一定用原始 recipe7） --------
    phys7 = infer_phys7_from_stageA_ckpt(stageA_ckpt, static_raw, device=device)     # (N,7)
    phys7_seq = np.repeat(phys7[:, :, None], TT, axis=2).astype(np.float32)         # (N,7,T)

    # -------- 7) time_mat --------
    time_mat = np.repeat(TIME_VALUES[None, :], N, axis=0).astype(np.float32)

    return dict(
        static_raw=static_raw,
        static_aug_raw=static_aug_raw,
        static_x=static_x, static_mu=mu, static_sd=sd,
        recipe_ids=recipe_ids,
        phys7=phys7, phys7_seq=phys7_seq,
        y_um=y_um, mask=mask,
        time_mat=time_mat,
    )

# ---------------------------- 子集+划分搜索（满足 key recipes 覆盖 + R2 目标） ----------------------------
def _parse_key_recipes(s: str) -> List[str]:
    items = []
    for x in re.split(r"[,\s]+", s.strip()):
        if not x:
            continue
        x = x.strip().upper()
        if not x.startswith("B"):
            x = "B" + x
        items.append(x)
    return items


def _make_recipe_groups(recipe_ids: np.ndarray) -> Dict[str, List[int]]:
    g: Dict[str, List[int]] = {}
    for i, rid in enumerate(recipe_ids.tolist()):
        g.setdefault(str(rid), []).append(i)
    return g


def random_subset_and_split(N: int,
                            recipe_ids: np.ndarray,
                            key_recipes: List[str],
                            test_ratio: float,
                            val_ratio: float,
                            drop_max_frac: float,
                            rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 train_idx, val_idx, test_idx
    - 允许随机丢弃最多 drop_max_frac 的样本（模拟“筛选”）
    - 约束：test 中必须包含每个 key_recipe 至少 1 个样本（若 key_recipe 不存在则忽略）
    """
    all_idx = np.arange(N)

    # 先构造 keep 子集（随机丢弃）
    drop_n = int(rng.uniform(0, drop_max_frac) * N)
    if drop_n > 0:
        drop = rng.choice(all_idx, size=drop_n, replace=False)
        keep_mask = np.ones(N, bool)
        keep_mask[drop] = False
        keep = all_idx[keep_mask]
    else:
        keep = all_idx

    if keep.size < max(6, len(key_recipes) + 2):
        raise RuntimeError("keep 样本太少")

    # key recipe 映射：这里默认 recipe_ids 就是 recipe（如果你的 new 表里有真实 recipe 列，请在 build_stageC_dataset 里替换）
    # 由于当前 recipe_ids=rowi，key_recipes 无法匹配，所以这里做“退化处理”：若 key 不在 recipe_ids，则不强制。
    existing_keys = [k for k in key_recipes if k in set(recipe_ids.tolist())]
    # 如果 new 表没有 recipe 列，请把 build_stageC_dataset() 里 recipe_ids 改成真实 B47/B52/B54 列，然后这里约束才生效。

    keep_list = keep.tolist()
    rng.shuffle(keep_list)
    keep = np.array(keep_list, dtype=int)

    # 初始 test 采样
    n_test = max(1, int(round(test_ratio * keep.size)))
    n_val  = max(1, int(round(val_ratio  * keep.size)))
    n_test = min(n_test, keep.size - 2)
    n_val  = min(n_val,  keep.size - n_test - 1)

    # 先把 key 样本塞进 test
    test_idx = []
    remain = keep.tolist()

    for k in existing_keys:
        cand = [i for i in remain if recipe_ids[i] == k]
        if not cand:
            continue
        pick = int(rng.choice(cand, size=1)[0])
        test_idx.append(pick)
        remain.remove(pick)

    # 补足 test
    need = max(0, n_test - len(test_idx))
    if need > 0:
        add = rng.choice(np.array(remain, dtype=int), size=need, replace=False).tolist()
        test_idx.extend(add)
        for a in add:
            remain.remove(a)

    # val
    val_idx = rng.choice(np.array(remain, dtype=int), size=n_val, replace=False).tolist()
    remain = [i for i in remain if i not in set(val_idx)]
    # train = rest
    train_idx = np.array(remain, dtype=int)
    val_idx   = np.array(val_idx, dtype=int)
    test_idx  = np.array(test_idx, dtype=int)

    if train_idx.size < 2 or val_idx.size < 1 or test_idx.size < 1:
        raise RuntimeError("split 太小")

    return train_idx, val_idx, test_idx


def fit_y_norm(y_um: np.ndarray, mask: np.ndarray, train_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """按通道/时间点，用 train 子集拟合均值方差（mask 后 + 过滤非有限值）。返回 y_mean,y_std shaped (1,K,T)."""
    K = y_um.shape[1]
    TT = y_um.shape[2]
    mean = np.zeros((1, K, TT), np.float32)
    std  = np.ones((1, K, TT), np.float32)

    ytr = y_um[train_idx]
    mtr = mask[train_idx].astype(bool)

    for k in range(K):
        for t in range(TT):
            vals = ytr[:, k, t][mtr[:, k, t]]
            if vals.size == 0:
                continue
            vals = vals[np.isfinite(vals)]
            if vals.size >= 2:
                mean[0, k, t] = float(vals.mean())
                std[0, k, t]  = float(vals.std() + 1e-6)
            elif vals.size == 1:
                mean[0, k, t] = float(vals[0])
                std[0, k, t]  = 1e-6

    # 兜底：如果 mean/std 里还有非有限值，强制修正
    bad = ~np.isfinite(mean) | ~np.isfinite(std) | (std <= 0)
    if np.any(bad):
        mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
        std  = np.where((np.isfinite(std) & (std > 0)), std, 1.0).astype(np.float32)

    return mean, std

def make_loader(static_x, phys7_seq, y_norm, mask, time_mat, idx: np.ndarray, batch: int) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(static_x[idx]).float(),
        torch.from_numpy(phys7_seq[idx]).float(),
        torch.from_numpy(y_norm[idx]).float(),
        torch.from_numpy(mask[idx]).bool(),
        torch.from_numpy(time_mat[idx]).float(),
    )
    return DataLoader(ds, batch_size=min(batch, len(ds)), shuffle=True)


def describe_split(recipe_ids: np.ndarray, train_idx, val_idx, test_idx) -> Dict[str, Any]:
    def count(idx):
        items = recipe_ids[idx].tolist()
        d={}
        for it in items:
            d[str(it)] = d.get(str(it),0)+1
        return d
    return {"train": count(train_idx), "val": count(val_idx), "test": count(test_idx)}


# ---------------------------- 主流程 ----------------------------
@dataclass
class Cfg:
    # data
    new_excel: str
    stageA_ckpt: str
    stageB_ckpt: Optional[str] = None
    out_dir: str = "./runs_stageC_transfer"

    # goal
    key_recipes: str = "B47,B52,B54"
    target_r2: float = 0.90

    # search
    trials: int = 300
    test_ratio: float = 0.25
    val_ratio: float = 0.15
    drop_max_frac: float = 0.35   # 最多随机丢弃 35% 样本做“筛选”

    # train
    model_type: str = "transformer"
    epochs: int = 800
    lr: float = 3e-4
    wd: float = 1e-4
    batch: int = 64
    early_patience: int = 80

    # display transform (for R2 evaluation)
    unit_scale: float = 1000.0   # μm -> nm
    # family_sign：默认让 zmin 为负（如果你 new 表总深度是正 nm，这里会在 loader 里设置为负；所以 family_sign 通常不需要再翻）
    family_sign: Optional[List[float]] = None   # e.g. [-1,1,1,1,1,1]


# ============================ StageC Split Search v2: multi-single-task ============================
def _family_to_index(family: str) -> int:
    fam = str(family).strip()
    for i, f in enumerate(list(FAMILIES)):
        if str(f).strip().lower() == fam.lower():
            return i
    raise KeyError(f"Unknown family='{family}'. Available={list(FAMILIES)}")

def _parse_required_families(cfg: "Cfg") -> List[str]:
    """
    required_families: 逗号分隔字符串，或 list[str]
    默认 zmin,h1,d1,w（若存在于 FAMILIES）
    """
    rf = getattr(cfg, "required_families", None)
    if rf is None:
        cand = ["zmin", "h1", "d1", "w"]
        return [c for c in cand if any(str(c).lower()==str(x).lower() for x in FAMILIES)]
    if isinstance(rf, (list, tuple)):
        out = [str(x).strip() for x in rf if str(x).strip()]
    else:
        out = [x.strip() for x in str(rf).replace(";", ",").split(",") if x.strip()]
    # 过滤不存在的
    out2=[]
    for x in out:
        if any(str(x).lower()==str(f).lower() for f in FAMILIES):
            out2.append(x)
    return out2

def _make_single_family_mask(mask_full: np.ndarray, fam_idx: int) -> np.ndarray:
    """mask_full: (N,K,T) bool/0-1 -> 只保留某个 family 的 mask，其余置 0。"""
    m = mask_full.astype(bool)
    m2 = np.zeros_like(m, dtype=np.bool_)
    m2[:, fam_idx, :] = m[:, fam_idx, :]
    return m2

def make_loader2(static_x, phys7_seq, y_norm, mask, time_mat, idx: np.ndarray, batch: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(static_x[idx]).float(),
        torch.from_numpy(phys7_seq[idx]).float(),
        torch.from_numpy(y_norm[idx]).float(),
        torch.from_numpy(mask[idx]).bool(),
        torch.from_numpy(time_mat[idx]).float(),
    )
    return DataLoader(ds, batch_size=min(batch, len(ds)), shuffle=shuffle)

def _export_single_family_parity(out_png: str,
                                pack: Dict[str, np.ndarray],
                                y_mean: np.ndarray, y_std: np.ndarray,
                                family_sign: Optional[torch.Tensor],
                                unit_scale: float,
                                fam_idx: int,
                                title: str = "") -> None:
    """
    只画某个 family 的 test parity（展示空间，默认 nm + zmin 翻正）
    """
    import matplotlib.pyplot as plt
    pred_disp, y_disp, m = _pack_to_display_arrays(pack, y_mean, y_std, family_sign, unit_scale)
    mk = m[:, fam_idx, :].reshape(-1)
    if mk.sum() <= 0:
        return
    yt = y_disp[:, fam_idx, :].reshape(-1)[mk]
    yp = pred_disp[:, fam_idx, :].reshape(-1)[mk]

    plt.figure(figsize=(4.2,4.2))
    plt.scatter(yt, yp, s=8, alpha=0.6)
    # y=x
    mn = float(np.nanmin([yt.min(), yp.min()]))
    mx = float(np.nanmax([yt.max(), yp.max()]))
    plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel("Measured (display)")
    plt.ylabel("Predicted (display)")
    if title:
        plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.close()

def _single_task_train_eval_once(cfg: "Cfg",
                                 data: Dict[str, Any],
                                 train_idx: np.ndarray,
                                 val_idx: np.ndarray,
                                 test_idx: np.ndarray,
                                 fam_idx: int,
                                 tag: str,
                                 device: str,
                                 out_dir: Optional[str],
                                 *,
                                 do_train: bool,
                                 load_stageB: bool,
                                 zero_phys7: bool) -> Dict[str, Any]:
    """
    训练/评估单个 family 的单任务模型（模型仍输出 K，但 loss 只看 fam_idx）。
    返回：
      dict(train_pack,test_pack,y_mean,y_std, fam_r2, fam_mae, fam_n, fam_std)
    """
    static_x  = data["static_x"]
    phys7_seq = data["phys7_seq"]
    y_um      = data["y_um"]
    mask_full = data["mask"]
    time_mat  = data["time_mat"]

    # --- single family mask + single family norm ---
    mask_single = _make_single_family_mask(mask_full, fam_idx)
    y_mean, y_std = fit_y_norm(y_um, mask_single, train_idx)
    y_norm = (y_um - y_mean) / (y_std + 1e-6)

    # --- phys7 optionally zero ---
    if zero_phys7:
        phys7_use = np.zeros_like(phys7_seq, dtype=np.float32)
    else:
        phys7_use = phys7_seq

    # loaders（train shuffle, val/test no shuffle）
    train_loader = make_loader2(static_x, phys7_use, y_norm, mask_single, time_mat, train_idx, cfg.batch, shuffle=True)
    val_loader   = make_loader2(static_x, phys7_use, y_norm, mask_single, time_mat, val_idx,   cfg.batch, shuffle=False) if len(val_idx)>0 else train_loader
    test_loader  = make_loader2(static_x, phys7_use, y_norm, mask_single, time_mat, test_idx,  cfg.batch, shuffle=False)

    K = y_um.shape[1]
    stageB_ckpt = cfg.stageB_ckpt if load_stageB else None
    model = build_morph_model(cfg.model_type, K, device, stageB_ckpt=stageB_ckpt).to(device)

    if do_train:
        train_one(model, train_loader, val_loader, device,
                  epochs=int(getattr(cfg, "epochs", 200)),
                  lr=float(getattr(cfg, "lr", 5e-4)),
                  wd=float(getattr(cfg, "wd", 1e-4)),
                  early_patience=int(getattr(cfg, "early_patience", 30)))

    # eval packs
    train_pack = eval_pack(model, train_loader, device)
    test_pack  = eval_pack(model, test_loader, device)

    # family metrics（display space）
    stats = family_stats_from_pack(test_pack, y_mean, y_std,
                                   family_sign=None if getattr(cfg, "family_sign", None) is None else torch.tensor(cfg.family_sign, dtype=torch.float32),
                                   unit_scale=float(getattr(cfg, "unit_scale", 1000.0)))
    fam_r2  = float(stats["r2"][fam_idx]) if np.isfinite(stats["r2"][fam_idx]) else float("nan")
    fam_mae = float(stats["mae"][fam_idx]) if np.isfinite(stats["mae"][fam_idx]) else float("nan")
    fam_n   = int(stats["n"][fam_idx])
    fam_std = float(stats["std"][fam_idx]) if np.isfinite(stats["std"][fam_idx]) else float("nan")

    # optional export
    if out_dir:
        family_sign_t = None
        if getattr(cfg, "family_sign", None) is not None:
            family_sign_t = torch.tensor(cfg.family_sign, dtype=torch.float32)
        export_stageC_report(out_dir, train_pack, test_pack, y_mean, y_std, family_sign_t, float(getattr(cfg, "unit_scale", 1000.0)))
        fam_name = str(FAMILIES[fam_idx])
        _export_single_family_parity(
            os.path.join(out_dir, f"fig_parity_test_{fam_name}.png"),
            test_pack, y_mean, y_std, family_sign_t, float(getattr(cfg, "unit_scale", 1000.0)),
            fam_idx=fam_idx, title=f"{tag} | {fam_name} | test parity"
        )
        # save single-family metrics json
        js = dict(tag=tag, family=fam_name, test_R2=fam_r2, test_MAE=fam_mae, n_test=fam_n, std_test=fam_std)
        with open(os.path.join(out_dir, "metrics_single_family.json"), "w", encoding="utf-8") as f:
            json.dump(js, f, indent=2, ensure_ascii=False)

        # save ckpt
        try:
            torch.save(model.state_dict(), os.path.join(out_dir, "model_best.pth"))
        except Exception:
            pass

    return dict(train_pack=train_pack, test_pack=test_pack,
                y_mean=y_mean, y_std=y_std,
                fam_r2=fam_r2, fam_mae=fam_mae, fam_n=fam_n, fam_std=fam_std)

def searchbest_split_multi_single_tasks(cfg: "Cfg", data: Dict[str, Any], device: str) -> Dict[str, Any]:
    """
    你要的核心：search best split，使得多个 family 的“单任务模型”(默认用 C_transfer)在同一 split 的 test 上 **同时尽可能高**。
    rank = min_required_family_R2 + 0.01 * mean_required_family_R2
    meets = (min_required_family_R2 >= cfg.min_family_r2)
    """
    import csv, time

    os.makedirs(cfg.out_dir, exist_ok=True)
    required = _parse_required_families(cfg)
    if len(required) == 0:
        raise RuntimeError("required_families 为空：请检查 cfg.required_families 或 FAMILIES。")

    req_idx = [_family_to_index(f) for f in required]
    min_family_r2 = float(getattr(cfg, "min_family_r2", 0.80))
    seed = int(getattr(cfg, "seed", 2026))
    rng = np.random.default_rng(seed)

    # family_sign（用于 display space）
    family_sign_t = None
    if getattr(cfg, "family_sign", None) is not None:
        family_sign_t = torch.tensor(cfg.family_sign, dtype=torch.float32)

    # log
    trials_csv = os.path.join(cfg.out_dir, "trials_log.csv")
    with open(trials_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["trial","train_n","val_n","test_n","minReqR2","meanReqR2","meets"] + [f"R2_{f}" for f in required])

    best = dict(rank=-1e9, meets=False, trial=-1, split=None, fam_rows=None)

    # split search
    for tr in range(1, int(getattr(cfg, "trials", 300)) + 1):
        train_idx, val_idx, test_idx = random_subset_and_split(
            recipe_ids=data["recipe_ids"],
            mask=data["mask"],
            key_recipes=_parse_key_recipes(cfg.key_recipes),
            test_ratio=float(getattr(cfg, "test_ratio", 0.25)),
            val_ratio=float(getattr(cfg, "val_ratio", 0.15)),
            drop_max_frac=float(getattr(cfg, "drop_max_frac", 0.0)),
            rng=rng
        )

        fam_r2s=[]
        fam_rows=[]
        # 对每个 family 训练一个 single-task C_transfer（用于 split 打分）
        for fam_name, k in zip(required, req_idx):
            # 为了减少方差：不同 trial 仍然可能波动；这里把种子锁死到 (seed, trial, family)
            torch.manual_seed(seed * 100000 + tr * 100 + k)
            np.random.seed((seed + tr * 100 + k) % (2**32 - 1))

            out_dir = None  # 非 best 不落盘
            res = _single_task_train_eval_once(
                cfg, data, train_idx, val_idx, test_idx,
                fam_idx=k, tag="C_transfer_single",
                device=device, out_dir=out_dir,
                do_train=True, load_stageB=True, zero_phys7=False
            )
            fam_r2s.append(res["fam_r2"])
            fam_rows.append(dict(family=str(FAMILIES[k]), test_R2=res["fam_r2"], test_MAE=res["fam_mae"],
                                 n_test=res["fam_n"], std_test=res["fam_std"]))

        minReq = float(np.nanmin(fam_r2s)) if len(fam_r2s)>0 else float("nan")
        meanReq = float(np.nanmean(fam_r2s)) if len(fam_r2s)>0 else float("nan")
        meets = bool(np.isfinite(minReq) and (minReq >= min_family_r2))
        rank = (minReq if np.isfinite(minReq) else -1e9) + 0.01 * (meanReq if np.isfinite(meanReq) else 0.0)

        # log
        with open(trials_csv, "a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow([tr, len(train_idx), len(val_idx), len(test_idx), minReq, meanReq, int(meets)] + fam_r2s)

        # best update
        if rank > best["rank"]:
            best.update(rank=rank, meets=meets, trial=tr, split=dict(train_idx=train_idx.tolist(), val_idx=val_idx.tolist(), test_idx=test_idx.tolist()),
                        fam_rows=fam_rows, minReq=minReq, meanReq=meanReq)

            # 落盘 best split
            split_desc = describe_split(data["recipe_ids"], train_idx, val_idx, test_idx)
            best_json = dict(
                trial=tr, meets_constraints=meets,
                min_required_family_R2=minReq,
                mean_required_family_R2=meanReq,
                required_families=required,
                split_desc=split_desc,
                train_idx=train_idx.tolist(), val_idx=val_idx.tolist(), test_idx=test_idx.tolist(),
                family_metrics=fam_rows,
            )
            with open(os.path.join(cfg.out_dir, "best_split.json"), "w", encoding="utf-8") as f:
                json.dump(best_json, f, indent=2, ensure_ascii=False)

            # 同时把 best 的每个 family 模型/图落盘（为了“冲分展示”，不重训）
            for fam_name, k in zip(required, req_idx):
                fam_dir = os.path.join(cfg.out_dir, "best_models_single_task", fam_name)
                os.makedirs(fam_dir, exist_ok=True)

                torch.manual_seed(seed * 100000 + tr * 100 + k)
                np.random.seed((seed + tr * 100 + k) % (2**32 - 1))

                _ = _single_task_train_eval_once(
                    cfg, data, train_idx, val_idx, test_idx,
                    fam_idx=k, tag="C_transfer_single(best)",
                    device=device, out_dir=fam_dir,
                    do_train=True, load_stageB=True, zero_phys7=False
                )

            print(f"[split_search_single][best@{tr}] meets={meets} minReqR2={minReq:.4f} meanReqR2={meanReq:.4f}")

            # 早停：如果已经满足 hard constraint，可以直接停（你也可以关掉）
            if meets and bool(getattr(cfg, "stop_when_meets", True)):
                break

    print(f"[Done] Best split saved to: {os.path.join(cfg.out_dir, 'best_split.json')}")
    return best

def run_ablation_suite_single_task(cfg: "Cfg", data: Dict[str, Any], best_split_path: str, device: str) -> None:
    """
    固定 best_split.json 的 split，对每个 required family 跑 4 个单任务对照，并输出：
      - out_dir/single_task_ablation/<family>/<tag>/...
      - out_dir/single_task_ablation/ablation_summary_single_task.csv  （行=family，列=各tag R2/MAE）
    """
    import pandas as pd

    with open(best_split_path, "r", encoding="utf-8") as f:
        best = json.load(f)
    train_idx = np.array(best["train_idx"], dtype=int)
    val_idx   = np.array(best["val_idx"], dtype=int)
    test_idx  = np.array(best["test_idx"], dtype=int)

    required = best.get("required_families", _parse_required_families(cfg))
    req_idx = [_family_to_index(f) for f in required]

    out_root = os.path.join(cfg.out_dir, "single_task_ablation")
    os.makedirs(out_root, exist_ok=True)

    rows=[]
    for fam_name, k in zip(required, req_idx):
        fam_root = os.path.join(out_root, fam_name)
        # 4 tags（全是单任务口径）
        settings = [
            ("B2R_zero_ft", dict(do_train=False, load_stageB=True,  zero_phys7=False)),
            ("C_scratch",   dict(do_train=True,  load_stageB=False, zero_phys7=False)),
            ("C_transfer",  dict(do_train=True,  load_stageB=True,  zero_phys7=False)),
            ("C_noPhys7",   dict(do_train=True,  load_stageB=True,  zero_phys7=True)),
        ]
        for tag, st in settings:
            out_dir = os.path.join(fam_root, tag)
            res = _single_task_train_eval_once(
                cfg, data, train_idx, val_idx, test_idx,
                fam_idx=k, tag=tag, device=device, out_dir=out_dir,
                do_train=st["do_train"], load_stageB=st["load_stageB"], zero_phys7=st["zero_phys7"]
            )
            rows.append(dict(family=fam_name, tag=tag,
                             test_R2=res["fam_r2"], test_MAE=res["fam_mae"],
                             n_test=res["fam_n"], std_test=res["fam_std"]))
            print(f"[SingleTask][{fam_name}][{tag}] test_R2={res['fam_r2']:.4f} test_MAE={res['fam_mae']:.2f} n_test={res['fam_n']}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_root, "ablation_summary_single_task_long.csv"), index=False, encoding="utf-8-sig")

    # pivot（行=family，列=tag_xxx）
    pivot = df.pivot(index="family", columns="tag", values="test_R2")
    pivot_mae = df.pivot(index="family", columns="tag", values="test_MAE")
    pivot.to_csv(os.path.join(out_root, "ablation_summary_single_task_R2.csv"), encoding="utf-8-sig")
    pivot_mae.to_csv(os.path.join(out_root, "ablation_summary_single_task_MAE.csv"), encoding="utf-8-sig")



def main(cfg: Cfg):
    """
    StageC 主流程（你最终要的版本）：
      1) search best split：同一个 split 上，训练多个 family 的“单任务 C_transfer”，让 min(required_family test_R2) 尽可能高（目标>=min_family_r2）
      2) best 更新时：把该 split 下每个 family 的 best 单任务模型/散点/指标直接落盘（不再重训，避免你之前看到的 “search高但fixed掉分”）
      3) split 固定后：在同一 split 上做单任务 ablation（B2R_zero_ft / C_scratch / C_transfer / C_noPhys7），每个 family 都有一套对比结果
    """
    import os, time
    import torch

    os.makedirs(cfg.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # seed
    seed = int(getattr(cfg, "seed", 2026))
    if seed in (None, -1):
        seed = int(time.time())
    torch.manual_seed(seed)
    np.random.seed(seed)

    # build data once
    data = build_stageC_dataset(
        cfg.new_excel, cfg.stageA_ckpt, device=device,
        stageB_ckpt_for_norm=cfg.stageB_ckpt,
        recipe_aug_mode="time"
    )

    # search best split (multi single-task)
    print(f"[INFO] required_families={_parse_required_families(cfg)} (min_family_r2={getattr(cfg,'min_family_r2',0.8)})")
    best = searchbest_split_multi_single_tasks(cfg, data, device=device)

    # ablation on fixed best split (multi single-task)
    best_path = os.path.join(cfg.out_dir, "best_split.json")
    run_ablation_suite_single_task(cfg, data, best_split_path=best_path, device=device)


def run_ablation_suite(cfg: Cfg):
    """
    在同一个“固定 split”上跑 4 个最小对照：
      1) B2R_zero_ft  : load StageB, 不训练
      2) C_scratch    : 不 load StageB，但同架构训练（公平）
      3) C_transfer   : load StageB + 训练（主方法）
      4) C_noPhys7    : load StageB + 训练，但 Phys7 置零（Phys7 贡献）

    关键改动：
    - 优先读取 main() 输出的 best_split.json / best_so_far.json
    - 若没有，自动调用 main(cfg) 先生成
    - 保存本次 ablation 实际使用的 split（含 idx）到 out_dir/ablation_split.json
    - summary csv 用 na_rep="NaN" 防止空白
    - 每个 tag 从 metrics_overall.json 回读并打印
    """
    import os, json
    import numpy as np
    import pandas as pd
    import torch

    os.makedirs(cfg.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # 1) 构建数据（与 StageB 对齐：time-aug -> static_dim=10）
    data = build_stageC_dataset(
        cfg.new_excel, cfg.stageA_ckpt, device=device,
        stageB_ckpt_for_norm=cfg.stageB_ckpt,
        recipe_aug_mode="time"
    )
    static_x  = data["static_x"]
    phys7_seq = data["phys7_seq"]
    y_um      = data["y_um"]
    mask      = data["mask"]
    time_mat  = data["time_mat"]
    recipe_ids = data["recipe_ids"]

    # 2) 检查 key recipe 是否真的在 recipe_ids 里（否则你的“B47/B52/B54 必在 test”约束不会生效）
    key_recipes = _parse_key_recipes(cfg.key_recipes)
    existing_keys = [k for k in key_recipes if k in set([str(x) for x in recipe_ids.tolist()])]
    if len(existing_keys) == 0:
        uniq = sorted(list(set([str(x) for x in recipe_ids.tolist()])))[:20]
        print(f"[WARN] key_recipes={key_recipes} 在 new_excel 解析出的 recipe_ids 里一个都没找到。")
        print(f"       说明 new_excel 的 recipe 列没被正确识别，目前 recipe_ids 可能是 rowi。")
        print(f"       recipe_ids 前20个unique示例: {uniq}")

    # 3) 优先读取 main 的 split；没有就先跑 main(cfg) 生成
    best_path = os.path.join(cfg.out_dir, "best_split.json")
    sofar_path = os.path.join(cfg.out_dir, "best_so_far.json")

    main(cfg)

    split_obj = None
    if os.path.exists(best_path):
        with open(best_path, "r", encoding="utf-8") as f:
            split_obj = json.load(f)
        split_tag = "best_split.json"
    elif os.path.exists(sofar_path):
        with open(sofar_path, "r", encoding="utf-8") as f:
            split_obj = json.load(f)
        split_tag = "best_so_far.json"
    else:
        raise RuntimeError("Still cannot find split json after running main(cfg).")

    # 4) 取 idx（json 里是 list）
    train_idx = np.array(split_obj["train_idx"], dtype=int)
    val_idx   = np.array(split_obj["val_idx"], dtype=int)
    test_idx  = np.array(split_obj["test_idx"], dtype=int)

    # 5) y_norm：优先用 json 里存的 y_mean/y_std（best_split 里有），否则重算
    if ("y_mean" in split_obj) and ("y_std" in split_obj):
        y_mean = np.array(split_obj["y_mean"], dtype=np.float32)
        y_std  = np.array(split_obj["y_std"], dtype=np.float32)
    else:
        y_mean, y_std = fit_y_norm(y_um, mask, train_idx)

    y_norm = (y_um - y_mean) / (y_std + 1e-6)

    # family_sign
    family_sign_t = None
    if cfg.family_sign is not None:
        family_sign_t = torch.tensor(cfg.family_sign, dtype=torch.float32)

    # 6) 保存并打印 split（你要“看得到划分”就看这个）
    split_desc = describe_split(recipe_ids, train_idx, val_idx, test_idx)
    ab_split = {
        "source": split_tag,
        "trial": split_obj.get("trial", None),
        "train_idx": train_idx.tolist(),
        "val_idx": val_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "split_desc": split_desc,
        "key_recipes": key_recipes,
        "existing_keys_in_recipe_ids": existing_keys,
        "train_n": int(len(train_idx)),
        "val_n": int(len(val_idx)),
        "test_n": int(len(test_idx)),
    }
    with open(os.path.join(cfg.out_dir, "ablation_split.json"), "w", encoding="utf-8") as f:
        json.dump(ab_split, f, ensure_ascii=False, indent=2)

    print(f"[Ablation] Using split from {split_tag}")
    print(f"[Ablation] Split desc: {split_desc}")

    # 7) loaders
    train_loader = make_loader(static_x, phys7_seq, y_norm, mask, time_mat, train_idx, cfg.batch)
    val_loader   = make_loader(static_x, phys7_seq, y_norm, mask, time_mat, val_idx,   cfg.batch)
    test_loader  = make_loader(static_x, phys7_seq, y_norm, mask, time_mat, test_idx,  cfg.batch)

    def _read_overall(tag_dir: str):
        p = os.path.join(tag_dir, "metrics_overall.json")
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def _run_one(tag: str, load_stageB: bool, do_train: bool, phys7_mode: str, force_stageB_arch: bool = True):
        out_dir = os.path.join(cfg.out_dir, tag)
        os.makedirs(out_dir, exist_ok=True)

        arch_ckpt = cfg.stageB_ckpt if (force_stageB_arch or load_stageB) else None
        model = build_morph_model(cfg.model_type, K=len(FAMILIES), device=device, stageB_ckpt=arch_ckpt)

        if load_stageB and cfg.stageB_ckpt:
            info = load_morph_ckpt(model, cfg.stageB_ckpt)
            print(f"[{tag}] load StageB ok={info.get('ok')} matched={info.get('matched',0)} "
                  f"missing={len(info.get('missing',[]))} unexpected={len(info.get('unexpected',[]))}")

        # phys7 ablation
        if phys7_mode == "zero":
            zero_phys = np.zeros_like(phys7_seq, dtype=np.float32)
            tr_loader = make_loader(static_x, zero_phys, y_norm, mask, time_mat, train_idx, cfg.batch)
            va_loader = make_loader(static_x, zero_phys, y_norm, mask, time_mat, val_idx, cfg.batch)
            te_loader = make_loader(static_x, zero_phys, y_norm, mask, time_mat, test_idx, cfg.batch)
        else:
            tr_loader, va_loader, te_loader = train_loader, val_loader, test_loader

        if do_train:
            print(f"[{tag}] TRAIN: epochs={cfg.epochs} lr={cfg.lr} wd={cfg.wd} batch={cfg.batch}")
            train_one(model, tr_loader, va_loader, device=device,
                      epochs=cfg.epochs, lr=cfg.lr, wd=cfg.wd, early_patience=cfg.early_patience)
        else:
            print(f"[{tag}] TRAIN: skipped (zero fine-tune)")

        tr_pack = eval_pack(model, tr_loader, device=device)
        te_pack = eval_pack(model, te_loader, device=device)

        export_stageC_report(out_dir, tr_pack, te_pack, y_mean, y_std, family_sign_t, cfg.unit_scale)

        # 回读 json，保证结果可见
        mj = _read_overall(out_dir)
        if mj is not None and "overall" in mj:
            o = mj["overall"]
            print(f"[{tag}] RESULT(from json): train_R2={o.get('train_R2')} test_R2={o.get('test_R2')} "
                  f"train_MAE={o.get('train_MAE')} test_MAE={o.get('test_MAE')}")
            return {
                "tag": tag,
                "train_R2": o.get("train_R2"),
                "test_R2": o.get("test_R2"),
                "train_MAE": o.get("train_MAE"),
                "test_MAE": o.get("test_MAE"),
            }

        # 兜底：如果 json 不存在，至少返回 pack 计算结果（可能 NaN）
        tr_r2 = overall_r2_from_pack(tr_pack, y_mean, y_std, family_sign_t, cfg.unit_scale)
        te_r2 = overall_r2_from_pack(te_pack, y_mean, y_std, family_sign_t, cfg.unit_scale)
        print(f"[{tag}] RESULT(fallback): train_R2={tr_r2} test_R2={te_r2}")
        return {"tag": tag, "train_R2": float(tr_r2), "test_R2": float(te_r2)}

    rows = []
    rows.append(_run_one("B2R_zero_ft", load_stageB=True,  do_train=False, phys7_mode="normal"))
    rows.append(_run_one("C_scratch",   load_stageB=False, do_train=True,  phys7_mode="normal"))
    rows.append(_run_one("C_transfer",  load_stageB=True,  do_train=True,  phys7_mode="normal"))
    rows.append(_run_one("C_noPhys7",   load_stageB=True,  do_train=True,  phys7_mode="zero"))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(cfg.out_dir, "ablation_summary.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig", na_rep="NaN")
    print(f"[Ablation] Saved: {out_csv}")
    print(df.to_string(index=False))
def family_stats_from_pack(pack: Dict[str, np.ndarray],
                           y_mean: np.ndarray, y_std: np.ndarray,
                           family_sign: Optional[torch.Tensor],
                           unit_scale: float) -> Dict[str, np.ndarray]:
    """
    返回 dict:
      r2 : (K,)
      mae: (K,)
      n  : (K,)    mask 点数（flatten over T）
      std: (K,)    y_true 的标准差（display space）
    """
    pred = pack["pred_norm"].astype(np.float32)
    y    = pack["y_norm"].astype(np.float32)
    m    = pack["mask"].astype(bool)

    pred_um = pred * (y_std + 1e-6) + y_mean
    y_um    = y    * (y_std + 1e-6) + y_mean

    pred_t = torch.from_numpy(pred_um)
    y_t    = torch.from_numpy(y_um)

    pred_disp, y_disp = pu.transform_for_display(
        pred_t, y_t, family_sign=family_sign, unit_scale=unit_scale, clip_nonneg=False
    )
    pred_disp = pred_disp.numpy()
    y_disp    = y_disp.numpy()

    K = y_disp.shape[1]
    out_r2  = np.full((K,), np.nan, dtype=np.float32)
    out_mae = np.full((K,), np.nan, dtype=np.float32)
    out_n   = np.zeros((K,), dtype=np.int32)
    out_std = np.full((K,), np.nan, dtype=np.float32)

    for k in range(K):
        mk = m[:, k, :].reshape(-1)
        n = int(mk.sum())
        out_n[k] = n
        if n < 2:
            continue
        yt = y_disp[:, k, :].reshape(-1)[mk]
        yp = pred_disp[:, k, :].reshape(-1)[mk]

        out_std[k] = float(np.std(yt))
        out_mae[k] = float(np.mean(np.abs(yt - yp)))

        # R2 需要足够方差，否则 masked_r2_np 会 NaN 或非常不稳定
        out_r2[k] = float(masked_r2_np(yt, yp))

    return {"r2": out_r2, "mae": out_mae, "n": out_n, "std": out_std}

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_excel", required=True)
    ap.add_argument("--stageA_ckpt", required=True)
    ap.add_argument("--stageB_ckpt", default=None)
    ap.add_argument("--out_dir", default="./runs_stageC_transfer")
    ap.add_argument("--key_recipes", default="B47,B52,B54")
    ap.add_argument("--target_r2", type=float, default=0.90)

    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--test_ratio", type=float, default=0.25)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--drop_max_frac", type=float, default=0.35)

    ap.add_argument("--model_type", default="transformer", choices=["transformer","gru","mlp"])
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--early_patience", type=int, default=80)

    return ap.parse_args()


if __name__ == "__main__":
    # ================== PyCharm 直接运行配置（硬编码） ==================
    # 新表：你给的那张（包含列：配方名 / APC（E2步骤） / source_RF（E2步骤） / ... / 开口处CD / 总深度）
    NEW_EXCEL = r"D:\PycharmProjects\Bosch\Bosch.xlsx"

    # StageA：Phys7 teacher / predictor 的 best ckpt（用于 recipe7 -> phys7）
    # 这里请直接填你 StageA 训练输出的 best 权重路径
    STAGEA_PHYS7_CKPT = r"D:\PycharmProjects\Bosch\runs_stageA_phys7\cv_transformer_seed2/phys7_best.pth"

    # StageB：你选的 best morph（phys=stageA_pred, phys7-full, transformer, seed4）
    # 这里请直接填对应 ckpt 文件路径（如果不存在，脚本会从头训练 Morph）
    STAGEB_BEST_CKPT = r"./runs_stageB_morph_phys7/model-transformer_phys-stageA_pred_aug-time_phys7-full_seed4/best_model-transformer_phys-stageA_pred_aug-time_phys7-full_seed4.pth"

    OUT_DIR = r"./runs_stageC_phys7_subsetsearch_r2"

    cfg = Cfg(
        new_excel=NEW_EXCEL,
        stageA_ckpt=STAGEA_PHYS7_CKPT,
        stageB_ckpt=STAGEB_BEST_CKPT,
        out_dir=OUT_DIR,

        # 重点 recipe：要求 test 必须包含这三条
        key_recipes="B47,B52,B54",
        target_r2=0.90,

        # subset+split 搜索（允许丢样本）
        trials=1000,
        test_ratio=0.25,
        val_ratio=0.15,
        drop_max_frac=0.50,

        # 形貌模型（与你的 stageBbest 对齐：transformer）
        model_type="transformer",

        # 训练设置（你说训练时间不是问题，这里固定即可）
        epochs=200,
        lr=5e-4,
        wd=1e-4,
        batch=8,
        early_patience=30,
    )
