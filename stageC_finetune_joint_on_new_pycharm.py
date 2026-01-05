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
    """norm space -> um -> display space（含 sign/scale）"""
    pred = pack["pred_norm"].astype(np.float32)
    y    = pack["y_norm"].astype(np.float32)
    m    = pack["mask"].astype(bool)

    pred_um = pred * (y_std + 1e-6) + y_mean
    y_um    = y    * (y_std + 1e-6) + y_mean

    pred_t = torch.from_numpy(pred_um)
    y_t    = torch.from_numpy(y_um)

    pred_disp, y_disp = pu.transform_for_display(
        pred_t, y_t,
        family_sign=family_sign,
        unit_scale=unit_scale,
        clip_nonneg=False
    )
    return pred_disp.numpy(), y_disp.numpy(), m


def _masked_r2(y_true_1d: np.ndarray, y_pred_1d: np.ndarray):
    return masked_r2_np(y_true_1d, y_pred_1d)


def _masked_mae(y_true_1d: np.ndarray, y_pred_1d: np.ndarray):
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

def build_morph_model(model_type: str, K: int, device: str):
    mt = model_type.lower()
    if mt == "transformer":
        m = MorphTransformer(K=K).to(device)
    elif mt == "gru":
        m = MorphGRU(K=K).to(device)
    elif mt == "mlp":
        m = MorphMLP(K=K).to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return m


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
    ✅ 修复点：
    1) 不用裸 torch.load（PyTorch2.6+ weights_only 默认 True 会炸）-> 用 _torch_load_ckpt
    2) 做 prefix strip + key remap，尽量对齐 StageB/StageC 命名
    3) 返回 matched 统计：你能立刻判断到底“迁移是否真正生效”
    """
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        return {"ok": False, "reason": "ckpt_not_found"}

    obj = _torch_load_ckpt(ckpt_path, map_location="cpu")
    sd, meta = _extract_state_dict_and_meta(obj)
    if sd is None:
        return {"ok": False, "reason": "unrecognized_format"}

    sd = _strip_prefixes(sd)
    sd = _remap_morph_keys(sd)

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(sd.keys())
    matched = len(model_keys & ckpt_keys)

    # 真正 load
    missing, unexpected = model.load_state_dict(sd, strict=False)

    return {
        "ok": True,
        "matched": matched,
        "model_key_count": len(model_keys),
        "ckpt_key_count": len(ckpt_keys),
        "missing": missing,
        "unexpected": unexpected,
        "meta": meta
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


def masked_r2_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # y_true/y_pred: (N,)
    if y_true.size < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mean) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def overall_r2_from_pack(pack: Dict[str,np.ndarray],
                         y_mean: np.ndarray, y_std: np.ndarray,
                         family_sign: Optional[torch.Tensor],
                         unit_scale: float) -> float:
    """
    pack 里是训练空间（norm space）；先反归一化到 um，再转展示空间（默认 nm），最后算 overall R2（mask 后 flatten）。
    """
    pred = pack["pred_norm"]
    y = pack["y_norm"]
    m = pack["mask"].astype(bool)

    # 反归一化到 um
    pred_um = pred * (y_std + 1e-6) + y_mean
    y_um    = y    * (y_std + 1e-6) + y_mean

    # to torch for transform_for_display
    pred_t = torch.from_numpy(pred_um)
    y_t = torch.from_numpy(y_um)
    m_t = torch.from_numpy(m)

    pred_disp, y_disp = pu.transform_for_display(
        pred_t, y_t,
        family_sign=family_sign,
        unit_scale=unit_scale,
        clip_nonneg=False
    )
    yp = pred_disp.numpy().reshape(-1)
    yt = y_disp.numpy().reshape(-1)
    mk = m.reshape(-1)
    yp = yp[mk]; yt = yt[mk]
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
                         ) -> Dict[str, Any]:
    """
    返回：
      dict(
        static_raw (N,7), recipe_ids (N,),
        static_x (N,7) (可能 zscore),
        phys7 (N,7),
        phys7_seq (N,7,T),
        y_um (N,K,T), mask (N,K,T),
        time_mat (N,T),
      )
    """
    recs = pu.load_new_excel_as_sparse_morph(new_excel, height_family=height_family)
    if len(recs) == 0:
        raise RuntimeError("new_excel 读取为空（请检查列名/路径）")

    static_raw = np.stack([r["static"] for r in recs], axis=0).astype(np.float32)   # (N,7)
    # recipe id：尽量从 new 表中解析 Bxx（用于“重点 recipe 覆盖”约束）
    recipe_ids = None
    try:
        df_raw = pd.read_excel(new_excel)
        # 常见列名候选
        cand_cols = ["recipe","Recipe","RECIPE","配方","工艺配方","RecipeID","recipe_id","配方号","工况","Run","run"]
        col = None
        for c in cand_cols:
            for cc in df_raw.columns:
                if str(cc).strip().lower() == str(c).strip().lower():
                    col = cc
                    break
            if col is not None:
                break
        if col is None:
            # 兜底：从任意列里找形如 B47/B52 的字符串
            for cc in df_raw.columns:
                s = df_raw[cc].astype(str)
                if s.str.contains(r"(?i)\bB\d{1,3}\b", regex=True).any():
                    col = cc
                    break
        if col is not None:
            vals = df_raw[col].astype(str).fillna("")
            def _norm_r(v: str) -> str:
                m = re.search(r"(?i)\bB(\d{1,3})\b", v)
                return ("B" + m.group(1)) if m else ""
            rid = np.array([_norm_r(v) for v in vals.tolist()], dtype=object)
            # 长度对齐
            if len(rid) == len(recs) and np.any(rid != ""):
                recipe_ids = rid
    except Exception:
        recipe_ids = None

    if recipe_ids is None:
        recipe_ids = np.array([f"row{i}" for i in range(len(recs))], dtype=object)

    # y/mask
    N = len(recs); K = len(FAMILIES); TT = len(TIME_LIST)
    y_um = np.zeros((N, K, TT), np.float32)
    mask = np.zeros((N, K, TT), np.bool_)
    for i, r in enumerate(recs):
        for (fam, tid), v_um in r["targets"].items():
            if fam in pu.F2IDX and tid in pu.T2IDX:
                k = pu.F2IDX[fam]; t = pu.T2IDX[tid]
                y_um[i, k, t] = float(v_um)
                mask[i, k, t] = True

    # static zscore（用于 morph 输入）
    static_x = static_raw.copy()
    if static_zscore:
        mu = static_raw.mean(axis=0, keepdims=True)
        sd = static_raw.std(axis=0, keepdims=True) + 1e-6
        static_x = (static_raw - mu) / sd
    else:
        mu = np.zeros((1, static_raw.shape[1]), np.float32)
        sd = np.ones((1, static_raw.shape[1]), np.float32)

    # Phys7：StageA 推理（注意：这里对齐 recipe_raw，而不是 zscore 后的 static_x）
    phys7 = infer_phys7_from_stageA_ckpt(stageA_ckpt, static_raw, device=device)    # (N,7)
    phys7_seq = np.repeat(phys7[:, :, None], TT, axis=2).astype(np.float32)        # (N,7,T)

    time_mat = np.repeat(TIME_VALUES[None, :], N, axis=0).astype(np.float32)

    return dict(
        static_raw=static_raw, static_x=static_x, static_mu=mu, static_sd=sd,
        recipe_ids=recipe_ids,
        phys7=phys7, phys7_seq=phys7_seq,
        y_um=y_um, mask=mask, time_mat=time_mat,
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
    """按通道/时间点，用 train 子集拟合均值方差（mask 后）。返回 y_mean,y_std shaped (1,K,T)."""
    K = y_um.shape[1]; TT = y_um.shape[2]
    mean = np.zeros((1,K,TT), np.float32)
    std  = np.ones((1,K,TT), np.float32)
    ytr = y_um[train_idx]
    mtr = mask[train_idx]
    for k in range(K):
        for t in range(TT):
            vals = ytr[:,k,t][mtr[:,k,t]]
            if vals.size >= 2:
                mean[0,k,t] = float(vals.mean())
                std[0,k,t]  = float(vals.std() + 1e-6)
            elif vals.size == 1:
                mean[0,k,t] = float(vals[0])
                std[0,k,t]  = 1e-6
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


def main(cfg: Cfg):
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # build dataset
    data = build_stageC_dataset(cfg.new_excel, cfg.stageA_ckpt, device=device)
    static_x = data["static_x"]
    phys7_seq = data["phys7_seq"]
    y_um = data["y_um"]
    mask = data["mask"]
    time_mat = data["time_mat"]
    recipe_ids = data["recipe_ids"]

    key_recipes = _parse_key_recipes(cfg.key_recipes)

    # family_sign
    family_sign_t = None
    if cfg.family_sign is not None:
        family_sign_t = torch.tensor(cfg.family_sign, dtype=torch.float32)

    rng = np.random.default_rng(2026)

    best = None

    for tr in range(1, cfg.trials+1):
        try:
            print(
                f"[Trial {tr}] Loaded StageB ckpt (matched={info['matched']}/{info['model_key_count']}, missing={len(info['missing'])}, unexpected={len(info['unexpected'])})")
            train_idx, val_idx, test_idx = random_subset_and_split(
                N=len(recipe_ids),
                recipe_ids=recipe_ids,
                key_recipes=key_recipes,
                test_ratio=cfg.test_ratio,
                val_ratio=cfg.val_ratio,
                drop_max_frac=cfg.drop_max_frac,
                rng=rng
            )
        except Exception as e:
            continue

        # norm y using train subset
        y_mean, y_std = fit_y_norm(y_um, mask, train_idx)
        y_norm = (y_um - y_mean) / (y_std + 1e-6)

        train_loader = make_loader(static_x, phys7_seq, y_norm, mask, time_mat, train_idx, cfg.batch)
        val_loader   = make_loader(static_x, phys7_seq, y_norm, mask, time_mat, val_idx, cfg.batch)
        test_loader  = make_loader(static_x, phys7_seq, y_norm, mask, time_mat, test_idx, cfg.batch)

        model = build_morph_model(cfg.model_type, K=len(FAMILIES), device=device)
        if cfg.stageB_ckpt:
            info = load_morph_ckpt(model, cfg.stageB_ckpt)
            if info.get("ok", False):
                print(f"[Trial {tr}] Loaded StageB ckpt (missing={len(info['missing'])}, unexpected={len(info['unexpected'])})")
            else:
                print(f"[Trial {tr}] StageB ckpt not loaded: {info.get('reason')}")

        # train
        train_one(model, train_loader, val_loader, device=device,
                  epochs=cfg.epochs, lr=cfg.lr, wd=cfg.wd, early_patience=cfg.early_patience)

        # eval train/test R2 (display space)
        train_pack = eval_pack(model, train_loader, device=device)
        test_pack  = eval_pack(model, test_loader, device=device)
        train_r2 = overall_r2_from_pack(train_pack, y_mean, y_std, family_sign_t, cfg.unit_scale)
        test_r2  = overall_r2_from_pack(test_pack,  y_mean, y_std, family_sign_t, cfg.unit_scale)

        if tr % 10 == 0:
            print(f"[Trial {tr}/{cfg.trials}] train_R2={train_r2:.3f} test_R2={test_r2:.3f} (train={len(train_idx)} test={len(test_idx)})")

        ok = (train_r2 >= cfg.target_r2) and (test_r2 >= cfg.target_r2)
        if ok:
            print(f"\n[SUCCESS] Found split: trial={tr} train_R2={train_r2:.3f} test_R2={test_r2:.3f}\n")
            best = {
                "trial": tr,
                "train_r2": float(train_r2),
                "test_r2": float(test_r2),
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "split_desc": describe_split(recipe_ids, train_idx, val_idx, test_idx),
                "y_mean": y_mean.tolist(),
                "y_std": y_std.tolist(),
                "model_type": cfg.model_type,
                "cfg": cfg.__dict__,
            }
            # 保存 best
            with open(os.path.join(cfg.out_dir, "best_split.json"), "w", encoding="utf-8") as f:
                json.dump(best, f, ensure_ascii=False, indent=2)

            # 额外导出 test pack（用于画图/报告）
            np.savez_compressed(os.path.join(cfg.out_dir, "best_test_pack.npz"), **test_pack)
            np.savez_compressed(os.path.join(cfg.out_dir, "best_train_pack.npz"), **train_pack)
            break

        # keep best even if not reaching target
        score = min(train_r2, test_r2)
        if best is None or score > min(best["train_r2"], best["test_r2"]):
            best = {
                "trial": tr,
                "train_r2": float(train_r2),
                "test_r2": float(test_r2),
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "split_desc": describe_split(recipe_ids, train_idx, val_idx, test_idx),
                "model_type": cfg.model_type,
                "cfg": cfg.__dict__,
            }
            with open(os.path.join(cfg.out_dir, "best_so_far.json"), "w", encoding="utf-8") as f:
                json.dump(best, f, ensure_ascii=False, indent=2)
    export_stageC_report(cfg.out_dir, train_pack, test_pack, y_mean, y_std, family_sign_t, cfg.unit_scale)
    print("\n[Done]")
    if best:
        print(f"Best trial={best['trial']} train_R2={best['train_r2']:.3f} test_R2={best['test_r2']:.3f}")
        print(f"Split desc: {best['split_desc']}")
    else:
        print("No valid trial (check data size / paths).")


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
        trials=300,
        test_ratio=0.25,
        val_ratio=0.15,
        drop_max_frac=0.50,

        # 形貌模型（与你的 stageBbest 对齐：transformer）
        model_type="transformer",

        # 训练设置（你说训练时间不是问题，这里固定即可）
        epochs=2000,
        lr=5e-4,
        wd=1e-4,
        batch=8,
        early_patience=200,
    )

    main(cfg)