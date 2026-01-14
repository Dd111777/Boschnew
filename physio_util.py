# -*- coding: utf-8 -*-
"""
physio_util.py
统一的工具库（不依赖旧项目）：
- 列名鲁棒解析、小写去空格
- Excel 读取：旧表（8输入+物理时序+形貌标签）、新表（8输入+稀疏形貌）
- 规范化：静态 mean/std；目标标准化可选
- 家族/时间轴：统一为 families=["zmin","h0","h1","d0","d1","w"], times=["1".."9","9_2"]
- 展示空间转换：单位 μm→nm、家族符号、全局取反、非负裁剪
- 指标：MSE/RMSE/R2/MAE/MAPE，按 (B,K,T) + mask
- 导出：predictions.xlsx（长表）、metrics.xlsx（K×T 网格）、summary.txt、manifest.json
- 作图：真假散点、残差直方、按时间/家族合并散点、热力图
"""

import os, json, re, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import torch
from torch.utils.data import TensorDataset, DataLoader

# ---------------- 基本配置 ----------------
FAMILIES = ["zmin", "h0", "h1", "d0", "d1", "w"]
TIME_LIST = ["1","2","3","4","5","6","7","8","9","9_2"]
F2IDX = {f:i for i,f in enumerate(FAMILIES)}
T2IDX = {t:i for i,t in enumerate(TIME_LIST)}

def set_seed(seed=42):
    import random
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

def _canon(s: str) -> str:
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("（", "(").replace("）", ")")
    return s

def _pick_one(df_cols: List[str], candidates: List[str]):
    cols_c = {c: _canon(c) for c in df_cols}
    for c in df_cols:
        v = cols_c[c]
        for pat in candidates:
            if pat in v:
                return c
    return None

def _find_series(df_cols: List[str], prefix_norm: str) -> Tuple[List[str], List[int]]:
    cols_c = {c: _canon(c) for c in df_cols}
    pairs = []
    for c in df_cols:
        v = cols_c[c]
        if v.startswith(prefix_norm):
            m = re.search(r"_(\d+)(?:_(\d+))?$", v)
            if m:
                main = int(m.group(1)); sub = int(m.group(2)) if m.group(2) else 0
                order = main * 10 + sub
                pairs.append((order, c))
    pairs.sort()
    if not pairs:
        raise RuntimeError(f"未找到以 {prefix_norm}_* 开头的列")
    cols_sorted = [c for _, c in pairs]
    orders = [o for o, _ in pairs]
    return cols_sorted, orders

# --------------- A：旧表→物理数据集 ----------------
def excel_to_physics_dataset(excel_path: str, sheet_name=None):
    import pandas as pd
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    cols = list(df.columns)

    # 8 输入列匹配
    key_alias = {
        "apc": ["apc"],
        "source_rf": ["source_rf", "sourcerf", "rfsource", "e2"],
        "lf_rf": ["lf_rf", "lfrf", "bias"],
        "sf6": ["sf6"],
        "c4f8": ["c4f8"],
        "dep_time": ["deptime", "dep_time", "depositiontime"],
        "etch_time": ["etchtime", "etch_time"],
    }
    def _pick_one(df_cols, candidates):
        cols_c = {c: _canon(c) for c in df_cols}
        for c in df_cols:
            v = cols_c[c]
            for pat in candidates:
                if pat in v:
                    return c
        return None

    static_keys = [
        _pick_one(cols, key_alias["apc"]),
        _pick_one(cols, key_alias["source_rf"]),
        _pick_one(cols, key_alias["lf_rf"]),
        _pick_one(cols, key_alias["sf6"]),
        _pick_one(cols, key_alias["c4f8"]),
        _pick_one(cols, key_alias["dep_time"]),
        _pick_one(cols, key_alias["etch_time"]),
    ]
    if not all(static_keys):
        raise RuntimeError(f"8个输入列不全：{static_keys}")

    import numpy as np, torch
    static_np = df[static_keys].to_numpy(dtype=np.float32)  # (N,8)
    mean = static_np.mean(axis=0, keepdims=True)
    std  = static_np.std(axis=0, keepdims=True) + 1e-6
    static_norm = (static_np - mean) / std                   # (N,8)

    # 物理时序列
    fcols, forders = _find_series(cols, "f_flux")
    icols, iorders = _find_series(cols, "ion_flux")
    if forders != iorders:
        raise RuntimeError("F_Flux 与 Ion_Flux 的时间后缀不一致")
    T = len(fcols)

    f_np = df[fcols].to_numpy(dtype=np.float32)  # (N,T)
    i_np = df[icols].to_numpy(dtype=np.float32)  # (N,T)
    mask_f = ~np.isnan(f_np)
    mask_i = ~np.isnan(i_np)
    f_np[np.isnan(f_np)] = 0.0
    i_np[np.isnan(i_np)] = 0.0

    phys_np   = np.stack([f_np, i_np], axis=1)            # (N,2,T)
    phys_mask = np.stack([mask_f, mask_i], axis=1)        # (N,2,T)

    # 时间刻度：1..T，并复制到每个样本，保证第一维=N
    time_values = np.arange(1, T+1, dtype=np.float32)     # (T,)
    time_batch  = np.tile(time_values[None, :], (static_np.shape[0], 1))  # (N,T)

    dataset = TensorDataset(
        torch.from_numpy(static_norm),                              # (N,8)
        torch.from_numpy(phys_np.astype(np.float32)),               # (N,2,T)
        torch.from_numpy(phys_mask.astype(np.bool_)),               # (N,2,T)
        torch.from_numpy(time_batch.astype(np.float32))             # (N,T) ★修复点
    )
    meta = {
        "T": T,
        "time_values": time_values,  # 仍保留单份 1D 刻度给绘图/导出
        "norm_static": {
            "mean": torch.from_numpy(mean.astype(np.float32)),
            "std":  torch.from_numpy(std.astype(np.float32))
        }
    }
    return dataset, meta


# --------------- B：旧表→形貌数据集 ----------------
def excel_to_morph_dataset_from_old(excel_path: str, sheet_name=None):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    cols = list(df.columns)
    # 8输入
    key_alias = {
        "temp": ["temp", "temperature"], "apc": ["apc"],
        "source_rf": ["source_rf","sourcerf","rfsource","e2"],
        "lf_rf": ["lf_rf","lfrf","bias"], "sf6": ["sf6"],
        "c4f8": ["c4f8"], "dep_time": ["deptime","dep_time"],
        "etch_time": ["etchtime","etch_time"],
    }
    static_keys = [
        _pick_one(cols, key_alias["apc"]),
        _pick_one(cols, key_alias["source_rf"]),
        _pick_one(cols, key_alias["lf_rf"]),
        _pick_one(cols, key_alias["sf6"]),
        _pick_one(cols, key_alias["c4f8"]),
        _pick_one(cols, key_alias["dep_time"]),
        _pick_one(cols, key_alias["etch_time"]),
    ]
    if not all(static_keys): raise RuntimeError(f"8个输入列不全：{static_keys}")
    static_np = df[static_keys].to_numpy(np.float32)
    mean = static_np.mean(axis=0, keepdims=True)
    std  = static_np.std(axis=0, keepdims=True) + 1e-6
    static_norm = (static_np - mean) / std

    # 物理时序
    fcols, forders = _find_series(cols, "f_flux")
    icols, iorders = _find_series(cols, "ion_flux")
    if forders != iorders: raise RuntimeError("F_Flux 与 Ion_Flux 的时间后缀不一致")
    T = len(fcols)
    f_np = df[fcols].to_numpy(np.float32)
    i_np = df[icols].to_numpy(np.float32)
    f_np[np.isnan(f_np)] = 0.0; i_np[np.isnan(i_np)] = 0.0
    phys_seq = np.stack([f_np, i_np], axis=1)  # (N,2,T)

    # 形貌标签
    targets = np.zeros((len(df), len(FAMILIES), T), np.float32)
    mask    = np.zeros_like(targets, dtype=np.bool_)
    cancols = {c:_canon(c) for c in cols}
    # 支持 zmin_*、h0_*、h1_*、d0_*、d1_*、w1..w9 / w9_2
    for c in cols:
        v = cancols[c]
        # w（无下划线家族）
        m_w = re.match(r"^w(\d+)(?:_(\d+))?$", v)
        if m_w:
            tid = m_w.group(1) if not m_w.group(2) else f"{m_w.group(1)}_{m_w.group(2)}"
            if tid in T2IDX:
                colv = df[c].to_numpy(np.float32)
                eps = 1e-12
                ok = (~np.isnan(colv)) & (np.abs(colv) > eps)
                targets[ok, F2IDX["w"], T2IDX[tid]] = colv[ok]
                mask[ok,    F2IDX["w"], T2IDX[tid]] = True
            continue
        # 其他家族
        m = re.match(r"^(zmin|h0|h1|d0|d1)_(\d+)(?:_(\d+))?$", v)
        if m:
            fam = m.group(1)
            tid = m.group(2) if not m.group(3) else f"{m.group(2)}_{m.group(3)}"
            if fam in F2IDX and tid in T2IDX:
                colv = df[c].to_numpy(np.float32)
                eps = 1e-12
                ok = (~np.isnan(colv)) & (np.abs(colv) > eps)
                targets[ok, F2IDX[fam], T2IDX[tid]] = colv[ok]
                mask[ok,    F2IDX[fam], T2IDX[tid]] = True

    time_values = np.arange(1, T + 1, dtype=np.float32)  # (T,)
    time_mat = np.tile(time_values[None, :], (len(df), 1))  # (N, T)
    # === 逐 family 标准化（仅用于训练空间；展示/导出前会自动反归一化）===
    K = len(FAMILIES)
    fam_mean = np.zeros(K, np.float32)
    fam_std = np.ones(K, np.float32)
    for k in range(K):
        vals = targets[:, k, :][mask[:, k, :]]
        if vals.size > 0:
            m = float(vals.mean())
            s = float(vals.std() + 1e-6)
        else:
            m, s = 0.0, 1.0
        fam_mean[k] = m
        fam_std[k] = s
        # 对整张 (N,T) 做线性变换（缺失位置对训练无影响，反归一化时保持一致）
        targets[:, k, :] = (targets[:, k, :] - m) / s

    # 在 meta 中保存统计量，供 Stage B 反归一化
    meta_norm_target = {
        "mean": torch.from_numpy(fam_mean),
        "std": torch.from_numpy(fam_std),
    }
    ds = TensorDataset(
        torch.from_numpy(static_norm),  # (N, 8)
        torch.from_numpy(phys_seq),  # (N, 2, T)
        torch.from_numpy(targets),  # (N, K, T)
        torch.from_numpy(mask),  # (N, K, T)
        torch.from_numpy(time_mat)  # (N, T)  ✅ 对齐了
    )
    meta = {
        "T": T,
        "time_values": time_values,  # 这里仍然保留 (T,) 给可视化/导出使用
        "families": FAMILIES,
        "norm_target": meta_norm_target,
        "norm_static": {
            "mean": torch.from_numpy(mean.astype(np.float32)),
            "std": torch.from_numpy(std.astype(np.float32)),

        },
    }
    return ds, meta

# --------------- C：新表→稀疏形貌监督 ----------------
def load_new_excel_as_sparse_morph(new_excel: str, height_family="h1") -> List[Dict]:
    df = pd.read_excel(new_excel)
    def col_like(name: str) -> str:
        can = _canon(name)
        for c in df.columns:
            if _canon(c) == can: return c
        raise KeyError(f"缺少列: {name}")

    k8 = ["APC（E2步骤）","source_RF（E2步骤）","LF_RF（E2步骤）",
          "SF6（E2步骤）","C4F8（DEP步骤）","DEP time","etch time"]
    static = df[[col_like(k) for k in k8]].to_numpy(np.float32)
    nm2um = 1/1000.0
    recs = []
    for i in range(len(df)):
        tg: Dict[Tuple[str,str], float] = {}
        # Zmin_9_2
        try:
            total_depth_nm = float(df.iloc[i][col_like("总深度")])
            tg[("zmin","9_2")] = -abs(total_depth_nm*nm2um)
        except Exception: pass
        # w*
        mapping_w = {("w","1"):"开口处CD",("w","3"):"第三个scallops的宽度",
                     ("w","5"):"第五个scallops的宽度",("w","9"):"最后一个scallops的宽度"}
        for (fam,tid), cname in mapping_w.items():
            try:
                v = float(df.iloc[i][col_like(cname)])*nm2um
                tg[(fam,tid)] = v
            except Exception: pass
        # h*
        mapping_h = {(height_family,"3"):"第三个scallops的高度",
                     (height_family,"5"):"第五个scallops的高度",
                     (height_family,"9"):"最后一个scallops的高度"}
        for (fam,tid), cname in mapping_h.items():
            try:
                v = float(df.iloc[i][col_like(cname)])*nm2um
                tg[(fam,tid)] = v
            except Exception: pass
        # d1_*
        mapping_d = {("d1","3"):"第三个scallops的深度",
                     ("d1","5"):"第五个scallops的深度",
                     ("d1","9"):"最后一个scallops的深度"}
        for (fam,tid), cname in mapping_d.items():
            try:
                v = float(df.iloc[i][col_like(cname)])*nm2um
                tg[(fam,tid)] = v
            except Exception: pass
        bottle_flag = None
        try:
            bottle_col = col_like("瓶型")      # 按你实际列名改
            v = df.iloc[i][bottle_col]
            bottle_flag = int(v)              # 0/1 或者别的
        except Exception:
            bottle_flag = None                # 没有这一列或解析失败，就当没有

        recs.append({
            "static": static[i],
            "targets": tg,
            "bottle_flag": bottle_flag,       # 新增字段
        })


    return recs


def build_sparse_batch(recs: List[Dict], norm_static_mean, norm_static_std, time_values):
    B = len(recs);
    K = len(FAMILIES);
    T = len(TIME_LIST)
    targets = np.zeros((B, K, T), np.float32)
    mask = np.zeros((B, K, T), np.bool_)
    static = np.stack([r["static"] for r in recs], 0).astype(np.float32)

    # 转换Tensor为numpy
    if isinstance(norm_static_mean, torch.Tensor):
        mean_np = norm_static_mean.cpu().numpy()
    else:
        mean_np = np.asarray(norm_static_mean)

    if isinstance(norm_static_std, torch.Tensor):
        std_np = norm_static_std.cpu().numpy()
    else:
        std_np = np.asarray(norm_static_std)

    # 处理维度不匹配（7 vs 8）
    if static.shape[1] != len(mean_np):
        if static.shape[1] < len(mean_np):
            # 用0填充缺失的特征
            padding = np.zeros((static.shape[0], len(mean_np) - static.shape[1]), dtype=np.float32)
            static = np.concatenate([static, padding], axis=1)
        else:
            # 截断多余的特征
            static = static[:, :len(mean_np)]

    static_norm = (static - mean_np) / (std_np + 1e-8)

    for b, r in enumerate(recs):
        for (fam, tid), val in r["targets"].items():
            if fam in F2IDX and tid in T2IDX:
                targets[b, F2IDX[fam], T2IDX[tid]] = float(val)
                mask[b, F2IDX[fam], T2IDX[tid]] = True

    return (torch.from_numpy(static_norm),
            torch.from_numpy(targets),
            torch.from_numpy(mask),
            torch.from_numpy(np.asarray(time_values, np.float32)))

# --------------- 展示空间转换与指标 ----------------
def transform_for_display(yhat: torch.Tensor,
                          ytrue: torch.Tensor,
                          family_sign: torch.Tensor | None = None,
                          clip_nonneg: bool = True,
                          nonneg_families: list[int] | None = None,
                          unit_scale: float | list | dict | torch.Tensor | None = None,
                          unit_offset: float | list | dict | torch.Tensor | None = None,
                          flip_sign: bool = False,
                          min_display_value: float | None = None):
    """
    将 (B,K,T) 的 yhat/ytrue 从训练空间变换到展示空间：
      - family_sign: (K,) 的 +1/-1 逐 family 符号翻转
      - flip_sign:   全局取反（在 family_sign 之后应用）
      - unit_scale / unit_offset: 统一或逐 family 的单位缩放/偏置（如 um->nm ×1000）
      - clip_nonneg + nonneg_families: 仅对指定 family 做非负裁剪
      - min_display_value: 若给定，则对所有 family 做全局下截断（展示下限）
    """

    def _to_K_broadcast(x: torch.Tensor, v, K: int, name: str):
        if v is None:
            return None
        if isinstance(v, (int, float)):
            t = torch.tensor([float(v)] * K, device=x.device, dtype=x.dtype)
        elif isinstance(v, torch.Tensor):
            if v.dim() == 1 and v.numel() == K:
                t = v.to(device=x.device, dtype=x.dtype)
            else:
                raise ValueError(f"{name} tensor must be (K,), got {tuple(v.shape)}")
        elif isinstance(v, dict):
            lst = [float(v.get(i, 1.0 if name == 'scale' else 0.0)) for i in range(K)]
            t = torch.tensor(lst, device=x.device, dtype=x.dtype)
        else:
            import numpy as np
            arr = np.asarray(v, dtype=float).reshape(-1)
            if arr.size != K:
                raise ValueError(f"{name} length must be K={K}, got {arr.size}")
            t = torch.from_numpy(arr.astype('float32')).to(device=x.device, dtype=x.dtype)
        return t.view(1, -1, 1)  # (1,K,1)

    def _tx(x: torch.Tensor) -> torch.Tensor:
        K = x.shape[1]

        # 1) 逐 family 符号
        if family_sign is not None:
            fs = torch.as_tensor(family_sign, device=x.device, dtype=x.dtype).view(1, -1, 1)
            x = x * fs

        # 2) 单位缩放 / 偏置
        sc = _to_K_broadcast(x, unit_scale,  K, name='scale')
        of = _to_K_broadcast(x, unit_offset, K, name='offset')
        if sc is not None:
            x = x * sc
        if of is not None:
            x = x + of

        # 3) 全局取反（在 family_sign 之后）
        if flip_sign:
            x = -x

        # 4) 非负裁剪（仅指定 family）
        if clip_nonneg and nonneg_families:
            mask_bool = torch.zeros(K, dtype=torch.bool, device=x.device)
            for k in nonneg_families:
                if 0 <= k < K:
                    mask_bool[k] = True
            nonneg_mask = mask_bool.view(1, -1, 1)
            x = torch.where(nonneg_mask, torch.clamp_min(x, 0.0), x)

        # 5) 全局下限截断（所有 family）
        if min_display_value is not None:
            x = torch.clamp_min(x, float(min_display_value))

        return x

    return _tx(yhat.clone()), _tx(ytrue.clone())


def metrics(yhat_disp, ytrue_disp, mask):
    """
    输入为展示空间（nm）的 (B,K,T)
    返回 dict：每个指标一个 (K,T) numpy
    """
    yh = yhat_disp.detach().cpu().numpy()
    yt = ytrue_disp.detach().cpu().numpy()
    m  = mask.detach().cpu().numpy().astype(bool)
    K,T = yh.shape[1], yh.shape[2]
    out = {}
    mse = np.zeros((K,T)); mae=np.zeros((K,T)); mape=np.zeros((K,T)); r2=np.zeros((K,T)); rmse=np.zeros((K,T))
    for k in range(K):
        for t in range(T):
            sel = m[:,k,t]
            if not np.any(sel):
                mse[k,t]=mae[k,t]=mape[k,t]=rmse[k,t]=np.nan; r2[k,t]=np.nan
                continue
            a = yh[sel,k,t]; b = yt[sel,k,t]
            diff = a-b
            mse[k,t]  = np.mean(diff**2)
            rmse[k,t] = np.sqrt(mse[k,t])
            mae[k,t]  = np.mean(np.abs(diff))
            # MAPE（对0做保护）
            denom = np.clip(np.abs(b), 1e-8, None)
            mape[k,t]= np.mean(np.abs(diff)/denom)*100.0
            # R2
            ss_res = np.sum(diff**2)
            ss_tot = np.sum((b - np.mean(b))**2) + 1e-8
            r2[k,t] = 1.0 - ss_res/ss_tot
    out["MSE"]=mse; out["RMSE"]=rmse; out["MAE"]=mae; out["MAPE"]=mape; out["R2"]=r2
    return out

# --------------- 导出：Excel & 图 ----------------
def _ensure(p): os.makedirs(p, exist_ok=True)

def export_predictions_longtable(yhat_disp, ytrue_disp, mask, families, time_values_1d, out_dir, filename="predictions.xlsx"):
    _ensure(out_dir)
    yh = yhat_disp.detach().cpu().numpy()
    yt = ytrue_disp.detach().cpu().numpy()
    m  = mask.detach().cpu().numpy().astype(bool)
    rows=[]
    for b in range(yh.shape[0]):
        for ki,f in enumerate(families):
            for ti,tv in enumerate(time_values_1d):
                rows.append({
                    "sample": b, "family": f, "time": str(tv),
                    "y_true": yt[b,ki,ti] if m[b,ki,ti] else np.nan,
                    "y_pred": yh[b,ki,ti]
                })
    pd.DataFrame(rows).to_excel(os.path.join(out_dir, filename), index=False)

def export_metrics_grid(mts: Dict[str,np.ndarray], families, time_values_1d, out_dir, filename="metrics.xlsx"):
    _ensure(out_dir)
    with pd.ExcelWriter(os.path.join(out_dir, filename)) as xw:
        for k,grid in mts.items():
            df = pd.DataFrame(grid, index=families, columns=[str(t) for t in time_values_1d])
            df.to_excel(xw, sheet_name=k)

def write_summary_txt(mts: Dict[str,np.ndarray], families, time_values_1d, out_dir):
    _ensure(out_dir)
    p = os.path.join(out_dir, "summary.txt")
    with open(p,"w",encoding="utf-8") as f:
        f.write("== Summary (display space) ==\n")
        for name in ["R2","RMSE","MAE","MAPE","MSE"]:
            if name in mts:
                arr = mts[name]
                f.write(f"\n[{name}]\n")
                for i,fam in enumerate(families):
                    vals = ", ".join([f"{arr[i,j]:.4g}" if not np.isnan(arr[i,j]) else "NaN" for j in range(arr.shape[1])])
                    f.write(f"{fam}: {vals}\n")

def save_manifest(out_dir):
    _ensure(out_dir)
    with open(os.path.join(out_dir,"manifest.json"),"w",encoding="utf-8") as f:
        json.dump({"ok":True},f,ensure_ascii=False,indent=2)

def heatmap(grid, families, time_values_1d, title, out_png):
    plt.figure(figsize=(10,4.2))
    plt.imshow(grid, aspect='auto')
    plt.colorbar()
    plt.yticks(range(len(families)), families)
    plt.xticks(range(len(time_values_1d)), [str(t) for t in time_values_1d])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()

def parity_scatter(yhat_disp, ytrue_disp, mask, out_png, title):
    yh = yhat_disp.detach().cpu().numpy().reshape(-1)
    yt = ytrue_disp.detach().cpu().numpy().reshape(-1)
    m  = mask.detach().cpu().numpy().reshape(-1).astype(bool)
    yh = yh[m]; yt = yt[m]
    plt.figure(figsize=(4.2,4.2))
    plt.scatter(yt, yh, s=8, alpha=0.6)
    lo = min(np.min(yt), np.min(yh)); hi = max(np.max(yt), np.max(yh))
    plt.plot([lo,hi],[lo,hi],'--')
    plt.xlabel("True (display)"); plt.ylabel("Pred (display)")
    plt.title(title); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def residual_hist(yhat_disp, ytrue_disp, mask, out_png, title):
    yh = yhat_disp.detach().cpu().numpy().reshape(-1)
    yt = ytrue_disp.detach().cpu().numpy().reshape(-1)
    m  = mask.detach().cpu().numpy().reshape(-1).astype(bool)
    diff = (yh-yt)[m]
    plt.figure(figsize=(4.6,3.6))
    plt.hist(diff, bins=40)
    plt.title(title); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

def excel_to_phys7_dataset_from_iedf(excel_path: str, sheet_name=None,
                                    case_id_col: str = "input",
                                    iedf_root: str = r"D:\BaiduNetdiskDownload\TSV",
                                    dropna: bool = True):
    """
    输入：case.xlsx（含 recipe 7维 + case_id）
    标签：从 IEDF 文件夹在线计算 Phys7（7维）
    输出：TensorDataset(x_norm, y_norm, y_mask) + meta
    """
    import numpy as np, torch
    import pandas as pd
    from torch.utils.data import TensorDataset

    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    cols = list(df.columns)
    if case_id_col not in cols:
        raise KeyError(f"case_id_col='{case_id_col}' 不存在，现有列：{cols}")

    # recipe7 列（复用你 excel_to_physics_dataset 的匹配规则）
    key_alias = {
        "apc": ["apc"],
        "source_rf": ["source_rf", "sourcerf", "rfsource", "e2"],
        "lf_rf": ["lf_rf", "lfrf", "bias"],
        "sf6": ["sf6"],
        "c4f8": ["c4f8"],
        "dep_time": ["deptime", "dep_time", "depositiontime"],
        "etch_time": ["etchtime", "etch_time"],
    }
    static_keys = [
        _pick_one(cols, key_alias["apc"]),
        _pick_one(cols, key_alias["source_rf"]),
        _pick_one(cols, key_alias["lf_rf"]),
        _pick_one(cols, key_alias["sf6"]),
        _pick_one(cols, key_alias["c4f8"]),
        _pick_one(cols, key_alias["dep_time"]),
        _pick_one(cols, key_alias["etch_time"]),
    ]
    if not all(static_keys):
        raise RuntimeError(f"recipe7 输入列不全：{static_keys}")

    X = df[static_keys].to_numpy(np.float32)
    x_mean = X.mean(axis=0, keepdims=True)
    x_std  = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - x_mean) / x_std

    phys_cols = [
        "logGamma_SF6_tot","pF_SF6","spread_SF6","qskew_SF6",
        "logGamma_C4F8_tot","rho_C4F8","spread_C4F8"
    ]

    Y = np.full((len(df), len(phys_cols)), np.nan, np.float32)
    for i, cid0 in enumerate(df[case_id_col].astype(str).tolist()):
        cid = normalize_case_id(cid0)
        feat = compute_phys7_for_case(cid, iedf_root=iedf_root)
        for j, k in enumerate(phys_cols):
            Y[i, j] = np.float32(feat.get(k, np.nan))

    y_mask = np.isfinite(Y)
    if dropna:
        keep = y_mask.all(axis=1)
        df = df.loc[keep].reset_index(drop=True)
        Xn = Xn[keep]
        Y  = Y[keep]
        y_mask = y_mask[keep]

    # y 标准化（列内）
    y_mean = np.zeros((1, Y.shape[1]), np.float32)
    y_std  = np.ones((1, Y.shape[1]), np.float32)
    Yn = Y.copy()
    for j in range(Y.shape[1]):
        vals = Y[:, j][y_mask[:, j]]
        m = float(vals.mean()) if vals.size else 0.0
        s = float(vals.std() + 1e-6) if vals.size else 1.0
        y_mean[0, j] = m
        y_std[0, j] = s
        Yn[:, j] = (Y[:, j] - m) / s

    ds = TensorDataset(
        torch.from_numpy(Xn),
        torch.from_numpy(Yn),
        torch.from_numpy(y_mask.astype(np.bool_)),
    )
    meta = {
        "recipe_cols": static_keys,
        "phys7_cols": phys_cols,
        "norm_x": {"mean": torch.from_numpy(x_mean), "std": torch.from_numpy(x_std)},
        "norm_y": {"mean": torch.from_numpy(y_mean), "std": torch.from_numpy(y_std)},
    }
    return ds, meta, df
