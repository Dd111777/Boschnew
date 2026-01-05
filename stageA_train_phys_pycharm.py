# -*- coding: utf-8 -*-
"""
StageA (Phys7 from IEDF) — recipe(7) -> phys7(7x1)

目标：
- 用 IEDF 提取的 Phys7 作为 StageA 的“物理信息”标签
- 输入仍是原来的 7 维 recipe: [APC, source_RF, LF_RF, SF6, C4F8, DEP time, etch time]
- 输出是 7 个物理降维特征（T=1），与 case.xlsx 按 case_id 一一对应
- 保留 Transformer/GRU/MLP baseline，对比试验和论文图表导出（parity/residual/heatmap/summary/pred_table）

注意：
- 依赖 extract_phys7_from_iedf.py（你已经有）
- 需要 phys_model.py 支持 out_dim 参数（见下面 patch）
"""

import os
import re
import csv
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset

import matplotlib.pyplot as plt

from physio_util import (
    set_seed,
    metrics,
    export_predictions_longtable,
    export_metrics_grid,
    write_summary_txt,
    heatmap,
    parity_scatter,
    residual_hist,
)

from phys_model import PhysicsSeqPredictor, PhysicsMLPBaseline, PhysicsGRUBaseline
import extract_phys7_from_iedf as iedf


# -------------------- Phys7 定义（固定 7 维） --------------------
FAMILIES = [
    "logGamma_SF6_tot",  # log10(总离子通量), SF6/sheath2 聚合
    "pF_SF6",            # F+ 占比
    "spread_SF6",        # (E90-E10)/E50
    "qskew_SF6",         # (E90+E10-2E50)/(E90-E10)
    "logGamma_C4F8_tot", # log10(总离子通量), C4F8/sheath1 聚合
    "rho_C4F8",          # log10( Gamma(CF3+)/Gamma(C2F3+) )
    "spread_C4F8",       # (E90-E10)/E50
]


# -------------------- 配置 --------------------
class Cfg:
    # 数据
    case_excel = r"D:\PycharmProjects\Bosch\case.xlsx"
    sheet_name = "case"
    case_id_col = "input"   # 你表里是 input: cas1/cas2...
    iedf_root = r"D:\BaiduNetdiskDownload\TSV"

    # 输出目录
    save_dir = "./runs_stageA_phys7"
    seed = 42
    batch = 64
    max_epochs = 200
    val_ratio = 0.1

    # 模型
    model_type = "transformer"  # 默认主模型 tf；可选 "gru","mlp"
    d_model = 64
    nhead = 4
    num_layers = 2
    dim_ff = 128
    dropout = 0.1

    # baseline 超参
    mlp_hidden = 128
    mlp_layers = 3
    gru_hidden = 128
    gru_layers = 1

    # 优化
    lr = 1e-3
    weight_decay = 1e-3
    clip_grad_norm = 1.0
    warmup_epochs = 10
    use_cosine = True

    # 输出归一化（只影响 loss，不影响导出物理量）
    use_output_norm = True

    # CV / baseline 对比
    # cv_model_types = ["transformer", "mlp", "gru"]
    # multi_split_seeds = [0, 1, 2, 3, 4]
    cv_model_types = ["transformer"]
    multi_split_seeds = [0, 1, 2, 3, 4]




# -------------------- 小工具 --------------------
def _canon(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.replace("（", "(").replace("）", ")")
    return s


def _pick_one(cols, candidates):
    cc = {c: _canon(c) for c in cols}
    for c in cols:
        v = cc[c]
        for pat in candidates:
            if pat in v:
                return c
    return None


def _norm_case_id(cid: str) -> str:
    cid = str(cid).strip()
    if re.fullmatch(r"\d+", cid):
        return f"cas{cid}"
    m = re.fullmatch(r"(?i)case(\d+)", cid)
    if m:
        return f"cas{m.group(1)}"
    return cid


def make_warmup_cosine(optimizer, total_epochs, warmup_epochs, use_cosine=True):
    if not use_cosine:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _compute_y_norm_stats(y: torch.Tensor, mask: torch.Tensor):
    """
    y:   (N, K, T) 这里 T=1
    mask:(N, K, T)
    返回 per-channel mean/std: (K,)
    """
    N, K, T = y.shape
    mean = np.zeros(K, np.float32)
    std = np.ones(K, np.float32)
    y_np = y.detach().cpu().numpy()
    m_np = mask.detach().cpu().numpy().astype(bool)

    for k in range(K):
        vals = y_np[:, k, :][m_np[:, k, :]]
        if vals.size == 0:
            mean[k] = 0.0
            std[k] = 1.0
        else:
            mean[k] = float(vals.mean())
            std[k] = float(vals.std() + 1e-6)
    return torch.from_numpy(mean), torch.from_numpy(std)


# -------------------- 1) 数据集构建：case.xlsx + IEDF -> Phys7 --------------------
def excel_to_phys7_dataset(excel_path: str, sheet_name: str, case_id_col: str, iedf_root: str):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    cols = list(df.columns)

    # 7 维 recipe 输入列匹配（沿用你旧逻辑口味）
    key_alias = {
        "apc": ["apc"],
        "source_rf": ["source_rf", "sourcerf", "rfsource"],
        "lf_rf": ["lf_rf", "lfrf", "bias"],
        "sf6": ["sf6"],
        "c4f8": ["c4f8"],
        "dep_time": ["deptime", "dep_time", "depositiontime"],
        "etch_time": ["etchtime", "etch_time"],
    }
    recipe_cols = [
        _pick_one(cols, key_alias["apc"]),
        _pick_one(cols, key_alias["source_rf"]),
        _pick_one(cols, key_alias["lf_rf"]),
        _pick_one(cols, key_alias["sf6"]),
        _pick_one(cols, key_alias["c4f8"]),
        _pick_one(cols, key_alias["dep_time"]),
        _pick_one(cols, key_alias["etch_time"]),
    ]
    # ✅ 防止重复选列（例如 APC 被选两次）
    if len(set(recipe_cols)) != len(recipe_cols):
        raise KeyError(
            f"recipe7 选列出现重复：{recipe_cols}\n"
            f"请检查 key_alias 是否过宽导致误匹配。"
        )

    if not all(recipe_cols):
        raise KeyError(f"recipe7 列没找全：{recipe_cols}\n现有列：{cols}")

    if case_id_col not in cols:
        raise KeyError(f"case_id_col={case_id_col} 不在表头里。现有列：{cols}")

    # 输入 X：标准化
    X = df[recipe_cols].to_numpy(np.float32)  # (N,7)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mean) / std

    # 输出 Y：Phys7, T=1
    N = len(df)
    K = len(FAMILIES)
    T = 1
    Y = np.full((N, K, T), np.nan, np.float32)

    # 临时改 extract_phys7_from_iedf 的根目录
    old_root = iedf.IEDF_ROOT
    iedf.IEDF_ROOT = iedf_root

    for i in range(N):
        cid = _norm_case_id(df.loc[i, case_id_col])
        files = iedf.read_target_iedf_for_case(cid)

        feat = {k: np.nan for k in FAMILIES}

        # SF6/sheath2
        key_sf6 = ("SF6", "sheath2")
        if key_sf6 in files:
            out = iedf.compute_phys7_from_file(files[key_sf6], "SF6", "sheath2", iedf.TARGETS[key_sf6])
            if out is not None:
                Gtot = out["Gamma_tot"]
                if np.isfinite(Gtot):
                    feat["logGamma_SF6_tot"] = float(np.log10(Gtot + iedf.EPS))
                GF = out["gammas"].get("F_1p", np.nan)
                if np.isfinite(GF) and np.isfinite(Gtot):
                    feat["pF_SF6"] = float(GF / (Gtot + iedf.EPS))
                feat["spread_SF6"] = out["spread"]
                feat["qskew_SF6"] = out["qskew"]

        # C4F8/sheath1（你说 C4F8 没 sheath2，这里严格按 sheath1）
        key_c4 = ("C4F8", "sheath1")
        if key_c4 in files:
            out = iedf.compute_phys7_from_file(files[key_c4], "C4F8", "sheath1", iedf.TARGETS[key_c4])
            if out is not None:
                Gtot = out["Gamma_tot"]
                if np.isfinite(Gtot):
                    feat["logGamma_C4F8_tot"] = float(np.log10(Gtot + iedf.EPS))
                G1 = out["gammas"].get("CF3_1p", np.nan)
                G2 = out["gammas"].get("C2F3_1p", np.nan)
                if np.isfinite(G1) and np.isfinite(G2):
                    feat["rho_C4F8"] = float(np.log10((G1 + iedf.EPS) / (G2 + iedf.EPS)))
                feat["spread_C4F8"] = out["spread"]

        Y[i, :, 0] = np.array([feat[k] for k in FAMILIES], np.float32)

    iedf.IEDF_ROOT = old_root

    mask = np.isfinite(Y)
    Y[np.isnan(Y)] = 0.0

    # time_values / time_mat
    time_values = np.array([1.0], np.float32)  # (T,)
    time_mat = np.tile(time_values[None, :], (N, 1))  # (N,T)

    ds = TensorDataset(
        torch.from_numpy(Xn.astype(np.float32)),          # (N,7)
        torch.from_numpy(Y.astype(np.float32)),           # (N,K,T)
        torch.from_numpy(mask.astype(np.bool_)),          # (N,K,T)
        torch.from_numpy(time_mat.astype(np.float32)),    # (N,T)
    )
    meta = {
        "T": 1,
        "time_values": time_values,
        "families": list(FAMILIES),
        "recipe_cols": recipe_cols,
        "norm_static": {
            "mean": torch.from_numpy(mean.astype(np.float32)),
            "std": torch.from_numpy(std.astype(np.float32)),
        }
    }
    return ds, meta


# -------------------- 2) 模型构建（tf/gru/mlp） --------------------
def build_stageA_model(T: int, out_dim: int, model_type: str):
    mt = str(model_type).lower()
    if mt in ["transformer", "tfm", "tf", "trans"]:
        return PhysicsSeqPredictor(
            d_model=Cfg.d_model,
            nhead=Cfg.nhead,
            num_layers=Cfg.num_layers,
            dim_ff=Cfg.dim_ff,
            dropout=Cfg.dropout,
            T=T,
            out_dim=out_dim,
        )
    if mt in ["mlp", "ffn"]:
        return PhysicsMLPBaseline(
            hidden_dim=Cfg.mlp_hidden,
            num_layers=Cfg.mlp_layers,
            T=T,
            out_dim=out_dim,
        )
    if mt in ["gru", "rnn"]:
        return PhysicsGRUBaseline(
            hidden_dim=Cfg.gru_hidden,
            num_layers=Cfg.gru_layers,
            T=T,
            out_dim=out_dim,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


# -------------------- 3) 训练（单模型多输出） --------------------
def train_stageA_phys7(dataset, meta, model_type: str, out_root: str, split_seed: int):
    os.makedirs(out_root, exist_ok=True)
    set_seed(split_seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T = int(meta["T"])
    K = len(meta["families"])

    N = len(dataset)
    nval = max(1, int(N * Cfg.val_ratio))
    tr_set, va_set = random_split(
        dataset, [N - nval, nval],
        generator=torch.Generator().manual_seed(split_seed)
    )
    tr = DataLoader(tr_set, batch_size=Cfg.batch, shuffle=True)
    va = DataLoader(va_set, batch_size=Cfg.batch, shuffle=False)

    model = build_stageA_model(T=T, out_dim=K, model_type=model_type).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=Cfg.lr, weight_decay=Cfg.weight_decay)
    sch = make_warmup_cosine(opt, Cfg.max_epochs, Cfg.warmup_epochs, use_cosine=Cfg.use_cosine)

    loss_fn = nn.SmoothL1Loss(reduction="none")

    # 输出归一化统计（仅 loss）
    y_mean_t = None
    y_std_t = None
    if getattr(Cfg, "use_output_norm", False):
        # 取训练子集统计
        idx_list = tr_set.indices if hasattr(tr_set, "indices") else list(range(len(tr_set)))
        _, Y_all, M_all, _ = dataset.tensors
        Y_tr = Y_all[idx_list]
        M_tr = M_all[idx_list]
        y_mean, y_std = _compute_y_norm_stats(Y_tr, M_tr)
        y_mean_t = y_mean.to(dev).view(1, K, 1)
        y_std_t = y_std.to(dev).view(1, K, 1)
        with open(os.path.join(out_root, "y_norm.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"mean": y_mean.cpu().tolist(), "std": y_std.cpu().tolist()},
                f, ensure_ascii=False, indent=2
            )

    best = 1e18
    best_path = os.path.join(out_root, "phys7_best.pth")
    hist = {"train": [], "val": []}

    for e in range(1, Cfg.max_epochs + 1):
        model.train()
        s = 0.0
        n = 0

        for Xn, Y, M, tvals in tr:
            Xn = Xn.to(dev)
            Y = Y.to(dev)
            M = M.to(dev)
            tvals = tvals.to(dev)

            pred = model(Xn, tvals)  # (B,K,T=1)

            if y_mean_t is not None and y_std_t is not None:
                pred_l = (pred - y_mean_t) / y_std_t
                y_l = (Y - y_mean_t) / y_std_t
            else:
                pred_l = pred
                y_l = Y

            loss_e = loss_fn(pred_l, y_l)  # (B,K,T)
            w = M.float()
            loss = (loss_e * w).sum() / w.sum().clamp_min(1e-6)

            opt.zero_grad()
            loss.backward()
            if Cfg.clip_grad_norm and Cfg.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), Cfg.clip_grad_norm)
            opt.step()

            s += loss.item() * Xn.size(0)
            n += Xn.size(0)

        trl = s / max(1, n)
        hist["train"].append(trl)
        sch.step()

        # val
        model.eval()
        s = 0.0
        n = 0
        with torch.no_grad():
            for Xn, Y, M, tvals in va:
                Xn = Xn.to(dev)
                Y = Y.to(dev)
                M = M.to(dev)
                tvals = tvals.to(dev)

                pred = model(Xn, tvals)

                if y_mean_t is not None and y_std_t is not None:
                    pred_l = (pred - y_mean_t) / y_std_t
                    y_l = (Y - y_mean_t) / y_std_t
                else:
                    pred_l = pred
                    y_l = Y

                loss_e = loss_fn(pred_l, y_l)
                w = M.float()
                loss = (loss_e * w).sum() / w.sum().clamp_min(1e-6)

                s += loss.item() * Xn.size(0)
                n += Xn.size(0)

        val = s / max(1, n)
        hist["val"].append(val)

        print(f"[StageA-Phys7][{model_type}][{e}/{Cfg.max_epochs}] train {trl:.4f} | val {val:.4f}")

        if val < best:
            best = val
            ckpt = {
                "model": model.state_dict(),
                "meta": meta,
                "model_type": str(model_type),
                "split_seed": int(split_seed),
                "hist": hist,
            }
            torch.save(ckpt, best_path)

    # learning curve
    with open(os.path.join(out_root, "learning_curve.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train", "val"])
        for i, (a, b) in enumerate(zip(hist["train"], hist["val"]), start=1):
            w.writerow([i, a, b])

    return best_path


# -------------------- 4) 推理 + 导出 + 论文图表 --------------------
def infer_and_export(dataset, meta, ckpt_path: str, out_root: str):
    os.makedirs(out_root, exist_ok=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    model_type = ckpt.get("model_type", Cfg.model_type)

    T = int(meta["T"])
    K = len(meta["families"])

    model = build_stageA_model(T=T, out_dim=K, model_type=model_type).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    loader = DataLoader(dataset, batch_size=Cfg.batch, shuffle=False)

    preds, trues, masks = [], [], []
    with torch.no_grad():
        for Xn, Y, M, tvals in loader:
            Xn = Xn.to(dev)
            tvals = tvals.to(dev)
            pred = model(Xn, tvals)
            preds.append(pred.detach().cpu())
            trues.append(Y.detach().cpu())
            masks.append(M.detach().cpu())

    yhat = torch.cat(preds, 0)
    ytrue = torch.cat(trues, 0)
    mask = torch.cat(masks, 0)

    # 指标
    mts = metrics(yhat, ytrue, mask)

    # 导出表格/summary
    export_predictions_longtable(
        yhat, ytrue, mask,
        families=meta["families"],
        time_values_1d=meta["time_values"],
        out_dir=out_root,
        filename="phys7_predictions.xlsx",
    )
    export_metrics_grid(
        mts,
        families=meta["families"],
        time_values_1d=meta["time_values"],
        out_dir=out_root,
        filename="phys7_metrics.xlsx",
    )
    write_summary_txt(mts, meta["families"], meta["time_values"], out_root)

    # 图：parity/residual/heatmap（论文够用）
    parity_scatter(
        yhat, ytrue, mask,
        out_png=os.path.join(out_root, "phys7_parity_all.png"),
        title="Phys7 Parity (ALL)"
    )
    residual_hist(
        yhat, ytrue, mask,
        out_png=os.path.join(out_root, "phys7_residual_all.png"),
        title="Phys7 Residuals (ALL)"
    )
    if "RMSE" in mts:
        heatmap(
            mts["RMSE"], meta["families"], meta["time_values"],
            title="Phys7 RMSE",
            out_png=os.path.join(out_root, "phys7_rmse.png")
        )
    if "R2" in mts:
        heatmap(
            mts["R2"], meta["families"], meta["time_values"],
            title="Phys7 R2",
            out_png=os.path.join(out_root, "phys7_r2.png")
        )

    # 可选：每个 family 的单独 parity（更像论文附录图）
    fam_dir = os.path.join(out_root, "per_family")
    os.makedirs(fam_dir, exist_ok=True)
    for k, name in enumerate(meta["families"]):
        yh = yhat[:, k:k+1, :]
        yt = ytrue[:, k:k+1, :]
        mk = mask[:, k:k+1, :]
        parity_scatter(
            yh, yt, mk,
            out_png=os.path.join(fam_dir, f"parity_{name}.png"),
            title=f"Parity — {name}"
        )

    return mts


# -------------------- 5) baseline/CV（可发论文的对比表） --------------------
def run_stageA_cv():
    os.makedirs(Cfg.save_dir, exist_ok=True)

    dataset, meta = excel_to_phys7_dataset(
        Cfg.case_excel, Cfg.sheet_name, Cfg.case_id_col, Cfg.iedf_root
    )

    rows = []
    for mt in Cfg.cv_model_types:
        for seed in Cfg.multi_split_seeds:
            out_root = os.path.join(Cfg.save_dir, f"cv_{mt}_seed{seed}")
            ckpt = train_stageA_phys7(dataset, meta, model_type=mt, out_root=out_root, split_seed=seed)
            mts = infer_and_export(dataset, meta, ckpt, out_root=os.path.join(out_root, "exports"))

            # 汇总一个小表（以全体均值为例）
            # metrics() 输出是 KxT 网格，这里 T=1，取 [:,0] 再做均值
            def _mean_grid(name):
                if name not in mts:
                    return float("nan")
                arr = np.asarray(mts[name])
                return float(np.nanmean(arr[:, 0]))

            row = {
                "model_type": mt,
                "seed": int(seed),
                "MAE_mean": _mean_grid("MAE"),
                "RMSE_mean": _mean_grid("RMSE"),
                "R2_mean": _mean_grid("R2"),
            }
            rows.append(row)

    # 写 csv + xlsx（可直接放论文表）
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(Cfg.save_dir, "stageA_phys7_cv_metrics.csv"), index=False)
    try:
        df.to_excel(os.path.join(Cfg.save_dir, "stageA_phys7_cv_metrics.xlsx"), index=False)
    except Exception:
        pass

    # 简单柱状图（R2）
    by_mt = df.groupby("model_type")["R2_mean"]
    means = by_mt.mean()
    stds = by_mt.std()

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    x = np.arange(len(means.index))
    ax.bar(x, means.values, yerr=stds.values, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(means.index.tolist())
    ax.set_ylabel("Mean R$^2$ (over Phys7)")
    ax.set_title("StageA-Phys7: Baseline Comparison (mean ± std)")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(Cfg.save_dir, "stageA_phys7_r2_bar.png"), dpi=300)
    plt.close(fig)


# -------------------- main：单次主结果 + 可选 CV --------------------
def main():
    set_seed(Cfg.seed)
    os.makedirs(Cfg.save_dir, exist_ok=True)

    # 1) dataset
    dataset, meta = excel_to_phys7_dataset(
        Cfg.case_excel, Cfg.sheet_name, Cfg.case_id_col, Cfg.iedf_root
    )

    # 2) 单次主结果：默认 transformer
    out_root = os.path.join(Cfg.save_dir, f"single_{Cfg.model_type}_seed{Cfg.seed}")
    ckpt = train_stageA_phys7(
        dataset, meta,
        model_type=Cfg.model_type,
        out_root=out_root,
        split_seed=Cfg.seed
    )
    infer_and_export(dataset, meta, ckpt, out_root=os.path.join(out_root, "exports"))

    # 3) 如需 baseline/CV 对比再开（默认不自动跑，避免一次跑太久）
    run_stageA_cv()
if __name__ == "__main__":
    main()
