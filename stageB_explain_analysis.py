# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import re
import matplotlib.pyplot as plt

from physio_util import (
    set_seed,
    excel_to_morph_dataset_from_old,
    transform_for_display,
    metrics,
    FAMILIES,
)
from phys_model import TemporalRegressor, TemporalRegressorGRU, TemporalRegressorMLP

class CfgExplain:
    # 与 StageB 训练时一致的数据文件
    old_excel = r"D:\data\pycharm\bosch\case.xlsx"
    sheet_name = "case"

    # 要分析的 StageB 模型 ckpt（best overall）
    # 比如: ./runs_morph_old/full/seed00/morph_best_overall.pth
    ckpt_path = r"./runs_morph_old/pm-full_im-full_phys_bb-transformer/seed00/morph_best_overall.pth"


    # 保存解释性结果的根目录
    out_dir = r"./runs_morph_old/explain_full_seed00"

    # 模型结构参数，需要与训练时保持一致
    d_model = 64
    nhead = 4
    num_layers = 2
    dim_ff = 128
    dropout = 0.3
    backbone = "transformer"   # "transformer" / "gru" / "mlp"

    # 物理模式 & 输入模式，与对应 ckpt 训练时一致
    phys_mode = "full"         # "full" / "ion_only" / "flux_only" / "none"
    input_mode = "full_phys"   # "full_phys" / "phys_mean" / "static_only"

    # DataLoader
    batch_size = 64

    # 随机种子
    seed = 0

    # 静态特征名（用于可视化 & 表头）
    # 按你 case.xlsx 中的 static 编码顺序填，比如：
    static_feature_names = [
        "APC", "Source_RF", "LF_RF", "SF6_flow", "C4F8_flow",
        "DEP_time", "ETCH_time"
    ]  # 如果实际维度不同，请对应修改


# =========================
# 工具函数（与 StageB 对齐）
# =========================
def _prepare_phys_input(phys: torch.Tensor, input_mode: str) -> torch.Tensor:
    if input_mode == "full_phys":
        return phys
    B, C, T = phys.shape
    if input_mode == "phys_mean":
        mean = phys.mean(dim=-1, keepdim=True)
        return mean.expand(B, C, T)
    if input_mode == "static_only":
        return torch.zeros_like(phys)
    raise ValueError(f"Unknown input_mode: {input_mode}")


def _apply_phys_mode(phys: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "full":
        return phys
    phys_used = torch.zeros_like(phys)
    if mode == "ion_only":
        phys_used[:, 1:2, :] = phys[:, 1:2, :]
        return phys_used
    if mode == "flux_only":
        phys_used[:, 0:1, :] = phys[:, 0:1, :]
        return phys_used
    if mode == "none":
        return phys_used
    return phys  # fallback


def _maybe_denorm_targets(pred, trg, meta, device):
    if isinstance(meta, dict) and "norm_target" in meta:
        mean = meta["norm_target"]["mean"].to(device=device, dtype=pred.dtype)  # (K,)
        std = meta["norm_target"]["std"].to(device=device, dtype=pred.dtype)    # (K,)
        pred = pred * std.view(1, -1, 1) + mean.view(1, -1, 1)
        trg = trg * std.view(1, -1, 1) + mean.view(1, -1, 1)
    return pred, trg


def _masked_mean_over_time(y, msk):
    """
    y, msk: (N,K,T)
    返回每个样本每个 family 按时间 masked mean 后的标量: (N,K)
    """
    m = msk.float()
    num = (y * m).sum(dim=-1)         # (N,K)
    den = m.sum(dim=-1).clamp_min(1.) # (N,K)
    return num / den


def _r2_score(y_pred, y_true):
    """
    计算每个 family 的 R²，y_pred, y_true: (N,K)
    返回: (K,) numpy array
    """
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)
    N, K = y_true.shape
    r2 = np.zeros(K, dtype=np.float32)
    for k in range(K):
        yt = y_true[:, k]
        yp = y_pred[:, k]
        mask = ~np.isnan(yt)
        if mask.sum() < 2:
            r2[k] = np.nan
            continue
        yt = yt[mask]
        yp = yp[mask]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-12
        r2[k] = 1.0 - ss_res / ss_tot
    return r2


def build_model(T: int, K: int, device: torch.device):
    if CfgExplain.backbone == "transformer":
        model = TemporalRegressor(
            K=K,
            d_model=CfgExplain.d_model,
            nhead=CfgExplain.nhead,
            num_layers=CfgExplain.num_layers,
            dim_ff=CfgExplain.dim_ff,
            dropout=CfgExplain.dropout,
            T=T,
        )
    elif CfgExplain.backbone == "gru":
        model = TemporalRegressorGRU(
            K=K,
            hidden_dim=CfgExplain.d_model,
            num_layers=CfgExplain.num_layers,
            dropout=CfgExplain.dropout,
            T=T,
        )
    elif CfgExplain.backbone == "mlp":
        model = TemporalRegressorMLP(
            K=K,
            hidden_dim=CfgExplain.d_model,
            num_layers=CfgExplain.num_layers,
            T=T,
        )
    else:
        raise ValueError(f"Unknown backbone: {CfgExplain.backbone}")
    model.to(device)
    return model


# =========================
# B5-1: feature–output / residual Pearson 相关性
# =========================
def analyze_feature_output_correlation(
    s8_all: torch.Tensor,
    ytrue_all: torch.Tensor,
    yhat_all: torch.Tensor,
    msk_all: torch.Tensor,
    out_dir: str,
):
    """
    s8_all   : (N,S)
    ytrue_all: (N,K,T)
    yhat_all : (N,K,T)
    msk_all  : (N,K,T)
    输出:
      - corr_feature_output_true.xlsx
      - corr_feature_output_residual.xlsx
      - 对应 IEEE 风格热力图
    """
    os.makedirs(out_dir, exist_ok=True)

    N, S = s8_all.shape
    _, K, T = ytrue_all.shape
    fam_names = list(FAMILIES)
    feat_names = CfgExplain.static_feature_names
    if len(feat_names) != S:
        feat_names = [f"static_{j}" for j in range(S)]

    # 按时间做 masked mean -> (N,K)
    ytrue_mean = _masked_mean_over_time(ytrue_all, msk_all).numpy()
    yhat_mean = _masked_mean_over_time(yhat_all, msk_all).numpy()
    resid_mean = yhat_mean - ytrue_mean

    # 只保留没有 NaN 的样本
    valid = ~np.isnan(ytrue_mean).any(axis=1)
    X = s8_all.numpy()[valid]        # (N_valid, S)
    Y_true = ytrue_mean[valid]       # (N_valid, K)
    Y_resid = resid_mean[valid]      # (N_valid, K)

    # feature vs output 真值相关性矩阵: (S,K)
    corr_feat_out = np.zeros((S, K), dtype=np.float32)
    corr_feat_res = np.zeros((S, K), dtype=np.float32)

    for i in range(S):
        xi = X[:, i]
        for k in range(K):
            yt = Y_true[:, k]
            yr = Y_resid[:, k]
            mask = ~np.isnan(yt)
            if mask.sum() < 3:
                corr_feat_out[i, k] = np.nan
                corr_feat_res[i, k] = np.nan
                continue
            xv = xi[mask]
            yv = yt[mask]
            rv = yr[mask]
            if np.std(xv) < 1e-8 or np.std(yv) < 1e-8:
                corr_feat_out[i, k] = np.nan
            else:
                r = np.corrcoef(xv, yv)[0, 1]
                corr_feat_out[i, k] = r

            if np.std(xv) < 1e-8 or np.std(rv) < 1e-8:
                corr_feat_res[i, k] = np.nan
            else:
                r2 = np.corrcoef(xv, rv)[0, 1]
                corr_feat_res[i, k] = r2

    df_out = pd.DataFrame(corr_feat_out, index=feat_names, columns=fam_names)
    df_res = pd.DataFrame(corr_feat_res, index=feat_names, columns=fam_names)
    df_out.to_excel(os.path.join(out_dir, "corr_feature_output_true.xlsx"))
    df_res.to_excel(os.path.join(out_dir, "corr_feature_output_residual.xlsx"))

    # IEEE 风格热力图
    try:
        import matplotlib.pyplot as plt

        def _heatmap(mat, title, fname):
            fig, ax = plt.subplots(figsize=(3.5, 3.0))  # 约一栏宽度
            im = ax.imshow(mat, interpolation="nearest", aspect="auto", cmap="viridis")
            ax.set_xticks(range(K))
            ax.set_yticks(range(S))
            ax.set_xticklabels(fam_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(feat_names, fontsize=8)
            ax.set_title(title, fontsize=10)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
            ax.tick_params(axis="both", which="both", labelsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
            plt.close(fig)

        _heatmap(
            corr_feat_out,
            "Pearson corr. between static features and true outputs",
            "corr_feature_output_true.png",
        )
        _heatmap(
            corr_feat_res,
            "Pearson corr. between static features and residuals",
            "corr_feature_output_residual.png",
        )
    except Exception as e:
        print(f"[WARN] plotting feature-output correlation failed: {e}")


# =========================
# B5-2: Permutation importance (ΔR²)
# =========================
def analyze_permutation_importance(
    model: torch.nn.Module,
    dataset,
    meta: dict,
    device: torch.device,
    out_dir: str,
):
    """
    对静态特征做 permutation importance:
    - 在 batch 内随机打乱某一列静态特征
    - 观察 time-mean 输出的 R² 降多少
    结果:
      - perm_importance_deltaR2.xlsx
      - perm_importance_deltaR2.png (S×K 热力图)
    """
    os.makedirs(out_dir, exist_ok=True)

    loader = DataLoader(dataset, batch_size=CfgExplain.batch_size, shuffle=False)
    model.eval()

    # ---- 计算 baseline R² ----
    static_all = []
    ytrue_all = []
    yhat_all = []
    msk_all = []

    with torch.no_grad():
        for s8, phys, trg, msk, tvals in loader:
            s8 = s8.to(device)
            phys = phys.to(device)
            trg = trg.to(device)
            msk = msk.to(device)
            tvals = tvals.to(device)

            phys_used = _apply_phys_mode(phys, CfgExplain.phys_mode)
            phys_in = _prepare_phys_input(phys_used, CfgExplain.input_mode)
            pred = model(s8, phys_in, tvals)
            pred, trg = _maybe_denorm_targets(pred, trg, meta, device)

            static_all.append(s8.cpu())
            ytrue_all.append(trg.cpu())
            yhat_all.append(pred.cpu())
            msk_all.append(msk.cpu())

    static_all = torch.cat(static_all, dim=0)  # (N,S)
    ytrue_all = torch.cat(ytrue_all, dim=0)    # (N,K,T)
    yhat_all = torch.cat(yhat_all, dim=0)      # (N,K,T)
    msk_all = torch.cat(msk_all, dim=0)        # (N,K,T)

    ytrue_mean = _masked_mean_over_time(ytrue_all, msk_all).numpy()
    yhat_mean = _masked_mean_over_time(yhat_all, msk_all).numpy()

    valid = ~np.isnan(ytrue_mean).any(axis=1)
    ytrue_valid = ytrue_mean[valid]
    yhat_valid = yhat_mean[valid]
    base_r2 = _r2_score(yhat_valid, ytrue_valid)   # (K,)

    print("[INFO] Baseline R2 per family (time-mean):")
    for f, r in zip(FAMILIES, base_r2):
        print(f"  {f}: {r:.4f}")

    # ---- 对每个静态特征做 permutation ----
    N, S = static_all.shape
    K = len(FAMILIES)
    delta_r2 = np.zeros((S, K), dtype=np.float32)

    for j in range(S):
        print(f"[Perm] feature index {j} ...")
        # 重新走 DataLoader，batch 内 permute 第 j 列
        ytrue_perm_all = []
        yhat_perm_all = []
        msk_perm_all = []

        with torch.no_grad():
            for s8, phys, trg, msk, tvals in loader:
                s8 = s8.to(device)
                phys = phys.to(device)
                trg = trg.to(device)
                msk = msk.to(device)
                tvals = tvals.to(device)

                # batch 内打乱第 j 列
                B = s8.size(0)
                idx = torch.randperm(B, device=device)
                s8_perm = s8.clone()
                if j < s8_perm.size(1):
                    s8_perm[:, j] = s8_perm[idx, j]

                phys_used = _apply_phys_mode(phys, CfgExplain.phys_mode)
                phys_in = _prepare_phys_input(phys_used, CfgExplain.input_mode)
                pred = model(s8_perm, phys_in, tvals)
                pred, trg = _maybe_denorm_targets(pred, trg, meta, device)

                ytrue_perm_all.append(trg.cpu())
                yhat_perm_all.append(pred.cpu())
                msk_perm_all.append(msk.cpu())

        ytrue_perm = torch.cat(ytrue_perm_all, dim=0)
        yhat_perm = torch.cat(yhat_perm_all, dim=0)
        msk_perm = torch.cat(msk_perm_all, dim=0)

        yt_mean = _masked_mean_over_time(ytrue_perm, msk_perm).numpy()
        yp_mean = _masked_mean_over_time(yhat_perm, msk_perm).numpy()
        valid2 = ~np.isnan(yt_mean).any(axis=1)
        yt_valid = yt_mean[valid2]
        yp_valid = yp_mean[valid2]
        r2_perm = _r2_score(yp_valid, yt_valid)

        delta_r2[j, :] = base_r2 - r2_perm

    feat_names = CfgExplain.static_feature_names
    if len(feat_names) != S:
        feat_names = [f"static_{j}" for j in range(S)]
    fam_names = list(FAMILIES)

    df_delta = pd.DataFrame(delta_r2, index=feat_names, columns=fam_names)
    df_delta.to_excel(os.path.join(out_dir, "perm_importance_deltaR2.xlsx"))

    # 热力图
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(3.5, 3.0))
        im = ax.imshow(delta_r2, interpolation="nearest", aspect="auto", cmap="viridis")
        ax.set_xticks(range(K))
        ax.set_yticks(range(S))
        ax.set_xticklabels(fam_names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(feat_names, fontsize=8)
        ax.set_title("Permutation importance (ΔR²)", fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
        ax.tick_params(axis="both", which="both", labelsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "perm_importance_deltaR2.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"[WARN] plotting permutation importance failed: {e}")


# =========================
# B5-3: SHAP surrogate (RandomForest on StageB outputs)
# =========================
def analyze_shap_surrogate(
    s8_all: torch.Tensor,
    yhat_all: torch.Tensor,
    msk_all: torch.Tensor,
    out_dir: str,
):
    """
    用静态特征 → StageB 预测的 time-mean 输出 (pred) 训练一个 RandomForest surrogate，
    然后用 SHAP(TreeExplainer) 解释每个 family 的特征重要性。

    结果：
      - shap_importance_<fam>.xlsx
      - shap_importance_<fam>.png (条形图)
    """
    try:
        import shap
        from sklearn.ensemble import RandomForestRegressor
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("[WARN] SHAP or scikit-learn not installed, skip SHAP analysis.")
        print(f"  {e}")
        return

    os.makedirs(out_dir, exist_ok=True)

    N, S = s8_all.shape
    _, K, T = yhat_all.shape
    feat_names = CfgExplain.static_feature_names
    if len(feat_names) != S:
        feat_names = [f"static_{j}" for j in range(S)]
    fam_names = list(FAMILIES)

    # time-mean 预测
    yhat_mean = _masked_mean_over_time(yhat_all, msk_all).numpy()
    X = s8_all.numpy()

    for fk, fam in enumerate(fam_names):
        y = yhat_mean[:, fk]
        mask = ~np.isnan(y)
        X_fam = X[mask]
        y_fam = y[mask]

        if X_fam.shape[0] < 10:
            print(f"[WARN] too few samples for SHAP on family {fam}, skip.")
            continue

        print(f"[SHAP] training RF surrogate for family {fam} ...")
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            random_state=CfgExplain.seed,
            n_jobs=-1,
        )
        rf.fit(X_fam, y_fam)

        print(f"[SHAP] computing SHAP values for family {fam} ...")
        explainer = shap.TreeExplainer(rf)
        # 可以只采样一部分数据做 SHAP，加快速度
        nsample = min(200, X_fam.shape[0])
        X_sample = X_fam[:nsample]
        shap_values = explainer.shap_values(X_sample)  # (nsample, S)

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)  # (S,)
        df_imp = pd.DataFrame(
            {"feature": feat_names, "mean_abs_shap": mean_abs_shap}
        ).sort_values("mean_abs_shap", ascending=False)

        df_imp.to_excel(
            os.path.join(out_dir, f"shap_importance_{fam}.xlsx"),
            index=False,
        )

        # 条形图 (IEEE 风格)
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        y_pos = np.arange(len(feat_names))
        ax.barh(y_pos, df_imp["mean_abs_shap"].values, align="center")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_imp["feature"].values, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("|SHAP value| (mean)", fontsize=9)
        ax.set_title(f"SHAP importance for {fam}", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"shap_importance_{fam}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


# =========================
# 主流程：加载数据 & 模型，跑 B5-1/2/3
# =========================
def main():
    os.makedirs(CfgExplain.out_dir, exist_ok=True)
    set_seed(CfgExplain.seed)

    # 1) 加载数据
    dataset, meta = excel_to_morph_dataset_from_old(
        CfgExplain.old_excel,
        sheet_name=CfgExplain.sheet_name,
    )
    T = int(meta["T"])
    K = len(FAMILIES)

    # >>> 新增：从 meta 里自动接管静态特征名
    if "static_cols" in meta:
        CfgExplain.static_feature_names = list(meta["static_cols"])
        print("[INFO] static_feature_names from meta:", CfgExplain.static_feature_names)
    else:
        print("[WARN] meta 中没有 'static_cols'，仍然使用手动配置的 static_feature_names")

    # 2) 构建模型 & 加载 ckpt
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(T, K, device)

    # PyTorch 2.6 默认 weights_only=True，这里显式关掉
    try:
        ckpt = torch.load(
            CfgExplain.ckpt_path,
            map_location=device,
            weights_only=False,  # 关键：允许加载带 numpy/meta 的完整 checkpoint
        )
    except TypeError:
        # 老版本 torch 没有 weights_only 参数，退回旧调用方式
        ckpt = torch.load(CfgExplain.ckpt_path, map_location=device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    # 3) 先跑一遍 forward，收集全数据的静态特征 + 预测/真值
    loader = DataLoader(dataset, batch_size=CfgExplain.batch_size, shuffle=False)

    static_all = []
    ytrue_all = []
    yhat_all = []
    msk_all = []

    with torch.no_grad():
        for s8, phys, trg, msk, tvals in loader:
            s8 = s8.to(device)
            phys = phys.to(device)
            trg = trg.to(device)
            msk = msk.to(device)
            tvals = tvals.to(device)

            phys_used = _apply_phys_mode(phys, CfgExplain.phys_mode)
            phys_in = _prepare_phys_input(phys_used, CfgExplain.input_mode)
            pred = model(s8, phys_in, tvals)
            pred, trg = _maybe_denorm_targets(pred, trg, meta, device)

            # display space (可选，如果你希望在 nm / 正向的空间做分析，可以用 transform_for_display)
            # 这里我们直接在物理单位空间做分析
            static_all.append(s8.cpu())
            ytrue_all.append(trg.cpu())
            yhat_all.append(pred.cpu())
            msk_all.append(msk.cpu())

    static_all = torch.cat(static_all, dim=0)  # (N,S)
    ytrue_all  = torch.cat(ytrue_all,  dim=0)
    yhat_all   = torch.cat(yhat_all,   dim=0)
    msk_all    = torch.cat(msk_all,    dim=0)

    S = static_all.shape[1]
    assert len(CfgExplain.static_feature_names) == S, \
        f"静态特征名数量 {len(CfgExplain.static_feature_names)} 和数据维度 S={S} 不一致，请检查 meta['static_cols'] 和编码顺序"
    # 4) B5-1: feature–output / residual Pearson 相关性
    print("\n[B5-1] feature-output & feature-residual Pearson correlation ...")
    out_dir_corr = os.path.join(CfgExplain.out_dir, "B5_pearson")
    analyze_feature_output_correlation(
        static_all,
        ytrue_all,
        yhat_all,
        msk_all,
        out_dir_corr,
    )

    # 5) B5-2: permutation importance (ΔR²)
    print("\n[B5-2] permutation importance (ΔR²) ...")
    out_dir_perm = os.path.join(CfgExplain.out_dir, "B5_permutation")
    analyze_permutation_importance(
        model=model,
        dataset=dataset,
        meta=meta,
        device=device,
        out_dir=out_dir_perm,
    )

    # 6) B5-3: SHAP surrogate
    print("\n[B5-3] SHAP surrogate (RandomForest on StageB outputs) ...")
    out_dir_shap = os.path.join(CfgExplain.out_dir, "B5_shap_surrogate")
    analyze_shap_surrogate(
        static_all,
        yhat_all,
        msk_all,
        out_dir_shap,
    )

    print("\n[OK] StageB explainability analysis done.")
class CfgPlot:
    # StageB 运行输出的根目录
    # 下面会有 pm-* / seedXX / metrics.xlsx / summary_all_combinations.csv
    root_dir = "./runs_morph_old"

    # StageB 训练脚本生成的总汇总表
    summary_csv = os.path.join(root_dir, "summary_all_combinations.csv")

    # B1: 多随机划分稳定性 —— 选一个“主打组合”目录
    # 例如 StageB 主模型 (phys_mode=full, input_mode=full_phys, backbone=transformer):
    # ./runs_morph_old/pm-full_im-full_phys_bb-transformer
    b1_combo_dir_name = "pm-full_im-full_phys_bb-transformer"

    # B2: 物理/输入模式消融固定的 backbone & family
    b2_backbone = "transformer"
    # 对 input_mode 消融时固定的 phys_mode
    b2_phys_fixed = "full"
    # 对 phys_mode 消融时固定的 input_mode
    b2_input_fixed = "full_phys"
    # B2 中重点展示的 family（例如 d1）
    b2_family_for_plot = "d1"

    # B3: backbone 消融时固定的 phys_mode 和 input_mode
    b3_phys_fixed = "full"
    b3_input_fixed = "full_phys"

    # 与 StageB 一致的模型超参数（用于估算参数量）
    old_excel = r"D:\data\pycharm\bosch\case.xlsx"
    sheet_name = "case"
    d_model = 64
    nhead = 4
    num_layers = 2
    dim_ff = 128
    dropout = 0.3

    # 输出图像目录
    out_dir = "./runs_morph_old/summary_plots"

    # Matplotlib 全局字体大小（IEEE 风格，大致 8–10 pt）
    font_size = 9


# =========================
# 通用小工具
# =========================
def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def _set_matplotlib_look():
    """简单设一下全局字体，方便直接拿图进 IEEE。"""
    plt.rcParams.update({
        "font.size": CfgPlot.font_size,
        "axes.labelsize": CfgPlot.font_size,
        "axes.titlesize": CfgPlot.font_size + 1,
        "xtick.labelsize": CfgPlot.font_size - 1,
        "ytick.labelsize": CfgPlot.font_size - 1,
        "legend.fontsize": CfgPlot.font_size - 1,
    })


# =========================
# B1: 多随机划分稳定性
# =========================
def load_b1_seed_r2():
    """
    从指定组合目录中读取各 seed 的 metrics.xlsx(R2 sheet)，
    计算：
      - 每个 seed 的 overall R²
      - 每个 seed 的 per-family R²
    返回：
      seed_names: [ "seed00", "seed01", ... ]
      overall_r2: (S,) numpy array
      per_fam_r2: (S,K) numpy array
    """
    combo_root = os.path.join(CfgPlot.root_dir, CfgPlot.b1_combo_dir_name)
    if not os.path.isdir(combo_root):
        raise FileNotFoundError(f"B1 combo dir not found: {combo_root}")

    seed_dirs = []
    for name in os.listdir(combo_root):
        if re.match(r"seed\d+", name) and os.path.isdir(os.path.join(combo_root, name)):
            seed_dirs.append(name)
    seed_dirs = sorted(seed_dirs)

    if not seed_dirs:
        raise RuntimeError(f"No seedXX dirs found under {combo_root}")

    K = len(FAMILIES)
    per_fam_r2_list = []
    overall_list = []

    for sd in seed_dirs:
        mpath = os.path.join(combo_root, sd, "metrics.xlsx")
        if not os.path.isfile(mpath):
            print(f"[WARN] metrics.xlsx not found in {sd}, skip")
            continue

        # 读取 R2 sheet：行是 families, 列是时间
        df = pd.read_excel(mpath, sheet_name="R2", index_col=0)
        grid = df.to_numpy(dtype=float)  # (K,T)
        if grid.shape[0] != K:
            print(f"[WARN] Unexpected R2 grid shape in {mpath}: {grid.shape}, skip")
            continue

        # per-family R²：按时间平均
        r2_per_fam = np.nanmean(grid, axis=1)  # (K,)
        overall = np.nanmean(r2_per_fam)       # scalar

        per_fam_r2_list.append(r2_per_fam)
        overall_list.append(overall)

    if not per_fam_r2_list:
        raise RuntimeError("No valid metrics.xlsx found for B1")

    per_fam_r2 = np.stack(per_fam_r2_list, axis=0)  # (S,K)
    overall_r2 = np.array(overall_list, dtype=float)

    return seed_dirs, overall_r2, per_fam_r2


def plot_b1_overall_and_per_family():
    seed_names, overall_r2, per_fam_r2 = load_b1_seed_r2()
    _ensure_dir(CfgPlot.out_dir)

    # --- 图 B1(a)：overall R² 的 boxplot ---
    fig, ax = plt.subplots(figsize=(2.2, 3.0))
    ax.boxplot(overall_r2, vert=True, widths=0.4)
    ax.set_xticks([1])
    ax.set_xticklabels(["StageB\n(overall)"])
    ax.set_ylabel(r"$R^2$ (overall)")
    ax.set_title("B1(a) Overall $R^2$ across random splits")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(CfgPlot.out_dir, "B1a_overall_R2_box.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- 图 B1(b)：per-family R² 的 boxplot ---
    K = len(FAMILIES)
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    # per_fam_r2: (S,K) -> 每一列一个 box
    ax.boxplot(per_fam_r2.T, vert=True)
    ax.set_xticks(range(1, K + 1))
    ax.set_xticklabels(FAMILIES, rotation=0)
    ax.set_ylabel(r"$R^2$ (per family)")
    ax.set_title("B1(b) Per-family $R^2$ across random splits")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(CfgPlot.out_dir, "B1b_per_family_R2_box.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] B1 plots saved to {CfgPlot.out_dir}")


# =========================
# B2: 物理/输入模式消融
# =========================
def load_summary_csv():
    if not os.path.isfile(CfgPlot.summary_csv):
        raise FileNotFoundError(f"summary csv not found: {CfgPlot.summary_csv}")
    df = pd.read_csv(CfgPlot.summary_csv)
    # 强制转成 float，可能存在 'nan' 字符串
    num_cols = ["overall_R2"] + list(FAMILIES)
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def plot_b2_ablation():
    df = load_summary_csv()
    _ensure_dir(CfgPlot.out_dir)

    fam = CfgPlot.b2_family_for_plot
    if fam not in FAMILIES:
        raise ValueError(f"b2_family_for_plot={fam} not in FAMILIES={FAMILIES}")
    fam_idx = FAMILIES.index(fam)

    # 只看指定 backbone
    df_bk = df[df["backbone"] == CfgPlot.b2_backbone].copy()
    if df_bk.empty:
        raise RuntimeError(f"No rows found for backbone={CfgPlot.b2_backbone} in summary csv.")

    # --- 图 B2(a)：input_mode 消融（phys_mode 固定） ---
    imodes = df_bk["input_mode"].unique().tolist()
    imodes.sort()
    x_labels_im = []
    vals_im = []

    for im in imodes:
        row = df_bk[
            (df_bk["input_mode"] == im) &
            (df_bk["phys_mode"] == CfgPlot.b2_phys_fixed)
        ]
        if row.empty:
            continue
        r2_val = float(row.iloc[0][fam])  # 该 family 的 R²
        x_labels_im.append(im)
        vals_im.append(r2_val)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x_pos = np.arange(len(vals_im))
    ax.bar(x_pos, vals_im)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels_im, rotation=15, ha="right")
    ax.set_ylabel(rf"$R^2$ for {fam}")
    ax.set_title(f"B2(a) {fam} $R^2$ vs input_mode\n"
                 f"(phys_mode={CfgPlot.b2_phys_fixed}, backbone={CfgPlot.b2_backbone})")
    ax.set_ylim(0.0, 1.0)
    # 在柱子顶部标注数值
    for x, v in zip(x_pos, vals_im):
        ax.text(x, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=CfgPlot.font_size - 2)
    fig.tight_layout()
    fig.savefig(os.path.join(CfgPlot.out_dir, f"B2a_{fam}_R2_vs_input_mode.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- 图 B2(b)：phys_mode 消融（input_mode 固定） ---
    pmodes = df_bk["phys_mode"].unique().tolist()
    pmodes.sort()
    x_labels_pm = []
    vals_pm = []

    for pm in pmodes:
        row = df_bk[
            (df_bk["phys_mode"] == pm) &
            (df_bk["input_mode"] == CfgPlot.b2_input_fixed)
        ]
        if row.empty:
            continue
        r2_val = float(row.iloc[0][fam])
        x_labels_pm.append(pm)
        vals_pm.append(r2_val)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x_pos = np.arange(len(vals_pm))
    ax.bar(x_pos, vals_pm)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels_pm, rotation=15, ha="right")
    ax.set_ylabel(rf"$R^2$ for {fam}")
    ax.set_title(f"B2(b) {fam} $R^2$ vs phys_mode\n"
                 f"(input_mode={CfgPlot.b2_input_fixed}, backbone={CfgPlot.b2_backbone})")
    ax.set_ylim(0.0, 1.0)
    for x, v in zip(x_pos, vals_pm):
        ax.text(x, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=CfgPlot.font_size - 2)
    fig.tight_layout()
    fig.savefig(os.path.join(CfgPlot.out_dir, f"B2b_{fam}_R2_vs_phys_mode.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] B2 plots saved to {CfgPlot.out_dir}")


# =========================
# B3: backbone 消融
# =========================
def build_model_for_backbone(backbone: str, T: int, K: int):
    if backbone == "transformer":
        model = TemporalRegressor(
            K=K,
            d_model=CfgPlot.d_model,
            nhead=CfgPlot.nhead,
            num_layers=CfgPlot.num_layers,
            dim_ff=CfgPlot.dim_ff,
            dropout=CfgPlot.dropout,
            T=T,
        )
    elif backbone == "gru":
        model = TemporalRegressorGRU(
            K=K,
            hidden_dim=CfgPlot.d_model,
            num_layers=CfgPlot.num_layers,
            dropout=CfgPlot.dropout,
            T=T,
        )
    elif backbone == "mlp":
        model = TemporalRegressorMLP(
            K=K,
            hidden_dim=CfgPlot.d_model,
            num_layers=CfgPlot.num_layers,
            T=T,
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model


def compute_param_counts_for_backbones(backbones):
    """
    构造一次模型（不需要加载权重），估算每种 backbone 的参数量。
    T / K 来自 excel_to_morph_dataset_from_old 的 meta。
    """
    dataset, meta = excel_to_morph_dataset_from_old(
        CfgPlot.old_excel,
        sheet_name=CfgPlot.sheet_name,
    )
    T = int(meta["T"])
    K = len(FAMILIES)

    param_counts = {}
    for bb in backbones:
        model = build_model_for_backbone(bb, T, K)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_counts[bb] = n_params
    return param_counts


def plot_b3_backbone_comparison():
    df = load_summary_csv()
    _ensure_dir(CfgPlot.out_dir)

    # 固定 phys_mode / input_mode，看不同 backbone 的 overall_R2
    df_sel = df[
        (df["phys_mode"] == CfgPlot.b3_phys_fixed) &
        (df["input_mode"] == CfgPlot.b3_input_fixed)
    ].copy()
    if df_sel.empty:
        raise RuntimeError(f"No rows for phys_mode={CfgPlot.b3_phys_fixed}, "
                           f"input_mode={CfgPlot.b3_input_fixed} in summary csv.")

    backbones = sorted(df_sel["backbone"].unique().tolist())
    r2_vals = []
    for bb in backbones:
        row = df_sel[df_sel["backbone"] == bb]
        if row.empty:
            continue
        r2_vals.append(float(row.iloc[0]["overall_R2"]))
    # 更新 backbones 顺序仅包含有数据的
    backbones = [bb for bb in backbones if bb in df_sel["backbone"].values]

    # 计算参数量
    param_counts = compute_param_counts_for_backbones(backbones)

    # --- 图 B3(a)：overall R² by backbone ---
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    x_pos = np.arange(len(backbones))
    ax.bar(x_pos, r2_vals)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(backbones)
    ax.set_ylabel(r"$R^2$ (overall)")
    ax.set_title(f"B3(a) Overall $R^2$ vs backbone\n"
                 f"(phys_mode={CfgPlot.b3_phys_fixed}, input_mode={CfgPlot.b3_input_fixed})")
    ax.set_ylim(0.0, 1.0)
    for x, v in zip(x_pos, r2_vals):
        ax.text(x, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=CfgPlot.font_size - 2)
    fig.tight_layout()
    fig.savefig(os.path.join(CfgPlot.out_dir, "B3a_overall_R2_by_backbone.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- 图 B3(b)：参数量 vs overall R² 散点 ---
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for bb, r2 in zip(backbones, r2_vals):
        n_params = param_counts.get(bb, None)
        if n_params is None:
            continue
        ax.scatter(n_params, r2)
        # 标注 backbone 名字
        ax.text(n_params, r2, bb, fontsize=CfgPlot.font_size - 2,
                ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters (log scale)")
    ax.set_ylabel(r"$R^2$ (overall)")
    ax.set_title("B3(b) Parameter count vs overall $R^2$")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(CfgPlot.out_dir, "B3b_param_vs_R2_scatter.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] B3 plots saved to {CfgPlot.out_dir}")

if __name__ == "__main__":
    main()
    _ensure_dir(CfgPlot.out_dir)
    _set_matplotlib_look()

    print("=== Plotting B1 (multi-split stability) ===")
    plot_b1_overall_and_per_family()

    print("=== Plotting B2 (input/phys ablation) ===")
    plot_b2_ablation()

    print("=== Plotting B3 (backbone comparison) ===")
    plot_b3_backbone_comparison()

    print(f"\n[ALL DONE] B1–B3 summary plots saved to: {CfgPlot.out_dir}")
