# -*- coding: utf-8 -*-
"""
Stage B: 旧表训练形貌网络（逐 family 标准化 + 独立输出头）
- 训练损失：按 family 加权（仅统计有标注的位置）
- 评估&导出：既有整体，也有“逐 family 独立”产物与检查点
"""
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split


from physio_util import (
    set_seed,
    excel_to_morph_dataset_from_old,
    transform_for_display,
    metrics,
    export_predictions_longtable,
    export_metrics_grid,
    write_summary_txt,
    save_manifest,
    heatmap,
    parity_scatter,
    residual_hist,
    FAMILIES,
)
from phys_model import TemporalRegressor, TemporalRegressorGRU, TemporalRegressorMLP

class Cfg:
    # 数据
    old_excel = r"D:\data\pycharm\bosch\case.xlsx"
    sheet_name = "case"
    save_dir = "./runs_morph_old"

    # 训练
    seed = 42  # 单次默认种子
    multi_split_seeds = [0, 1, 2, 3, 4]  # B1: 多随机划分要跑的 seed 列表
    batch = 64
    max_epochs = 200
    val_ratio = 0.1
    lr = 1e-3
    amp = False

    # 模型（后面 B2/B3 还要用）
    d_model = 64
    nhead = 4
    num_layers = 2
    dim_ff = 128
    dropout = 0.3

    # 展示空间
    unit_scale = 1000.0               # μm → nm
    family_sign = np.array([-1, +1, +1, +1, +1, +1], dtype=np.float32)  # zmin 取正
    flip_sign = False
    clip_nonneg = False               # 如需开启，记得在 transform_for_display 传 nonneg_families
    min_display_value = None

    # 导出控制
    export_family_alone = True        # 导出每个 family 的独立文件/图
    nonneg_families = None            # 例：list(range(len(FAMILIES))) 让所有家族展示非负

    phys_mode = "full"
    input_mode = "full_phys"

    backbone = "transformer"  # "transformer" / "gru" / "mlp"

def _prepare_phys_input(phys: torch.Tensor) -> torch.Tensor:
    if Cfg.input_mode == "full_phys":
        return phys

    B, C, T = phys.shape
    if Cfg.input_mode == "phys_mean":
        # 时间均值 → (B,2,1)，再在时间维上广播回 (B,2,T)
        mean = phys.mean(dim=-1, keepdim=True)   # (B,2,1)
        return mean.expand(B, C, T)

    if Cfg.input_mode == "static_only":
        return torch.zeros_like(phys)

    raise ValueError(f"Unknown input_mode: {Cfg.input_mode}")

def _masked_l1_per_family(pred, target, mask):
    """
    返回：
      loss_mean: 标量（按各 family 的有效点数加权平均）
      per_fam:   (K,) 张量，逐 family 的 L1（按自身有效点数平均）
      counts:    (K,) 张量，每个 family 的有效点数
    形状：
      pred/target/mask: (B,K,T)
    """
    with torch.no_grad():
        counts = mask.float().sum(dim=(0, 2))  # (K,)

    abs_e = torch.abs(pred - target) * mask.float()  # (B,K,T)
    per_fam = abs_e.sum(dim=(0, 2)) / counts.clamp_min(1.0)  # (K,)
    loss_mean = (per_fam * (counts > 0).float()).sum() / (counts > 0).float().sum().clamp_min(1.0)
    return loss_mean, per_fam, counts


def _maybe_denorm_targets(pred, trg, meta, device):
    if isinstance(meta, dict) and "norm_target" in meta:
        mean = meta["norm_target"]["mean"].to(device=device, dtype=pred.dtype)  # (K,)
        std  = meta["norm_target"]["std"].to(device=device, dtype=pred.dtype)   # (K,)
        pred = pred * std.view(1, -1, 1) + mean.view(1, -1, 1)
        trg  = trg  * std.view(1, -1, 1) + mean.view(1, -1, 1)
    return pred, trg


def _select_family(x, k):
    """从 (B,K,T) 张量中取出第 k 个 family → (B,1,T)"""
    return x[:, k:k+1, :]

def _apply_phys_mode(phys, mode):
    if mode == "full":
        return phys  # (B,2,T)
    phys_used = torch.zeros_like(phys)
    if mode == "ion_only":
        phys_used[:, 1:2, :] = phys[:, 1:2, :]
        return phys_used
    if mode == "flux_only":
        phys_used[:, 0:1, :] = phys[:, 0:1, :]
        return phys_used
    if mode == "none":
        return phys_used
    return phys  # 默认不变

def _export_heatmap_to_excel(mtx, families, time_values, out_path):
    df = pd.DataFrame(mtx, index=families, columns=time_values)
    df.index.name = "Family\\Time"
    df.to_excel(out_path)
def _export_parity_to_excel(yhat, ytrue, mask, families, time_values, out_path):
    B,K,T = yhat.shape
    records = []
    for b in range(B):
        for k in range(K):
            for t in range(T):
                if mask[b,k,t]:
                    records.append({
                        "Sample": b,
                        "Family": families[k],
                        "Time": time_values[t],
                        "Predicted": float(yhat[b,k,t].cpu().numpy()),
                        "True": float(ytrue[b,k,t].cpu().numpy())
                    })
    df = pd.DataFrame.from_records(records)
    df.to_excel(out_path, index=False)

def _export_residuals_to_excel(yhat, ytrue, mask, families, time_values, out_path):
    B,K,T = yhat.shape
    records = []
    for b in range(B):
        for k in range(K):
            for t in range(T):
                if mask[b,k,t]:
                    records.append({
                        "Sample": b,
                        "Family": families[k],
                        "Time": time_values[t],
                        "Residual": float((yhat[b,k,t] - ytrue[b,k,t]).cpu().numpy())
                    })
    df = pd.DataFrame.from_records(records)
    df.to_excel(out_path, index=False)

def build_model(T: int, K: int, device: torch.device):
    if Cfg.backbone == "transformer":
        model = TemporalRegressor(
            K=K,
            d_model=Cfg.d_model,
            nhead=Cfg.nhead,
            num_layers=Cfg.num_layers,
            dim_ff=Cfg.dim_ff,
            dropout=Cfg.dropout,
            T=T,
        )
    elif Cfg.backbone == "gru":
        model = TemporalRegressorGRU(
            K=K,
            hidden_dim=Cfg.d_model,
            num_layers=Cfg.num_layers,
            dropout=Cfg.dropout,
            T=T,
        )
    elif Cfg.backbone == "mlp":
        model = TemporalRegressorMLP(
            K=K,
            hidden_dim=Cfg.d_model,
            num_layers=Cfg.num_layers,
            T=T,
        )
    else:
        raise ValueError(f"Unknown backbone: {Cfg.backbone}")
    model.to(device)
    return model
def run_one_split(split_seed, dataset, meta, root_dir):
    # root_dir 类似 "./runs_morph_old/full" 这种
    Cfg.seed = split_seed
    save_dir = os.path.join(root_dir, f"seed{split_seed:02d}")
    Cfg.save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n========== StageB split seed = {split_seed} ==========")
    set_seed(split_seed)
    N = len(dataset)
    nval = int(N * Cfg.val_ratio + 0.5)
    ntr = N - nval
    # 用这个 seed 来控制 random_split
    g = torch.Generator()
    g.manual_seed(split_seed)
    tr_set, va_set = random_split(dataset, [ntr, nval], generator=g)

    tr_loader = DataLoader(tr_set, batch_size=Cfg.batch, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=Cfg.batch, shuffle=False)
    T = int(meta["T"])

    # ===== 构建模型（后面 B3 会扩展成支持多 backbone，这里先保留）=====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(T, len(FAMILIES), device)
    opt = torch.optim.AdamW(model.parameters(), lr=Cfg.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=Cfg.max_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=Cfg.amp)

    # 最佳记录：整体 + 逐 family
    best_overall = -1e9
    best_per_fam = {k: -1e9 for k in range(len(FAMILIES))}
    best_overall_path = os.path.join(save_dir, "morph_best_overall.pth")
    best_fam_path = {
        k: os.path.join(save_dir, f"morph_best_{FAMILIES[k]}.pth")
        for k in range(len(FAMILIES))
    }
    # ===== Train =====
    for epoch in range(1, Cfg.max_epochs + 1):
        model.train()
        tot_loss, num = 0.0, 0
        tr_loss_per_fam_acc = torch.zeros(len(FAMILIES), dtype=torch.float32, device=device)
        tr_count_per_fam_acc = torch.zeros(len(FAMILIES), dtype=torch.float32, device=device)

        for s8, phys, trg, msk, tvals in tr_loader:
            s8, phys, trg, msk, tvals = [x.to(device) for x in (s8, phys, trg, msk, tvals)]
            phys = _apply_phys_mode(phys, Cfg.phys_mode)
            phys_in = _prepare_phys_input(phys)
            opt.zero_grad(set_to_none=True)

            if Cfg.amp:
                with torch.cuda.amp.autocast():
                    pred = model(s8, phys_in, tvals)
                    loss_mean, per_fam_l1, counts = _masked_l1_per_family(pred, trg, msk)
                scaler.scale(loss_mean).backward()
                scaler.step(opt)
                scaler.update()
            else:
                pred = model(s8, phys_in, tvals)
                loss_mean, per_fam_l1, counts = _masked_l1_per_family(pred, trg, msk)
                loss_mean.backward()
                opt.step()

            tot_loss += float(loss_mean.item()) * s8.size(0); num += s8.size(0)
            tr_loss_per_fam_acc += per_fam_l1 * counts
            tr_count_per_fam_acc += counts

        sch.step()

        tr_l1 = tot_loss / max(1, num)
        tr_l1_per_fam = (tr_loss_per_fam_acc / tr_count_per_fam_acc.clamp_min(1)).detach().cpu().numpy()
        tr_l1_msg = " | ".join([f"{FAMILIES[k]}={tr_l1_per_fam[k]:.4f}" for k in range(len(FAMILIES))])

        # ===== Val (display space) =====
        model.eval()
        agg_mts = None
        with torch.no_grad():
            for s8, phys, trg, msk, tvals in va_loader:
                s8, phys, trg, msk, tvals = [x.to(device) for x in (s8, phys, trg, msk, tvals)]
                phys = _apply_phys_mode(phys, Cfg.phys_mode)
                phys_in = _prepare_phys_input(phys)
                pred = model(s8, phys_in, tvals)
                pred, trg = _maybe_denorm_targets(pred, trg, meta, device)

                yhat_disp, ytrue_disp = transform_for_display(
                    pred, trg,
                    family_sign=Cfg.family_sign,
                    unit_scale=Cfg.unit_scale,
                    flip_sign=Cfg.flip_sign,
                    clip_nonneg=Cfg.clip_nonneg,
                    min_display_value=Cfg.min_display_value,
                    nonneg_families=Cfg.nonneg_families
                )
                mts = metrics(yhat_disp, ytrue_disp, msk)  # dict: each (K,T)
                if agg_mts is None:
                    agg_mts = {k: v.copy() for k, v in mts.items()}
                else:
                    # 验证集多 batch 时求平均
                    for k in agg_mts.keys():
                        agg_mts[k] += mts[k]
            # 按批数平均
            if agg_mts is not None:
                for k in agg_mts.keys():
                    agg_mts[k] = agg_mts[k] / max(1, len(va_loader))

        # 逐 family R2
        R2_grid = agg_mts["R2"]  # (K,T)
        R2_per_fam = np.nanmean(R2_grid, axis=1)  # (K,)
        R2_overall = np.nanmean(R2_per_fam)

        r2_msg = " | ".join([f"{FAMILIES[k]}={R2_per_fam[k]:.4f}" for k in range(len(FAMILIES))])
        print(f"[StageB][{epoch}/{Cfg.max_epochs}] train_L1={tr_l1:.4f} ({tr_l1_msg}) | "
              f"val_R2={R2_overall:.4f} ({r2_msg})")

        # 保存整体最佳
        if R2_overall > best_overall:
            best_overall = R2_overall
            torch.save({"model": model.state_dict(), "meta": meta}, best_overall_path)
            print(f"  -> saved overall best to {best_overall_path}")

        # 保存逐 family 最佳（基于该 family 的均值 R2）
        for k in range(len(FAMILIES)):
            if np.isnan(R2_per_fam[k]):
                continue
            if R2_per_fam[k] > best_per_fam[k]:
                best_per_fam[k] = R2_per_fam[k]
                torch.save({"model": model.state_dict(), "meta": meta, "best_family": FAMILIES[k]},
                           best_fam_path[k])
                print(f"  -> saved best for {FAMILIES[k]} to {best_fam_path[k]}")

    # ===== 综合导出（整体 + 相关性） =====
    model.eval()
    static_list = []
    yhat_list = []
    ytrue_list = []
    msk_list = []

    with torch.no_grad():
        for s8, phys, trg, msk, tvals in va_loader:
            s8, phys, trg, msk, tvals = [x.to(device) for x in (s8, phys, trg, msk, tvals)]
            phys = _apply_phys_mode(phys, Cfg.phys_mode)
            phys_in = _prepare_phys_input(phys)
            pred = model(s8, phys_in, tvals)
            pred, trg = _maybe_denorm_targets(pred, trg, meta, device)

            yhat_disp, ytrue_disp = transform_for_display(
                pred, trg,
                family_sign=Cfg.family_sign,
                unit_scale=Cfg.unit_scale,
                flip_sign=Cfg.flip_sign,
                clip_nonneg=Cfg.clip_nonneg,
                min_display_value=Cfg.min_display_value,
                nonneg_families=Cfg.nonneg_families
            )

            static_list.append(s8.cpu())
            yhat_list.append(yhat_disp.cpu())
            ytrue_list.append(ytrue_disp.cpu())
            msk_list.append(msk.cpu())

        # 汇总所有 batch
        s8_all = torch.cat(static_list, dim=0)
        yhat_all = torch.cat(yhat_list, dim=0)
        ytrue_all = torch.cat(ytrue_list, dim=0)
        msk_all = torch.cat(msk_list, dim=0)

        # 整体导出（原样保留，只是现在一次性对全 val）
        export_predictions_longtable(
            yhat_all, ytrue_all, msk_all, FAMILIES, meta["time_values"],
            Cfg.save_dir, filename="predictions.xlsx"
        )
        mts = metrics(yhat_all, ytrue_all, msk_all)
        export_metrics_grid(mts, FAMILIES, meta["time_values"], Cfg.save_dir, filename="metrics.xlsx")
        write_summary_txt(mts, FAMILIES, meta["time_values"], Cfg.save_dir)

        heatmap(mts["R2"], FAMILIES, meta["time_values"], "Morph R2",
                os.path.join(Cfg.save_dir, "morph_r2.png"))
        parity_scatter(yhat_all, ytrue_all, msk_all,
                       os.path.join(Cfg.save_dir, "morph_scatter.png"), "Morph Parity")
        residual_hist(yhat_all, ytrue_all, msk_all,
                      os.path.join(Cfg.save_dir, "morph_residual.png"), "Morph Residuals")
        _export_heatmap_to_excel(
            mts["R2"], FAMILIES, meta["time_values"],
            os.path.join(Cfg.save_dir, "morph_r2_data.xlsx")
        )
        _export_parity_to_excel(
            yhat_all, ytrue_all, msk_all, FAMILIES, meta["time_values"],
            os.path.join(Cfg.save_dir, "morph_scatter_data.xlsx")
        )
        _export_residuals_to_excel(
            yhat_all, ytrue_all, msk_all, FAMILIES, meta["time_values"],
            os.path.join(Cfg.save_dir, "morph_residual_data.xlsx")
        )

        # B4：输出之间相关性分析（对 z/h/d/w）
        analyze_output_correlation(
            static_list,  # 这里其实没用，但接口保留
            ytrue_list,
            yhat_list,
            msk_list,
            Cfg.save_dir
        )

        # 如果以后要启用 B5：
        # analyze_permutation_importance(model, va_loader, meta, Cfg.save_dir, device)

        # ===== 逐 family 独立导出（原逻辑基本不变，只是用 *_all）=====
        if Cfg.export_family_alone:
            for k, fam in enumerate(FAMILIES):
                fam_dir = os.path.join(Cfg.save_dir, "family", fam)
                os.makedirs(fam_dir, exist_ok=True)

                yh_k = _select_family(yhat_all, k)  # (N,1,T)
                yt_k = _select_family(ytrue_all, k)
                m_k = _select_family(msk_all, k)
                fam_list = [fam]

                export_predictions_longtable(
                    yh_k, yt_k, m_k, fam_list, meta["time_values"],
                    fam_dir, filename=f"{fam}_predictions.xlsx"
                )

                mts_k = metrics(yh_k, yt_k, m_k)
                export_metrics_grid(mts_k, fam_list, meta["time_values"],
                                    fam_dir, filename=f"{fam}_metrics.xlsx")
                write_summary_txt(mts_k, fam_list, meta["time_values"], fam_dir)

                heatmap(mts_k["R2"], fam_list, meta["time_values"], f"{fam} R2",
                        os.path.join(fam_dir, f"{fam}_r2.png"))
                parity_scatter(
                    yh_k, yt_k, m_k,
                    os.path.join(fam_dir, f"{fam}_scatter.png"),
                    f"{fam} Parity"
                )
                residual_hist(
                    yh_k, yt_k, m_k,
                    os.path.join(fam_dir, f"{fam}_residual.png"),
                    f"{fam} Residuals"
                )

                _export_heatmap_to_excel(
                    mts_k["R2"], fam_list, meta["time_values"],
                    os.path.join(fam_dir, f"{fam}_r2_data.xlsx")
                )
                _export_parity_to_excel(
                    yh_k, yt_k, m_k, fam_list, meta["time_values"],
                    os.path.join(fam_dir, f"{fam}_scatter_data.xlsx")
                )
                _export_residuals_to_excel(
                    yh_k, yt_k, m_k, fam_list, meta["time_values"],
                    os.path.join(fam_dir, f"{fam}_residual_data.xlsx")
                )
    save_manifest(save_dir)
    print("[OK] Stage B done.")
    return best_overall, best_per_fam
def _masked_mean_over_time(y, msk):
    """
    y, msk: (B,K,T)
    返回每个样本每个 family 按时间 masked mean 后的标量: (B,K)
    """
    m = msk.float()
    num = (y * m).sum(dim=-1)          # (B,K)
    den = m.sum(dim=-1).clamp_min(1.)  # (B,K)
    return num / den


def _r2_score(y_pred, y_true):
    """
    计算每个 family 的 R²，y_pred, y_true: (N,K)
    返回: (K,) numpy array
    """
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)
    K = y_true.shape[1]
    r2 = np.zeros(K, dtype=np.float32)
    for k in range(K):
        yt = y_true[:, k]
        yp = y_pred[:, k]
        mask = ~np.isnan(yt)
        if mask.sum() < 2:
            r2[k] = np.nan
            continue
        yt = yt[mask]; yp = yp[mask]
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-12
        r2[k] = 1.0 - ss_res / ss_tot
    return r2
def analyze_output_correlation(static_list, ytrue_list, yhat_list, msk_list, out_dir):
    """
    static_list: [ (B_i, S) ... ]   目前只用 ytrue/yhat/msk，用不到 static。
    ytrue_list : [ (B_i,K,T) ... ]
    yhat_list  : [ (B_i,K,T) ... ]
    msk_list   : [ (B_i,K,T) ... ]
    out_dir    : 当前 seed 的 save_dir
    """
    # 汇总所有 batch
    ytrue = torch.cat(ytrue_list, dim=0)  # (N,K,T)
    yhat  = torch.cat(yhat_list,  dim=0)  # (N,K,T)
    msk   = torch.cat(msk_list,   dim=0)  # (N,K,T)

    # 按时间做 masked mean，得到每个样本一个标量输出: (N,K)
    ytrue_mean = _masked_mean_over_time(ytrue, msk).cpu().numpy()
    yhat_mean  = _masked_mean_over_time(yhat,  msk).cpu().numpy()

    # 只对有标注的样本取交集 mask
    valid = ~np.isnan(ytrue_mean).any(axis=1)
    ytrue_mean = ytrue_mean[valid]
    yhat_mean  = yhat_mean[valid]

    fam_names = list(FAMILIES)

    # 1) 输出之间的相关性：真值
    df_true = pd.DataFrame(ytrue_mean, columns=[f"{f}_true" for f in fam_names])
    corr_true = df_true.corr(method="pearson")
    corr_true.to_excel(os.path.join(out_dir, "corr_outputs_true.xlsx"))

    # 2) 输出之间的相关性：预测
    df_pred = pd.DataFrame(yhat_mean, columns=[f"{f}_pred" for f in fam_names])
    corr_pred = df_pred.corr(method="pearson")
    corr_pred.to_excel(os.path.join(out_dir, "corr_outputs_pred.xlsx"))

    # 3) 真值 vs 预测（同一 family）相关性（其实就是 R² 的另一种视角）
    #    可以构造一个 6x6 的矩阵，其中对角元素是 corr(true, pred)
    mat = np.zeros((len(fam_names), len(fam_names)), dtype=np.float32)
    for i in range(len(fam_names)):
        yt = ytrue_mean[:, i]
        yp = yhat_mean[:, i]
        mask = ~np.isnan(yt)
        if mask.sum() < 2:
            mat[i, i] = np.nan
            continue
        yt = yt[mask]; yp = yp[mask]
        # 皮尔逊相关系数
        r = np.corrcoef(yt, yp)[0, 1]
        mat[i, i] = r
    df_diag = pd.DataFrame(mat, index=fam_names, columns=fam_names)
    df_diag.to_excel(os.path.join(out_dir, "corr_true_pred_diag.xlsx"))

    # 4) 画两个热力图（IEEE 风格：灰度兼容、字体稍大）
    try:
        import matplotlib.pyplot as plt

        def _heatmap(mat, title, fname):
            fig, ax = plt.subplots(figsize=(4, 3))
            im = ax.imshow(mat, interpolation="nearest", aspect="auto")
            ax.set_xticks(range(len(fam_names)))
            ax.set_yticks(range(len(fam_names)))
            ax.set_xticklabels(fam_names, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(fam_names, fontsize=8)
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
            plt.close(fig)

        _heatmap(corr_true.values, "True outputs correlation", "corr_outputs_true.png")
        _heatmap(corr_pred.values, "Predicted outputs correlation", "corr_outputs_pred.png")

    except Exception as e:
        print(f"[WARN] correlation heatmap plotting failed: {e}")
def main():
    root_dir = Cfg.save_dir        # 当前 phys_mode 下的根目录，如 ./runs_morph_old/full
    os.makedirs(root_dir, exist_ok=True)
    set_seed(Cfg.seed)

    dataset, meta = excel_to_morph_dataset_from_old(
        Cfg.old_excel, sheet_name=Cfg.sheet_name
    )

    all_overall = []
    all_per_fam = []

    for split_seed in Cfg.multi_split_seeds:
        best_overall, best_per_fam = run_one_split(split_seed, dataset, meta, root_dir)
        all_overall.append(best_overall)
        per_fam_array = np.array(
            [best_per_fam[k] for k in range(len(FAMILIES))], dtype=float
        )
        all_per_fam.append(per_fam_array)

    all_overall = np.array(all_overall)
    all_per_fam = np.stack(all_per_fam, axis=0)  # (S,K)

    overall_mean = float(all_overall.mean())
    per_fam_mean = all_per_fam.mean(axis=0)      # (K,)

    print("[OK] Stage B multi-split done.")
    return overall_mean, per_fam_mean

if __name__ == "__main__":
    # -----------------------------
    # 一键跑完 B1 + B2 + B3 对比实验
    #   - phys_mode : 物理通道使用方式
    #   - input_mode: 物理时序输入配置
    #   - backbone  : 模型结构
    # -----------------------------
    phys_modes  = ["full", "ion_only", "flux_only", "none"]
    input_modes = ["full_phys", "phys_mean", "static_only"]
    backbones   = ["transformer", "gru", "mlp"]

    header = ["phys_mode", "input_mode", "backbone", "overall_R2"] + list(FAMILIES)
    summary_rows = []

    print("Running Stage B grid search over (phys_mode, input_mode, backbone)...\n")

    for pm in phys_modes:
        for im in input_modes:
            for bb in backbones:
                # 配置当前组合
                Cfg.phys_mode  = pm
                Cfg.input_mode = im
                Cfg.backbone   = bb

                # 保存路径：./runs_morph_old/pm-*_im-*_bb-*
                combo_dir_name = f"pm-{pm}_im-{im}_bb-{bb}"
                # 替换一下可能不适合文件名的字符（保险起见）
                combo_dir_name = combo_dir_name.replace(" ", "").replace("/", "-")
                Cfg.save_dir = os.path.join("./runs_morph_old", combo_dir_name)

                print("\n" + "=" * 80)
                print(f"[RUN] phys_mode={pm}, input_mode={im}, backbone={bb}")
                print(f"[SAVE DIR] {Cfg.save_dir}")
                print("=" * 80)

                # 跑 multi-split，返回 overall R2 和 per-family R2 的均值
                try:
                    overall_mean, per_fam_mean = main()
                except Exception as e:
                    # 如果某个组合失败，记录 NaN，继续其他组合
                    print(f"[ERROR] Combination failed: pm={pm}, im={im}, bb={bb} -> {e}")
                    overall_mean = float("nan")
                    per_fam_mean = np.full(len(FAMILIES), np.nan, dtype=float)

                # 汇总一行
                row = [
                    pm,
                    im,
                    bb,
                    f"{overall_mean:.4f}" if np.isfinite(overall_mean) else "nan",
                ]
                for k in range(len(FAMILIES)):
                    v = per_fam_mean[k]
                    row.append(f"{v:.4f}" if np.isfinite(v) else "nan")

                summary_rows.append(row)

    # 写总汇总表
    summary_path = "./runs_morph_old/summary_all_combinations.csv"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in summary_rows:
            f.write(",".join(row) + "\n")

    print(f"\n[ALL DONE] Summary of all (phys_mode, input_mode, backbone) combinations "
          f"saved to {summary_path}.")

