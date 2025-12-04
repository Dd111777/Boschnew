# ============================================
# Block A：基础配置 + 数据入口
# ============================================
import math
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression

# ---- 依赖你现有工程里的模块 ----
from physio_util import (
    excel_to_morph_dataset_from_old,
    load_new_excel_as_sparse_morph,
    FAMILIES, TIME_LIST, F2IDX, T2IDX,
)
from stageC_finetune_joint_on_new_pycharm_new import Cfg as CfgC
from stageB_train_morph_on_old_pycharm import Cfg as Bcfg
from phys_model import TemporalRegressor


try:
    from scipy.stats import spearmanr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def ensure_dir(path: str):
    """确保目录存在。"""
    os.makedirs(path, exist_ok=True)


class ExpConfig:
    # ========= 1. 数据路径（与 StageC 对齐） =========
    # 旧表：包含全时序 F_Flux / Ion_Flux + 全 family 形貌标签
    old_excel: str = getattr(CfgC, "old_excel", r"D:\data\pycharm\bosch\case.xlsx")
    # 新表：Bosch_new，包含稀疏形貌（w/h/d/zmin）
    new_excel: str = getattr(CfgC, "new_excel", CfgC.new_excel)

    # 保存目录（主动选点结果 + 敏感度分析报告）
    save_dir: str = "./stageC_advanced_selection"

    # ========= 2. 预训练权重路径（Block B 用） =========
    # 物理网（StageA） & 旧表形貌网（StageB）
    phys_ckpt_F: str = getattr(
        CfgC, "phys_ckpt_F", "./runs_phys_split/F_Flux/phys_best.pth"
    )
    phys_ckpt_I: str = getattr(
        CfgC, "phys_ckpt_I", "./runs_phys_split/Ion_Flux/phys_best.pth"
    )
    stageB_morph_ckpt: str = getattr(
        CfgC, "morph_ckpt", "./runs_morph_old/morph_best_overall.pth"
    )

    # ========= 3. 静态输入参数范围（顺序与旧表/新表完全一致） =========
    # 顺序必须和 physio_util.excel_to_morph_dataset_from_old 中的 static_keys 一致：
    # [APC, source_RF, LF_RF, SF6, C4F8, DEP_time, etch_time]
    static_params: Dict[str, Dict[str, float]] = {
        "APC": {"min": 10, "max": 100, "step": 5, "unit": "mT"},
        "source_RF": {"min": 1000, "max": 3500, "step": 100, "unit": "W"},
        "LF_RF": {"min": 25, "max": 150, "step": 5, "unit": "W"},
        "SF6": {"min": 50, "max": 500, "step": 50, "unit": "sccm"},
        "C4F8": {"min": 50, "max": 500, "step": 50, "unit": "sccm"},
        "DEP_time": {"min": 0.4, "max": 2.6, "step": 0.4, "unit": "s"},
        "etch_time": {"min": 0.4, "max": 2.6, "step": 0.4, "unit": "s"},
    }
    # 静态参数名列表（和列顺序一致）
    static_names: List[str] = list(static_params.keys())

    # ========= 4. d 目标约束（单位/数值可按实际再调整） =========
    # 这里先沿用旧版 selection 的标准：目标 d1_3/5/9 ≈ 0.01 μm
    d_constraints: Dict[str, Dict[str, float]] = {
        "d1_3": {"min": 0.0, "max": 0.05, "unit": "μm", "center": 0.01},
        "d1_5": {"min": 0.0, "max": 0.05, "unit": "μm", "center": 0.01},
        "d1_9": {"min": 0.0, "max": 0.05, "unit": "μm", "center": 0.01},
    }

    d_names: List[str] = list(d_constraints.keys())
    d_weights: List[float] = [0.3, 0.3, 0.4]  # 多目标加权权重

    # ========= 5. 其他基本参数 =========
    # 候选池大小 / 每轮推荐 top-K 数量
    n_candidates: int = 20000
    n_select: int = 30
    n_rounds: int = 1  # 当前场景：一次性推荐即可

    # 随机种子（后续 Block 可扩展成 seeds 列表）
    seed: int = 42
    # ========= 6. 瓶型惩罚强度 =========
    bottle_penalty_weight: float = 2.0


# ============================================================
# 旧表：StageB 数据入口
# ============================================================

def load_stageB_dataset(config: ExpConfig) -> Dict[str, object]:
    # 关键：一定要指定 sheet_name，否则 pd.read_excel 返回的是 dict
    sheet = getattr(Bcfg, "sheet_name", None)  # Bcfg.sheet_name 一般是 "case"
    ds, meta = excel_to_morph_dataset_from_old(
        config.old_excel,
        sheet_name=sheet,
    )    # ds.tensors: (static_norm, phys_seq, targets, mask, time_mat)
    static_norm, phys_seq, targets, mask, time_mat = ds.tensors

    # 2. 反标准化得到原始静态参数（便于解释）
    #    meta["norm_static"]["mean/std"] 形状为 (1, D)
    norm_mean = meta["norm_static"]["mean"].to(static_norm)
    norm_std = meta["norm_static"]["std"].to(static_norm)
    static_raw = static_norm * norm_std + norm_mean  # (N, D)

    # 3. 抽取 d1 family 在 t=3/5/9 的值（反归一化到物理空间）
    fam2idx = {name: i for i, name in enumerate(FAMILIES)}
    if "d1" not in fam2idx:
        raise RuntimeError(f"FAMILIES 中缺少 'd1'，当前为：{FAMILIES}")
    d1_idx = fam2idx["d1"]

    # 时间索引：对应 TIME_LIST = ["1","2",...,"9","9_2"]
    try:
        t_indices = [TIME_LIST.index(t) for t in ["3", "5", "9"]]
    except ValueError as e:
        raise RuntimeError(f"TIME_LIST 中未找到 3/5/9，请检查 TIME_LIST 定义") from e

    # 归一化空间下的 d1(t) 以及对应 mask
    d1_norm = targets[:, d1_idx, t_indices]  # (N, 3)
    d1_mask = mask[:, d1_idx, t_indices]     # (N, 3) bool

    # family-wise 归一化统计量
    norm_target = meta["norm_target"]
    fam_mean = norm_target["mean"]  # (K,)
    fam_std = norm_target["std"]    # (K,)
    d1_mean = fam_mean[d1_idx]      # 标量 Tensor
    d1_std = fam_std[d1_idx]        # 标量 Tensor

    # 反归一化到物理量
    d1_phys = d1_norm * d1_std + d1_mean  # (N, 3)

    # 只保留同时有 d1_3/5/9 的样本（避免 NaN 干扰）
    valid_rows = d1_mask.all(dim=1)  # (N,)
    X_stageB_d1 = static_raw[valid_rows].cpu().numpy()  # (N_valid, D)
    Y_stageB_d1 = d1_phys[valid_rows].cpu().numpy()     # (N_valid, 3)

    # 一些方便后续使用的信息
    out = {
        # 原始 TensorDataset 与 meta：给后续 StageB 模型 / 梯度分析用
        "dataset": ds,
        "meta": meta,
        # 全量静态输入（标准化/反标准化）
        "X_all_norm": static_norm,                 # (N, D) torch.FloatTensor
        "X_all_raw": static_raw,                   # (N, D) torch.FloatTensor
        # 只包含 d1_3/5/9 全部存在的样本（做相关性/GP 的干净子集）
        "X_d1_raw": X_stageB_d1,                   # (N_valid, D) numpy
        "Y_d1_phys": Y_stageB_d1,                  # (N_valid, 3) numpy
        # 一些索引信息
        "d1_family_index": d1_idx,
        "time_indices_d1": t_indices,
        "d1_mask": d1_mask,                        # (N, 3) torch.BoolTensor
    }
    return out


# ============================================================
# 新表：Bosch 稀疏形貌数据入口
# ============================================================

def load_new_table_d1(config: ExpConfig) -> Tuple[np.ndarray, np.ndarray]:
    recs = load_new_excel_as_sparse_morph(
        config.new_excel,
        height_family="h1",  # 与 StageB/C 形貌高度 family 保持一致
    )

    X_list: List[np.ndarray] = []
    d_list: List[List[float]] = []
    b_list: List[float] = []

    for rec in recs:
        static = rec["static"]  # shape: (7,)
        X_list.append(static.astype(np.float32))

        tg = rec["targets"]     # dict: (family, tid) -> value
        d_vals: List[float] = []
        for tid in ["3", "5", "9"]:
            key = ("d1", tid)
            v = tg.get(key, np.nan)
            # 统一用 numpy 判断有效性，避免 None / NaN 混用
            if v is None or not np.isfinite(float(v)):
                d_vals.append(np.nan)
            else:
                d_vals.append(float(v))
        d_list.append(d_vals)

        # 瓶型标签（None / 0 / 1）
        b = rec.get("bottle_flag", None)
        b_list.append(np.nan if b is None else float(b))

    X_new = np.asarray(X_list, np.float32)
    d_new = np.asarray(d_list, np.float32)
    b_new = np.asarray(b_list, np.float32)

    mask_valid = np.isfinite(d_new).all(axis=1)
    X_new = X_new[mask_valid]
    d_new = d_new[mask_valid]
    b_new = b_new[mask_valid]

    print(f"[load_new_table_d1] 加载 Bosch 新表有效样本数：{X_new.shape[0]}")
    if X_new.shape[0] == 0:
        raise ValueError(
            f"从 {config.new_excel} 中未找到同时包含 d1_3/5/9 的有效样本，请检查新表内容。"
        )

    return X_new, d_new, b_new

# ========================= Block B.1：StageB 模型包装 =========================
STAGEB_DEFAULT_CKPT = os.path.join(Bcfg.save_dir, "morph_best_overall.pth")


class StageBOracle:
    def __init__(
            self,
            ckpt_path: Optional[str] = None,
            device: Optional[torch.device] = None,
            old_excel_path: Optional[str] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = ckpt_path or STAGEB_DEFAULT_CKPT
        self.old_excel_path = old_excel_path or Bcfg.old_excel

        # 这几个成员在 _load_model_and_meta / _build_phys_template 中初始化
        self.model: Optional[nn.Module] = None
        self.static_mean: Optional[torch.Tensor] = None
        self.static_std: Optional[torch.Tensor] = None
        self.time_values: Optional[torch.Tensor] = None
        self.family_names = list(FAMILIES)
        self.idx_d1: int = F2IDX["d1"]
        self.T: int = len(TIME_LIST)

        self.phys_template: Optional[torch.Tensor] = None   # (1,2,T)
        self.tvals_template: Optional[torch.Tensor] = None  # (T,)

        # 保留旧表数据集，后续 Block B.2 可以用来做相关性/灵敏度分析
        self.old_dataset = None
        self.old_meta = None

        self._load_model_and_meta()
        self._build_phys_template()

    # ------------------------------------------------------------------
    # 内部：加载模型和 meta
    def _load_model_and_meta(self):
        if not os.path.isfile(self.ckpt_path):
            raise FileNotFoundError(f"StageB ckpt 不存在：{self.ckpt_path}")

        # 读取 ckpt
        ckpt = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)

        # ---- 1) 兼容两种字段命名：model / model_state_dict ----
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            raise KeyError(
                f"StageB ckpt 中既没有 'model' 也没有 'model_state_dict'，"
                f"请检查 {self.ckpt_path} 的保存格式。"
            )

        # ---- 2) 兼容 meta 字段：meta / stageB_meta ----
        if "meta" in ckpt:
            meta = ckpt["meta"]
        elif "stageB_meta" in ckpt:
            meta = ckpt["stageB_meta"]
        else:
            raise KeyError(
                f"StageB ckpt 中既没有 'meta' 也没有 'stageB_meta'，"
                f"请检查 {self.ckpt_path} 的保存格式。"
            )

        # 取静态归一化参数
        if "norm_static" not in meta:
            raise KeyError(
                f"StageB ckpt.meta 中没有 'norm_static' 字段，"
                f"请确认使用的是 morph_best_overall.pth。"
            )

        norm_static = meta["norm_static"]
        mean = norm_static["mean"].float().view(1, -1)  # (1,7)
        std = norm_static["std"].float().view(1, -1)  # (1,7)

        self.static_mean = mean.to(self.device)
        self.static_std = std.to(self.device)

        # 时间轴信息
        self.T = int(meta.get("T", len(TIME_LIST)))
        tv = meta.get("time_values", TIME_LIST[: self.T])
        self.time_values = torch.as_tensor(
            np.asarray(tv, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

        # family 名称
        fams = meta.get("families", FAMILIES)
        self.family_names = list(fams)
        if "d1" in self.family_names:
            self.idx_d1 = self.family_names.index("d1")
        else:
            # 理论上不会进这里，兜底用全局表
            self.idx_d1 = F2IDX["d1"]

        # 构建并加载 TemporalRegressor
        model = TemporalRegressor(
            K=len(self.family_names),
            d_model=Bcfg.d_model,
            nhead=Bcfg.nhead,
            num_layers=Bcfg.num_layers,
            dim_ff=Bcfg.dim_ff,
            dropout=Bcfg.dropout,
            T=self.T,
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(self.device)
        model.eval()

        self.model = model
        self.old_meta = meta
    # ------------------------------------------------------------------
    # 内部：构造“平均物理时序”模板 + 保存旧表数据
    # ------------------------------------------------------------------
    def _build_phys_template(self):
        ds, meta = excel_to_morph_dataset_from_old(
            self.old_excel_path, sheet_name=getattr(Bcfg, "sheet_name", None)
        )
        # ds.tensors: static_norm, phys_seq, targets, mask, time_mat
        static_norm, phys_seq, targets, mask, time_mat = ds.tensors

        self.old_dataset = ds
        self.old_meta = meta

        # 平均物理时序 (1,2,T)
        phys_mean = phys_seq.float().mean(dim=0, keepdim=True)  # (1,2,T)
        self.phys_template = phys_mean.to(self.device)

        # 一行时间刻度 (T,)
        tvals = time_mat[0].float()  # (T,)
        self.tvals_template = tvals.to(self.device)

    # ------------------------------------------------------------------
    # 对外接口：预测完整 (B,K,T) 形貌序列
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_full_profile(self, X_static: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("StageB 模型尚未加载。")

        x = np.asarray(X_static, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[1] != self.static_mean.shape[1]:
            raise ValueError(
                f"StageBOracle.predict_full_profile 期望输入维度 {self.static_mean.shape[1]}，"
                f"但得到 {x.shape[1]}。"
            )

        x_t = torch.from_numpy(x).to(self.device)  # (B,7)
        x_norm = (x_t - self.static_mean) / (self.static_std + 1e-8)  # (B,7)

        B = x_norm.size(0)
        phys = self.phys_template.expand(B, -1, -1)                  # (B,2,T)
        tvals = self.tvals_template.unsqueeze(0).expand(B, -1)       # (B,T)

        y = self.model(x_norm, phys, tvals)                          # (B,K,T)
        return y.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # 对外接口：只取 d1 在 t=3/5/9 的预测值
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_d1_3_5_9(
        self,
        X_static: np.ndarray,
        time_ids: Tuple[str, ...] = ("3", "5", "9"),
    ) -> np.ndarray:
        full = self.predict_full_profile(X_static)  # (B,K,T)

        # 将 time_ids 转成索引；如果某个时间不在 T2IDX 中则跳过
        valid_tids = [t for t in time_ids if t in T2IDX]
        if not valid_tids:
            raise ValueError(f"time_ids={time_ids} 中没有任何元素在 TIME_LIST 中，请检查。")

        t_indices = [T2IDX[t] for t in valid_tids]  # e.g. [2,4,8]
        d = full[:, self.idx_d1, t_indices]         # (B, len(valid_tids))
        return d

# ========================= Block B.2：StageB 敏感度 & Active Subspace =========================

def compute_stageB_correlations(stageB_data: dict, cfg: "ExpConfig") -> dict:
    X = np.asarray(stageB_data["X_d1_raw"], dtype=np.float64)   # (N, D)
    Y = np.asarray(stageB_data["Y_d1_phys"], dtype=np.float64)  # (N, 3)

    N, D = X.shape
    _, Dy = Y.shape
    assert Dy == len(cfg.d_names), \
        f"Y 列数 {Dy} 与 cfg.d_names 长度 {len(cfg.d_names)} 不一致"

    # ---------- Pearson ----------
    pearson_mat = np.zeros((D, Dy), dtype=np.float64)
    for i in range(D):
        xi = X[:, i]
        xi = xi - xi.mean()
        xi_std = np.sqrt(np.sum(xi * xi) + 1e-12)
        for j in range(Dy):
            yj = Y[:, j]
            yj = yj - yj.mean()
            yj_std = np.sqrt(np.sum(yj * yj) + 1e-12)
            num = np.sum(xi * yj)
            pearson_mat[i, j] = float(num / (xi_std * yj_std + 1e-12))

    pearson_df = pd.DataFrame(
        pearson_mat,
        index=list(cfg.static_names),
        columns=list(cfg.d_names),
    )

    # ---------- Spearman（如果有 scipy） ----------
    if _HAS_SCIPY:
        spearman_mat = np.zeros((D, Dy), dtype=np.float64)
        for i in range(D):
            for j in range(Dy):
                rho, _ = spearmanr(X[:, i], Y[:, j])
                spearman_mat[i, j] = float(rho)
        spearman_df = pd.DataFrame(
            spearman_mat,
            index=list(cfg.static_names),
            columns=list(cfg.d_names),
        )
    else:
        spearman_df = None
        print("[compute_stageB_correlations] 未找到 scipy，Spearman 相关系数将跳过，仅返回 Pearson。")

    return {
        "pearson": pearson_df,
        "spearman": spearman_df,
    }


def estimate_active_subspace_from_stageB(
    oracle: "StageBOracle",
    cfg: "ExpConfig",
    n_samples: int = 512,
    batch_size: int = 64,
    target_time: str = "9",
) -> dict:
    if oracle.old_dataset is None or oracle.old_meta is None:
        raise RuntimeError(
            "StageBOracle.old_dataset 为空，"
            "请确认 StageBOracle 初始化时已经成功从旧表构造了 phys_template。"
        )

    ds = oracle.old_dataset
    static_norm, phys_seq, targets, mask, time_mat = ds.tensors
    # static_norm: (N, D)
    N, D = static_norm.shape
    D_cfg = len(cfg.static_names)
    if D != D_cfg:
        print(
            f"[estimate_active_subspace_from_stageB] 警告：数据维度 D={D} "
            f"与 cfg.static_names 长度 {D_cfg} 不一致，将以数据维度为准。"
        )

    # 抽样 n_samples 个索引
    M = min(n_samples, N)
    idx_all = np.random.permutation(N)[:M]

    if target_time not in T2IDX:
        raise ValueError(f"target_time={target_time} 不在 TIME_LIST / T2IDX 中，请检查。")
    tidx = T2IDX[target_time]

    model = oracle.model
    device = oracle.device

    # J(x) = Σ_j w_j * (d1_j(x) - center_j)^2 的参数（3/5/9 三个时刻）
    centers = torch.tensor(
        [cfg.d_constraints[name]["center"] for name in cfg.d_names],
        dtype=torch.float32,
        device=device,
    )  # (3,)
    w = torch.tensor(cfg.d_weights, dtype=torch.float32, device=device)
    w = w / (w.sum() + 1e-12)
    tidx_list = [T2IDX[t] for t in ["3", "5", "9"]]

    grads_list = []

    # 按 batch 求梯度
    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        batch_idx = idx_all[start:end]
        if len(batch_idx) == 0:
            continue

        x_norm = static_norm[batch_idx].to(device)       # (B,D)
        phys = phys_seq[batch_idx].to(device)            # (B,2,T)
        tvals = time_mat[batch_idx].to(device)           # (B,T)

        x_norm = x_norm.detach().clone().requires_grad_(True)

        y = model(x_norm, phys, tvals)  # (B,K,T)
        idx_d1 = oracle.idx_d1

        # 取 d1 family 在 3/5/9 三个时刻的输出，按目标 J 聚合
        d_list = []
        for tid in tidx_list:
            d_list.append(y[:, idx_d1, tid])  # 每个是 (B,)
        d_vals = torch.stack(d_list, dim=1)  # (B,3)

        diff = d_vals - centers[None, :]  # (B,3)
        J_batch = torch.sum(w[None, :] * diff ** 2, dim=1)  # (B,)

        # 对所有样本求平均再回传，数值更稳定
        val = J_batch.mean()
        val.backward()

        g = x_norm.grad.detach().cpu().numpy()           # (B,D)
        grads_list.append(g)

    if not grads_list:
        raise RuntimeError("未能计算到任何梯度，请检查 n_samples/batch_size 设置。")

    grads = np.concatenate(grads_list, axis=0)           # (M_eff, D)
    M_eff = grads.shape[0]

    # 构造 C = 1/M Σ g_i g_i^T
    C = grads.T @ grads / float(M_eff)                   # (D,D)

    # 对称矩阵特征分解
    eigvals, eigvecs = np.linalg.eigh(C)                 # eigh 结果是升序
    idx_sort = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx_sort]
    eigvecs = eigvecs[:, idx_sort]                       # 列为特征向量

    eigvals_frac = eigvals / (eigvals.sum() + 1e-12)

    # 第一主方向上，各静态参数的绝对权重（归一化后方便对比）
    v1 = eigvecs[:, 0]
    importance_first = np.abs(v1) / (np.abs(v1).sum() + 1e-12)

    # 为了方便下游使用，自动补全 param_names 长度
    if len(cfg.static_names) != D:
        param_names = [f"x{i}" for i in range(D)]
    else:
        param_names = list(cfg.static_names)

    result = {
        "eigenvalues": eigvals,
        "eigenvalues_frac": eigvals_frac,
        "eigenvectors": eigvecs,
        "param_importance_first": importance_first,
        "param_names": param_names,
        "target_time": target_time,
    }

    # 简单打印一下能量占比，便于快速判断维度是否可压缩
    print("[estimate_active_subspace_from_stageB] eigenvalues_frac (前 3 项)：",
          np.round(eigvals_frac[:3], 4))

    return result
def compute_param_importance_and_trust_region(
    cfg: "ExpConfig",
    corr_info: dict,
    as_info: dict,
    alpha: float = 0.6,
    base_trust_radius: float = 1.0,
) -> dict:
    # 1) Active Subspace 第一主方向重要度
    imp_as = np.asarray(as_info["param_importance_first"], dtype=np.float64)  # (D,)
    imp_as = imp_as / (imp_as.sum() + 1e-12)

    # 2) Pearson：对 d1_3/5/9 的绝对值相关性取平均
    pearson_df = corr_info.get("pearson", None)
    if pearson_df is None:
        imp_corr = np.ones_like(imp_as) / imp_as.size
    else:
        pearson_abs = np.abs(pearson_df.values)  # (D, 3)
        imp_corr = pearson_abs.mean(axis=1)
        imp_corr = imp_corr / (imp_corr.sum() + 1e-12)

    # 3) 融合（alpha 越大越偏向 Active Subspace）
    imp = alpha * imp_as + (1.0 - alpha) * imp_corr
    imp = imp / (imp.sum() + 1e-12)

    # 4) 定义各向异性 trust-region 半径：重要度越大，允许步长越大
    #    把平均半径归一到 base_trust_radius
    scale = base_trust_radius / (imp.mean() + 1e-12)
    trust_radius_norm = scale * imp

    as_info = dict(as_info)  # 拷一份，避免外部误改
    as_info["param_importance_combined"] = imp           # (D,)
    as_info["trust_radius_norm"] = trust_radius_norm     # (D,)

    return as_info

# ========================= Block C：多保真 Residual GP 代理 =========================
class MultiFidelityResidualGP:
    def __init__(
        self,
        cfg: "ExpConfig",
        oracle: "StageBOracle",
        X_train: np.ndarray,
        y_real_train: np.ndarray,
        jitter: float = 1e-6,
        n_restarts_optimizer: int = 5,
        param_importance: Optional[np.ndarray] = None,
    ):
        self.cfg = cfg
        self.oracle = oracle

        X = np.asarray(X_train, dtype=np.float64)
        Y_real = np.asarray(y_real_train, dtype=np.float64)
        assert X.shape[0] == Y_real.shape[0], "X_train 与 y_real_train 样本数不一致"

        self.X_train_raw = X
        self.y_real_train = Y_real
        self.input_dim = self.X_train_raw.shape[1]
        self.output_dim = self.y_real_train.shape[1]
        if param_importance is not None:
            imp = np.asarray(param_importance, dtype=np.float64)
            imp = imp / (imp.mean() + 1e-12)  # 归一到均值=1，方便直接缩放
            self.param_importance = imp  # (D,)
        else:
            self.param_importance = None
        # d1 输出的物理上下界（用于后处理裁剪）
        self.y_min = np.array(
            [cfg.d_constraints[name]["min"] for name in cfg.d_names],
            dtype=np.float64,
        )[None, :]  # (1,3)
        self.y_max = np.array(
            [cfg.d_constraints[name]["max"] for name in cfg.d_names],
            dtype=np.float64,
        )[None, :]  # (1,3)
        # 特征标准化（防止 GP length_scale 优化异常）
        self.x_mean = self.X_train_raw.mean(axis=0, keepdims=True)  # (1,D)
        self.x_std = self.X_train_raw.std(axis=0, keepdims=True)    # (1,D)
        self.x_std[self.x_std < 1e-6] = 1.0

        X_scaled = (self.X_train_raw - self.x_mean) / self.x_std

        # 低保真：StageB 在新表上的预测
        self.y_sim_train = self.oracle.predict_d1_3_5_9(self.X_train_raw)  # (N,3)
        assert self.y_sim_train.shape == self.y_real_train.shape

        # 残差：高保真 - 低保真
        self.residual_train = self.y_real_train - self.y_sim_train         # (N,3)
        if self.param_importance is not None:
            imp = self.param_importance  # (D,)
            length_scale0 = 1.0 / (imp + 1e-3)  # 避免除零，尺度可以按需调
        else:
            length_scale0 = np.ones(self.input_dim, dtype=np.float64)

        self.gps: list[GaussianProcessRegressor] = []
        for j in range(self.output_dim):
            kernel = (
                    C(1.0, (1e-3, 1e3))
                    * RBF(
                length_scale=length_scale0,
                length_scale_bounds=(1e-2, 1e3),
            )
                    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e0))
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=jitter,
                n_restarts_optimizer=n_restarts_optimizer,
                normalize_y=True,
            )
            gp.fit(X_scaled, self.residual_train[:, j])
            self.gps.append(gp)

        # 拟合后做一个简单诊断
        self.diagnostics = self._compute_diagnostics(X_scaled)

    # ------------------------------------------------------------------
    # 诊断：StageB vs 多保真 surrogate 的 R²
    # ------------------------------------------------------------------
    def _compute_diagnostics(self, X_scaled: np.ndarray) -> dict:
        r2_stageB = {}
        r2_multi = {}
        y_multi_pred = np.zeros_like(self.y_real_train)

        for j, d_name in enumerate(self.cfg.d_names):
            # StageB 单独
            r2_s = r2_score(self.y_real_train[:, j], self.y_sim_train[:, j])
            r2_stageB[d_name] = float(r2_s)

            # 残差 GP 校正后的预测
            y_r_pred = self.gps[j].predict(X_scaled)  # (N,)
            y_pred_j = self.y_sim_train[:, j] + y_r_pred
            y_multi_pred[:, j] = y_pred_j

            r2_m = r2_score(self.y_real_train[:, j], y_pred_j)
            r2_multi[d_name] = float(r2_m)

        print("[MultiFidelityResidualGP] 诊断 R²（StageB → Multi-fidelity）：")
        for d_name in self.cfg.d_names:
            print(
                f"  {d_name:6s}  StageB: {r2_stageB[d_name]:.3f}  "
                f"Multi: {r2_multi[d_name]:.3f}"
            )

        return {
            "r2_stageB": r2_stageB,
            "r2_multi": r2_multi,
        }

    # ------------------------------------------------------------------
    # 对外接口：只要多保真均值预测
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        X_raw = np.asarray(X, dtype=np.float64)
        if X_raw.ndim == 1:
            X_raw = X_raw[None, :]
        if X_raw.shape[1] != self.input_dim:
            raise ValueError(
                f"MultiFidelityResidualGP.predict 期望输入维度 {self.input_dim}，"
                f"但得到 {X_raw.shape[1]}。"
            )

        # 标准化输入
        X_scaled = (X_raw - self.x_mean) / self.x_std

        # StageB 低保真预测
        y_sim = self.oracle.predict_d1_3_5_9(X_raw)  # (N,3)

        # 残差 GP 预测的均值
        mu_r = np.zeros_like(y_sim)
        for j, gp in enumerate(self.gps):
            mu_r[:, j] = gp.predict(X_scaled)

        # 多保真均值 = 低保真 + 残差均值
        mu_real = y_sim + mu_r
        # 物理裁剪：d1 不可能为负或超过约束上限
        mu_real = np.clip(mu_real, self.y_min, self.y_max)
        return mu_real

    # ------------------------------------------------------------------
    # 对外接口：多保真均值 + 不确定度（σ）
    # ------------------------------------------------------------------
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_raw = np.asarray(X, dtype=np.float64)
        if X_raw.ndim == 1:
            X_raw = X_raw[None, :]
        if X_raw.shape[1] != self.input_dim:
            raise ValueError(
                f"MultiFidelityResidualGP.predict_with_uncertainty 期望输入维度 {self.input_dim}，"
                f"但得到 {X_raw.shape[1]}。"
            )

        X_scaled = (X_raw - self.x_mean) / self.x_std
        y_sim = self.oracle.predict_d1_3_5_9(X_raw)  # (N,3)

        mu_r = np.zeros_like(y_sim)
        std_r = np.zeros_like(y_sim)
        for j, gp in enumerate(self.gps):
            m, s = gp.predict(X_scaled, return_std=True)
            mu_r[:, j] = m
            std_r[:, j] = s

        mu_real = y_sim + mu_r
        # 物理裁剪：d1 不可能为负或超过约束上限
        mu_real = np.clip(mu_real, self.y_min, self.y_max)
        sigma_real = std_r  # 不确定度完全来自残差 GP 部分

        return mu_real, sigma_real


class BottlePenaltyModel:
    """用静态参数预测瓶型概率的简单模型"""
    def __init__(self, clf, x_mean, x_std):
        self.clf = clf
        self.x_mean = x_mean
        self.x_std = x_std

    def _transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.x_mean) / (self.x_std + 1e-8)

    def predict_prob(self, X):
        Xn = self._transform(X)
        p = self.clf.predict_proba(Xn)[:, 1]
        return np.clip(p, 0.0, 1.0)


def build_bottle_penalty_model(cfg: "ExpConfig",
                               X_new: np.ndarray,
                               bottle_flag: np.ndarray) -> Optional["BottlePenaltyModel"]:
    b = np.asarray(bottle_flag, dtype=np.float64)
    mask = np.isfinite(b)
    X_train = X_new[mask]
    y_train = (b[mask] > 0.5).astype(int)  # 1=瓶型, 0=正常

    if X_train.shape[0] < 5 or len(np.unique(y_train)) < 2:
        print("[build_bottle_penalty_model] 有效样本太少或标签单一，暂不启用瓶型惩罚。")
        return None

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std < 1e-6] = 1.0

    Xn = (X_train - x_mean) / x_std
    clf = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=cfg.seed,
    )
    clf.fit(Xn, y_train)

    model = BottlePenaltyModel(clf, x_mean, x_std)

    p_train = model.predict_prob(X_train)
    print("[build_bottle_penalty_model] 瓶型惩罚模型训练完成。")
    print(f"  训练样本数: {X_train.shape[0]}，阳性比例: {y_train.mean():.3f}")
    print(f"  瓶型样本平均 p: {p_train[y_train==1].mean():.3f}, 非瓶型平均 p: {p_train[y_train==0].mean():.3f}")

    return model

def build_multi_fidelity_surrogate(
    cfg: "ExpConfig",
    oracle: "StageBOracle",
    X_new: np.ndarray,
    d_new: np.ndarray,
    jitter: float = 1e-6,
    n_restarts_optimizer: int = 5,
    param_importance: Optional[np.ndarray] = None,
) -> MultiFidelityResidualGP:
    print("[build_multi_fidelity_surrogate] 使用新表数据训练 Residual GP...")
    surrogate = MultiFidelityResidualGP(
        cfg=cfg,
        oracle=oracle,
        X_train=X_new,
        y_real_train=d_new,
        jitter=jitter,
        n_restarts_optimizer=n_restarts_optimizer,
        param_importance=param_importance,
    )
    print("[build_multi_fidelity_surrogate] 训练完成。")
    return surrogate
# ========================= Block D：候选生成（StageB + Active Subspace） =========================
def clip_and_quantize_static(cfg: "ExpConfig", X_raw: np.ndarray) -> np.ndarray:
    X = np.asarray(X_raw, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    N, D = X.shape

    assert D == len(cfg.static_names), (
        f"clip_and_quantize_static: 维度不匹配，X.shape[1]={D} "
        f"cfg.static_names={len(cfg.static_names)}"
    )

    X_q = X.copy()
    for j, name in enumerate(cfg.static_names):
        p = cfg.static_params[name]
        v = X_q[:, j]
        v = np.clip(v, p["min"], p["max"])
        step = float(p["step"])
        base = float(p["min"])
        # 量化到离 base 最近的 step 网格上
        v = base + np.round((v - base) / step) * step
        # 再做一次裁剪，避免 rounding 超界
        v = np.clip(v, p["min"], p["max"])
        X_q[:, j] = v
    # ---- 额外跨参数约束：SF6 >= C4F8 - 150 ----
    names = list(cfg.static_names)
    if "SF6" in names and "C4F8" in names:
        idx_sf6 = names.index("SF6")
        idx_c4f8 = names.index("C4F8")
        sf6 = X_q[:, idx_sf6]
        c4f8 = X_q[:, idx_c4f8]
        lower = c4f8 - 150.0
        sf6_new = np.maximum(sf6, lower)
        X_q[:, idx_sf6] = sf6_new
    return X_q


def sample_random_static(cfg: "ExpConfig", n: int) -> np.ndarray:
    D = len(cfg.static_names)
    X = np.zeros((n, D), dtype=np.float64)

    rng = np.random.RandomState(cfg.seed)
    for j, name in enumerate(cfg.static_names):
        p = cfg.static_params[name]
        v = rng.rand(n) * (p["max"] - p["min"]) + p["min"]
        X[:, j] = v

    X_q = clip_and_quantize_static(cfg, X)
    return X_q


def stageB_objective_from_preds(cfg: "ExpConfig", d_pred: np.ndarray) -> np.ndarray:
    d_pred = np.asarray(d_pred, dtype=np.float64)
    centers = np.array(
        [cfg.d_constraints[name]["center"] for name in cfg.d_names],
        dtype=np.float64,
    )  # (3,)
    weights = np.array(cfg.d_weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)

    diff2 = (d_pred - centers[None, :]) ** 2  # (N,3)
    J = (weights[None, :] * diff2).sum(axis=1)  # (N,)
    return J


def evaluate_stageB_loss(
    cfg: "ExpConfig",
    oracle: "StageBOracle",
    X: np.ndarray,
    return_pred: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    d_pred = oracle.predict_d1_3_5_9(X)  # (N,3)
    J_sim = stageB_objective_from_preds(cfg, d_pred)
    if return_pred:
        return J_sim, d_pred
    else:
        return J_sim, None


def refine_candidates_with_stageB(
    cfg: "ExpConfig",
    oracle: "StageBOracle",
    X_init: np.ndarray,
    as_info: Optional[Dict] = None,
    n_steps: int = 3,
    lr: float = 0.5,
    subspace_dim: int = 1,
) -> np.ndarray:
    X0 = np.asarray(X_init, dtype=np.float64)
    if X0.ndim == 1:
        X0 = X0[None, :]
    N, D = X0.shape

    # Active Subspace 投影矩阵（在归一化坐标系）
    V_sub = None
    if as_info is not None and subspace_dim > 0:
        eigvecs = np.asarray(as_info["eigenvectors"], dtype=np.float64)  # (D,D)
        subspace_dim = min(subspace_dim, eigvecs.shape[1])
        V_sub = eigvecs[:, :subspace_dim]  # (D,k)

    # 统一的重要度和各向异性 trust-region 半径
    param_importance = None
    trust_radius_norm = None
    if as_info is not None:
        if "param_importance_combined" in as_info:
            param_importance = np.asarray(
                as_info["param_importance_combined"], dtype=np.float64
            )
        elif "param_importance_first" in as_info:
            param_importance = np.asarray(
                as_info["param_importance_first"], dtype=np.float64
            )
        if "trust_radius_norm" in as_info:
            trust_radius_norm = np.asarray(
                as_info["trust_radius_norm"], dtype=np.float64
            )
    if as_info is not None and subspace_dim > 0:
        eigvecs = np.asarray(as_info["eigenvectors"], dtype=np.float64)  # (D,D)
        subspace_dim = min(subspace_dim, eigvecs.shape[1])
        V_sub = eigvecs[:, :subspace_dim]  # (D,k)

    device = oracle.device
    model = oracle.model
    mean = oracle.static_mean       # (1,D) on device
    std = oracle.static_std         # (1,D) on device
    phys_template = oracle.phys_template  # (1,2,T)
    tvals_template = oracle.tvals_template  # (T,)
    idx_d1 = oracle.idx_d1

    # d 目标中心 & 权重（全部放在 device 上的 torch.tensor）
    centers = torch.tensor(
        [cfg.d_constraints[name]["center"] for name in cfg.d_names],
        dtype=torch.float32,
        device=device,
    )  # (3,)

    w_d = torch.tensor(
        cfg.d_weights,  # 比如 [0.3, 0.4, 0.3]
        dtype=torch.float32,
        device=device,
    )  # (3,)
    w_d = w_d / (w_d.sum() + 1e-12)

    # 时间索引（3,5,9）
    tidx_list = [T2IDX[t] for t in ["3", "5", "9"]]

    X_cur = clip_and_quantize_static(cfg, X0)  # 先保证合法
    X_refined = np.zeros_like(X_cur, dtype=np.float64)

    for i in range(N):
        x_raw_t = torch.from_numpy(X0[i].astype(np.float32)).to(device).view(1, -1)
        # 记录归一化基点，用于 trust-region 约束
        x_norm_base = ((x_raw_t - mean) / (std + 1e-8)).detach().cpu().numpy()[0]
        for _ in range(n_steps):
            # 归一化静态参数
            x_norm = ((x_raw_t - mean) / (std + 1e-8)).clone().detach()
            x_norm.requires_grad_(True)
            # 使用固定的平均 phys_template & tvals_template
            B = x_norm.size(0)  # 1
            phys = phys_template.expand(B, -1, -1)          # (1,2,T)
            tvals = tvals_template.unsqueeze(0).expand(B, -1)  # (1,T)
            y = model(x_norm, phys, tvals)                  # (1,K,T)
            d1 = y[:, idx_d1, tidx_list]                    # (1,3)
            diff2 = (d1 - centers.view(1, -1)) ** 2         # (1,3)
            loss = (w_d.view(1, -1) * diff2).sum()            # scalar
            loss.backward()
            g_norm = x_norm.grad.detach().cpu().numpy()[0]  # (D,)

            # 若有 Active Subspace，则把梯度投影到低维子空间
            if V_sub is not None:
                g_norm = V_sub @ (V_sub.T @ g_norm)

            # 各向异性梯度缩放：重要参数步长更大
            if param_importance is not None:
                imp_scale = param_importance / (param_importance.mean() + 1e-12)  # numpy (D,)
                g_norm = g_norm * imp_scale  # 这里都是 numpy
            # 在归一化空间做梯度下降
            x_norm_np = x_norm.detach().cpu().numpy()[0]
            x_norm_new = x_norm_np - lr * g_norm

            # 各向异性 trust-region 约束：限制相对初始点的增量
            if trust_radius_norm is not None:
                delta = x_norm_new - x_norm_base
                delta = np.clip(delta, -trust_radius_norm, trust_radius_norm)
                x_norm_new = x_norm_base + delta

            # 映射回原始空间
            mean_np = mean.detach().cpu().numpy()[0]
            std_np = std.detach().cpu().numpy()[0]
            x_raw_new = mean_np + x_norm_new * std_np  # (D,)

            # 裁剪 + 量化
            x_raw_new = clip_and_quantize_static(cfg, x_raw_new)[0]

            # 准备下一个 step
            x_raw_t = torch.from_numpy(
                x_raw_new.astype(np.float32)
            ).to(device).view(1, -1)

        X_refined[i] = x_raw_t.detach().cpu().numpy()[0]
    return X_refined


def generate_candidates_with_stageB(
    cfg: "ExpConfig",
    oracle: "StageBOracle",
    as_info: Optional[Dict] = None,
    oversample_factor: float = 2.0,
    stageB_keep_ratio: float = 0.5,
    n_grad_steps: int = 3,
    grad_lr: float = 0.5,
    subspace_dim: int = 1,
    n_refine_max: int = 512,
) -> Dict[str, np.ndarray]:
    # ---------- 1. 基础随机采样 ----------
    N0 = int(cfg.n_candidates * oversample_factor)
    print(f"[generate_candidates_with_stageB] 基础采样 N0={N0} ...")
    X0 = sample_random_static(cfg, N0)  # (N0,D)

    # ---------- 2. StageB 快速筛选 ----------
    print("[generate_candidates_with_stageB] StageB 评估损失并筛选候选 ...")
    J_all, d_all = evaluate_stageB_loss(cfg, oracle, X0, return_pred=True)  # (N0,), (N0,3)

    order = np.argsort(J_all)  # 损失越小越好
    N_keep = max(cfg.n_candidates, int(stageB_keep_ratio * N0))
    N_keep = min(N_keep, N0)
    idx_keep = order[:N_keep]

    X_top = X0[idx_keep]       # (N_keep,D)
    J_top = J_all[idx_keep]    # (N_keep,)
    d_top = d_all[idx_keep]    # (N_keep,3)

    print(
        f"[generate_candidates_with_stageB] "
        f"从 {N0} 个候选中保留 {N_keep} 个 StageB 低损失样本，"
        f"J_stageB ∈ [{J_top.min():.3e}, {J_top.max():.3e}]"
    )

    # ---------- 3. 梯度微调 ----------
    N_refine = min(n_refine_max, X_top.shape[0])
    if N_refine > 0 and n_grad_steps > 0:
        print(
            f"[generate_candidates_with_stageB] "
            f"对其中前 {N_refine} 个做 {n_grad_steps} 步梯度微调 "
            f"(lr={grad_lr}, subspace_dim={subspace_dim}) ..."
        )
        X_refine_input = X_top[:N_refine]
        X_refined = refine_candidates_with_stageB(
            cfg=cfg,
            oracle=oracle,
            X_init=X_refine_input,
            as_info=as_info,
            n_steps=n_grad_steps,
            lr=grad_lr,
            subspace_dim=subspace_dim,
        )
        # 重新计算这些微调后点的 StageB 损失（仅用于记录）
        J_ref, d_ref = evaluate_stageB_loss(cfg, oracle, X_refined, return_pred=True)
        print(
            "[generate_candidates_with_stageB] "
            f"微调后 J_stageB ∈ [{J_ref.min():.3e}, {J_ref.max():.3e}]"
        )

        # 合并微调 + 未微调部分
        if N_refine < X_top.shape[0]:
            X_candidates = np.vstack([X_refined, X_top[N_refine:]])
        else:
            X_candidates = X_refined
    else:
        print("[generate_candidates_with_stageB] 跳过梯度微调，直接使用 StageB 筛选结果。")
        X_candidates = X_top

    result = {
        "X_candidates": X_candidates,
        "X_stageB_all": X0,
        "J_stageB_all": J_all,
        "X_stageB_top": X_top,
        "J_stageB_top": J_top,
        "d_stageB_top": d_top,
    }
    return result

# ========================= Block E：EI 采集函数 + StageB 先验 =========================
def _normal_pdf(z: np.ndarray) -> np.ndarray:
    """标准正态分布密度函数 φ(z)。"""
    return np.exp(-0.5 * z ** 2) / math.sqrt(2.0 * math.pi)


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    """标准正态分布分布函数 Φ(z)，支持标量和向量。"""
    z = np.asarray(z, dtype=np.float64)
    # 用 np.vectorize 包一层，把 math.erf 变成逐元素 ufunc
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(z / math.sqrt(2.0)))



def scalar_objective_from_mu_sigma(
    cfg: "ExpConfig",
    mu_real: np.ndarray,
    sigma_real: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mu_real = np.asarray(mu_real, dtype=np.float64)
    sigma_real = np.asarray(sigma_real, dtype=np.float64)
    assert mu_real.shape == sigma_real.shape

    centers = np.array(
        [cfg.d_constraints[name]["center"] for name in cfg.d_names],
        dtype=np.float64,
    )  # (3,)
    weights = np.array(cfg.d_weights, dtype=np.float64)
    weights = weights / (weights.sum() + 1e-12)

    # J 的均值：直接用 μ 代入
    diff = (mu_real - centers[None, :])  # (N,3)
    J_mu = (weights[None, :] * (diff ** 2)).sum(axis=1)  # (N,)

    # J 的方差近似
    grad_J = 2.0 * weights[None, :] * diff             # (N,3)
    J_var = ((grad_J ** 2) * (sigma_real ** 2)).sum(axis=1)  # (N,)
    J_var = np.maximum(J_var, 0.0)
    J_sigma = np.sqrt(J_var + 1e-18)

    return J_mu, J_sigma


def expected_improvement_for_minimization(
    J_mu: np.ndarray,
    J_sigma: np.ndarray,
    J_best: float,
) -> np.ndarray:
    J_mu = np.asarray(J_mu, dtype=np.float64)
    J_sigma = np.asarray(J_sigma, dtype=np.float64)
    assert J_mu.shape == J_sigma.shape

    EI = np.zeros_like(J_mu)
    # sigma 非零的索引
    mask = J_sigma > 1e-12
    if not np.any(mask):
        # 全部 sigma≈0，退化为 exploitation
        diff = J_best - J_mu
        EI = np.maximum(diff, 0.0)
        return EI

    J_mu_m = J_mu[mask]
    J_sigma_m = J_sigma[mask]

    Z = (J_best - J_mu_m) / J_sigma_m
    Phi = _normal_cdf(Z)
    phi = _normal_pdf(Z)

    EI_m = (J_best - J_mu_m) * Phi + J_sigma_m * phi
    EI_m = np.maximum(EI_m, 0.0)

    EI[mask] = EI_m
    # 对于 sigma≈0 的点，直接用改善值
    if np.any(~mask):
        diff = J_best - J_mu[~mask]
        EI[~mask] = np.maximum(diff, 0.0)

    return EI


def compute_stageB_prior_weight(
    cfg: "ExpConfig",
    stageB_data: dict,
    X_candidates: np.ndarray,
    length_scale: float = 1.0,
    prior_floor: float = 0.1,
) -> np.ndarray:
    X_cand = np.asarray(X_candidates, dtype=np.float64)
    if X_cand.ndim == 1:
        X_cand = X_cand[None, :]

    X_old = stageB_data["X_all_raw"].cpu().numpy().astype(np.float64)  # (N_old,D)
    # 标准化到旧表的统计量
    mean_old = X_old.mean(axis=0, keepdims=True)
    std_old = X_old.std(axis=0, keepdims=True)
    std_old[std_old < 1e-6] = 1.0

    X_old_n = (X_old - mean_old) / std_old
    X_cand_n = (X_cand - mean_old) / std_old

    N_cand = X_cand_n.shape[0]
    prior = np.zeros(N_cand, dtype=np.float64)

    # 简单的 brute-force 最近邻距离（旧表数据量不大时可接受）
    for i in range(N_cand):
        diff = X_old_n - X_cand_n[i : i + 1]        # (N_old,D)
        dist2 = np.sum(diff ** 2, axis=1)           # (N_old,)
        d_min = math.sqrt(float(dist2.min()) + 1e-12)
        prior_i = math.exp(-0.5 * (d_min / length_scale) ** 2)
        prior[i] = prior_i

    # 截断下限，避免完全消失
    prior = np.maximum(prior, prior_floor)
    return prior


def score_candidates_with_EI_and_prior(
        cfg: "ExpConfig",
        surrogate: "MultiFidelityResidualGP",
        X_candidates: np.ndarray,
        stageB_data: Optional[dict],
        J_best: Optional[float] = None,
        prior_length_scale: float = 1.0,
        prior_floor: float = 0.1,
        bottle_model: Optional["BottlePenaltyModel"] = None,
        bottle_penalty_weight: float = 1.0,
    ) -> dict:
    X_cand = np.asarray(X_candidates, dtype=np.float64)
    if X_cand.ndim == 1:
        X_cand = X_cand[None, :]

    mu_real, sigma_real = surrogate.predict_with_uncertainty(X_cand)
    J_mu_raw, J_sigma = scalar_objective_from_mu_sigma(cfg, mu_real, sigma_real)
    J_mu = J_mu_raw.copy()
    # 先算一遍训练集的原始 J
    J_train_raw = stageB_objective_from_preds(cfg, surrogate.y_real_train)

    bottle_prob_cand = np.zeros_like(J_mu_raw)
    if bottle_model is not None and bottle_penalty_weight > 0:
        # 用训练集的典型尺度做缩放
        J_ref = float(max(np.median(J_train_raw), 1e-6))

        # 候选点的瓶型概率 & 惩罚
        bottle_prob_cand = bottle_model.predict_prob(X_cand)
        penalty_cand = bottle_penalty_weight * J_ref * bottle_prob_cand
        J_mu = J_mu + penalty_cand

        # 训练点上的惩罚，用来定义 J_best
        bottle_prob_train = bottle_model.predict_prob(surrogate.X_train_raw)
        penalty_train = bottle_penalty_weight * J_ref * bottle_prob_train
        J_train_eff = J_train_raw + penalty_train
    else:
        J_train_eff = J_train_raw
    # 3) 当前已知最佳 J_best（来自高保真数据）
    if J_best is None:
        J_best = float(J_train_eff.min())
        print(f"[score_candidates_with_EI_and_prior] 自动推得 J_best(含瓶型惩罚) = {J_best:.3e}")

    EI = expected_improvement_for_minimization(J_mu, J_sigma, J_best)
    # 5) StageB 分布先验
    if stageB_data is not None:
        prior = compute_stageB_prior_weight(
            cfg, stageB_data, X_cand,
            length_scale=prior_length_scale,
            prior_floor=prior_floor,
        )
    else:
        prior = np.ones_like(EI)

    # 6) 综合分数：EI × StageB 先验 × 瓶型“可行概率”
    if bottle_model is not None and bottle_penalty_weight > 0:
        # 上面已经算过 bottle_prob_cand
        p_feas = 1.0 - bottle_prob_cand
    else:
        p_feas = np.ones_like(EI)
    score = EI * prior * p_feas
    info = {
        "X_candidates": X_cand,
        "mu_real": mu_real,
        "sigma_real": sigma_real,
        "J_mu": J_mu,  # 已含惩罚
        "J_mu_raw": J_mu_raw,  # 原始 d1 目标
        "bottle_prob": bottle_prob_cand,  # 候选点瓶型概率
        "J_sigma": J_sigma,
        "EI": EI,
        "prior": prior,
        "score": score,
        "J_best": J_best,
    }
    return info


def select_topK_recipes(
    cfg: "ExpConfig",
    score_info: dict,
    topK: Optional[int] = None,
) -> dict:
    X_cand = score_info["X_candidates"]
    score = score_info["score"]
    J_mu = np.asarray(score_info["J_mu"], dtype=np.float64)
    N = X_cand.shape[0]
    ...
    if topK is None:
        topK = cfg.n_select
    topK = min(topK, N)

    # 先按 J_mu 从小到大排序，再用 score 做 tie-break（score 大者优先）
    # 这样保证推荐点是“预测损失尽量小”的，同时兼顾 EI / 先验信息。
    order = np.lexsort((-score, J_mu))
    idx_sel = order[:topK]

    # 如果最好的候选 J_mu 仍明显大于当前已知 J_best，给出一个提示
    J_best = float(score_info["J_best"])
    J_mu_best_cand = float(J_mu[idx_sel[0]])
    if J_mu_best_cand > 1.5 * J_best:
        print(
            "[select_topK_recipes] 警告：候选集中最小的 J_mu 仍明显大于当前已知 J_best，"
            "本轮推荐点主要用于探索而非显著优于现有最优 recipe。"
        )

    result = {
        "X_selected": X_cand[idx_sel],
        "mu_real": score_info["mu_real"][idx_sel],
        "sigma_real": score_info["sigma_real"][idx_sel],
        "EI": score_info["EI"][idx_sel],
        "prior": score_info["prior"][idx_sel],
        "score": score[idx_sel],
        "indices": idx_sel,
        "J_best": score_info["J_best"],
    }
    return result
# ========================= Block F：结果导出 & 主流程 =========================

def save_results_advanced(
    cfg: "ExpConfig",
    X_new: np.ndarray,
    d_new: np.ndarray,
    stageB_data: dict,
    corr_info: dict,
    as_info: dict,
    surrogate: "MultiFidelityResidualGP",
    cand_info: dict,
    score_info: dict,
    selected: dict,
    oracle: "StageBOracle",
):
    ensure_dir(cfg.save_dir)
    excel_path = os.path.join(cfg.save_dir, "stageC_advanced_selection.xlsx")
    csv_path = os.path.join(cfg.save_dir, "推荐recipe清单.csv")

    # -------- sheet1: 推荐清单 --------
    X_sel = selected["X_selected"]          # (K,D)
    mu_sel = selected["mu_real"]            # (K,3)
    sigma_sel = selected["sigma_real"]      # (K,3)
    EI_sel = selected["EI"]                 # (K,)
    prior_sel = selected["prior"]           # (K,)
    score_sel = selected["score"]           # (K,)

    # StageB 对推荐点的预测
    d_stageB_sel = oracle.predict_d1_3_5_9(X_sel)  # (K,3)

    rec_rows = []
    for i in range(X_sel.shape[0]):
        row = {
            "序号": i + 1,
        }
        # 静态参数
        for j, name in enumerate(cfg.static_names):
            row[name] = float(X_sel[i, j])

        # 多保真预测
        for j, d_name in enumerate(cfg.d_names):
            row[f"多保真预测{d_name}(μm)"] = float(mu_sel[i, j])
            row[f"多保真不确定度{d_name}"] = float(sigma_sel[i, j])

        # StageB 单独预测
        for j, d_name in enumerate(cfg.d_names):
            row[f"StageB预测{d_name}(μm)"] = float(d_stageB_sel[i, j])

        # EI / 先验 / 综合分数
        row["EI"] = float(EI_sel[i])
        row["StageB先验"] = float(prior_sel[i])
        row["综合评分"] = float(score_sel[i])

        rec_rows.append(row)

    df_rec = pd.DataFrame(rec_rows)

    # -------- sheet2: 候选池评分 --------
    X_cand = score_info["X_candidates"]   # (N,D)
    mu_cand = score_info["mu_real"]       # (N,3)
    sigma_cand = score_info["sigma_real"] # (N,3)
    J_mu = score_info["J_mu"]             # (N,)
    J_sigma = score_info["J_sigma"]       # (N,)
    EI_all = score_info["EI"]             # (N,)
    prior_all = score_info["prior"]       # (N,)
    score_all = score_info["score"]       # (N,)

    cand_rows = []
    for i in range(X_cand.shape[0]):
        row = {
            "候选序号": i,
            "J_mu": float(J_mu[i]),
            "J_sigma": float(J_sigma[i]),
            "EI": float(EI_all[i]),
            "StageB先验": float(prior_all[i]),
            "综合评分": float(score_all[i]),
        }
        for j, name in enumerate(cfg.static_names):
            row[name] = float(X_cand[i, j])
        for j, d_name in enumerate(cfg.d_names):
            row[f"多保真预测{d_name}(μm)"] = float(mu_cand[i, j])
            row[f"多保真不确定度{d_name}"] = float(sigma_cand[i, j])
        cand_rows.append(row)
    df_cand = pd.DataFrame(cand_rows)

    # -------- sheet3: 新表实测 --------
    df_new = pd.DataFrame(
        np.concatenate([X_new, d_new], axis=1),
        columns=list(cfg.static_names) + [f"实测{d}" for d in cfg.d_names],
    )

    # -------- sheet4: StageB 相关性 --------
    pearson_df = corr_info.get("pearson", None)
    spearman_df = corr_info.get("spearman", None)

    # -------- sheet5: Active Subspace --------
    eigvals = np.asarray(as_info["eigenvalues"], dtype=np.float64)
    eigfrac = np.asarray(as_info["eigenvalues_frac"], dtype=np.float64)
    param_names = list(as_info["param_names"])
    importance_first = np.asarray(as_info["param_importance_first"], dtype=np.float64)

    df_eig = pd.DataFrame(
        {
            "eig_index": np.arange(len(eigvals)),
            "eigenvalue": eigvals,
            "energy_frac": eigfrac,
        }
    )
    df_imp = pd.DataFrame(
        {
            "param_name": param_names,
            "importance_in_first_subspace": importance_first,
        }
    )

    # -------- sheet6: Surrogate R² --------
    diag = surrogate.diagnostics
    rows_r2 = []
    for d_name in cfg.d_names:
        rows_r2.append(
            {
                "目标": d_name,
                "StageB_R2_在新表上": diag["r2_stageB"][d_name],
                "MultiFidelity_R2_在新表上": diag["r2_multi"][d_name],
            }
        )
    df_r2 = pd.DataFrame(rows_r2)

    # -------- sheet7: StageB筛选 --------
    X_stageB_top = cand_info["X_stageB_top"]
    J_stageB_top = cand_info["J_stageB_top"]
    d_stageB_top = cand_info["d_stageB_top"]

    rows_stageB = []
    for i in range(X_stageB_top.shape[0]):
        row = {
            "序号": i,
            "StageB_loss": float(J_stageB_top[i]),
        }
        for j, name in enumerate(cfg.static_names):
            row[name] = float(X_stageB_top[i, j])
        for j, d_name in enumerate(cfg.d_names):
            row[f"StageB预测{d_name}(μm)"] = float(d_stageB_top[i, j])
        rows_stageB.append(row)
    df_stageB = pd.DataFrame(rows_stageB)

    # -------- 写 Excel --------
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_rec.to_excel(writer, sheet_name="推荐清单", index=False)
        df_cand.to_excel(writer, sheet_name="候选池评分", index=False)
        df_new.to_excel(writer, sheet_name="新表实测", index=False)

        if pearson_df is not None:
            pearson_df.to_excel(writer, sheet_name="StageB_Pearson", index=True)
        if spearman_df is not None:
            spearman_df.to_excel(writer, sheet_name="StageB_Spearman", index=True)

        df_eig.to_excel(writer, sheet_name="ActiveSubspace_特征值", index=False)
        df_imp.to_excel(writer, sheet_name="ActiveSubspace_主方向", index=False)
        df_r2.to_excel(writer, sheet_name="Surrogate_R2", index=False)
        df_stageB.to_excel(writer, sheet_name="StageB筛选候选", index=False)

    # -------- 写 CSV（推荐清单简化版） --------
    csv_rows = []
    for i in range(X_sel.shape[0]):
        row = {
            "序号": i + 1,
        }
        for j, name in enumerate(cfg.static_names):
            row[name] = float(X_sel[i, j])
        for j, d_name in enumerate(cfg.d_names):
            row[f"多保真预测{d_name}(μm)"] = float(mu_sel[i, j])
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n[save_results_advanced] 结果已保存至：{cfg.save_dir}")
    print(f"  - 详细结果：{os.path.basename(excel_path)}")
    print(f"  - 推荐清单：{os.path.basename(csv_path)}")

    # 控制台简单打印一下 StageB 相关性 & Active Subspace 信息
    if pearson_df is not None:
        print("\n[StageB 参数-目标 Pearson 相关系数]：")
        print(pearson_df)

    print("\n[Active Subspace 主方向权重]：")
    for name, w in zip(param_names, importance_first):
        print(f"  {name:10s}: {w:.3f}")


def run_advanced_selection(cfg: "ExpConfig") -> dict:
    np.random.seed(cfg.seed)
    # 如需结果完全可复现，可以同步设置 torch 随机种子
    try:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    except Exception:
        pass

    print("\n========== Block A：加载数据 ==========")
    stageB_data = load_stageB_dataset(cfg)
    X_new, d_new, bottle_flag = load_new_table_d1(cfg)

    print("\n========== Block B.1：加载 StageB 模型 ==========")
    oracle = StageBOracle(
        ckpt_path=cfg.stageB_morph_ckpt,
        old_excel_path=cfg.old_excel,
    )

    print("\n========== Block B.2：StageB 敏感度 & Active Subspace ==========")
    corr_info = compute_stageB_correlations(stageB_data, cfg)
    as_info = estimate_active_subspace_from_stageB(
        oracle,
        cfg,
        n_samples=512,
        batch_size=64,
        target_time="9")
    as_info = compute_param_importance_and_trust_region(cfg, corr_info, as_info)

    print("\n========== Block C：构建多保真 Residual GP ==========")
    surrogate = build_multi_fidelity_surrogate(cfg, oracle, X_new, d_new, param_importance=as_info.get("param_importance_combined", None))

    print("\n========== Block C+：瓶型惩罚模型 ==========")
    bottle_model = build_bottle_penalty_model(cfg, X_new, bottle_flag)

    print("\n========== Block D：生成候选点 ==========")
    cand_info = generate_candidates_with_stageB(
        cfg=cfg,
        oracle=oracle,
        as_info=as_info)

    print("\n========== Block E：多保真打分 + 先验 + 瓶型惩罚 ==========")
    score_info = score_candidates_with_EI_and_prior(
        cfg=cfg,
        surrogate=surrogate,
        X_candidates=cand_info["X_candidates"],
        stageB_data=stageB_data,
        J_best=None,
        prior_length_scale=1.0,
        prior_floor=0.1,
        bottle_model=bottle_model,
        bottle_penalty_weight=cfg.bottle_penalty_weight,
    )
    selected = select_topK_recipes(cfg, score_info, topK=cfg.n_select)
    print("\n========== Block F：保存结果 ==========")
    save_results_advanced(
        cfg=cfg,
        X_new=X_new,
        d_new=d_new,
        stageB_data=stageB_data,
        corr_info=corr_info,
        as_info=as_info,
        surrogate=surrogate,
        cand_info=cand_info,
        score_info=score_info,
        selected=selected,
        oracle=oracle,
    )

    return {
        "cfg": cfg,
        "stageB_data": stageB_data,
        "X_new": X_new,
        "d_new": d_new,
        "oracle": oracle,
        "corr_info": corr_info,
        "as_info": as_info,
        "surrogate": surrogate,
        "cand_info": cand_info,
        "score_info": score_info,
        "selected": selected,
    }
if __name__ == "__main__":
    # 想对比的瓶型惩罚权重
    penalty_list = [0.0, 1.0, 2.0]

    for w in penalty_list:
        cfg = ExpConfig()
        cfg.bottle_penalty_weight = w

        # 每个权重单独一个保存目录，避免 Excel / CSV 覆盖
        cfg.save_dir = f"./stageC_BO_penalty_{w:g}"
        ensure_dir(cfg.save_dir)

        print("\n" + "=" * 80)
        print(f"运行主动选点：bottle_penalty_weight = {w}")
        print("=" * 80)

        _ = run_advanced_selection(cfg)