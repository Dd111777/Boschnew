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
    """
    StageB + StageC 多保真主动选点器的全局配置（Block A）

    只负责：
      - Excel 路径
      - 静态参数范围/步长
      - d1_3 / d1_5 / d1_9 的目标区间 & 权重
      - 保存目录
    后续 Block B/C/D/E 会在此基础上继续补充。
    """

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
        "source_RF": {"min": 500, "max": 3500, "step": 100, "unit": "W"},
        "LF_RF": {"min": 25, "max": 150, "step": 5, "unit": "W"},
        "SF6": {"min": 50, "max": 500, "step": 50, "unit": "sccm"},
        "C4F8": {"min": 50, "max": 500, "step": 50, "unit": "sccm"},
        "DEP_time": {"min": 0.4, "max": 4.0, "step": 0.4, "unit": "s"},
        "etch_time": {"min": 0.4, "max": 4.0, "step": 0.4, "unit": "s"},
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
    n_candidates: int = 2000
    n_select: int = 10
    n_rounds: int = 1  # 当前场景：一次性推荐即可

    # 随机种子（后续 Block 可扩展成 seeds 列表）
    seed: int = 42


# ============================================================
# 旧表：StageB 数据入口
# ============================================================

def load_stageB_dataset(config: ExpConfig) -> Dict[str, object]:
    """
    从旧表 case.xlsx 读取 StageB 训练用的数据，并抽取：
      - 全部静态输入（标准化 & 反标准化）
      - d1 family 在 t=3/5/9 的物理量（去掉缺失的样本）

    返回一个 dict，供后续：
      - 相关性分析（X vs d1_3/5/9）
      - Active Subspace / 梯度敏感度（Block B）
      - 多保真 GP 残差拟合（Block C）
    使用。
    """
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
    """
    从 Bosch 新表中读取：
      - 静态参数 X_new: (N, 7)
      - 实测/仿真的 d1_3 / d1_5 / d1_9: (N, 3)

    逻辑与现有 stageCselection.py 中的 load_real_measurements 保持一致：
      - 利用 load_new_excel_as_sparse_morph 做统一解析
      - 只保留 d1_3/5/9 都存在的样本
      - 单位与 physio_util 中的转换保持一致（nm → μm）
    """
    recs = load_new_excel_as_sparse_morph(
        config.new_excel,
        height_family="h1",  # 与 StageB/C 形貌高度 family 保持一致
    )

    X_list: List[np.ndarray] = []
    d_list: List[List[float]] = []

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

    X_new = np.asarray(X_list, dtype=np.float32)  # (N, 7)
    d_new = np.asarray(d_list, dtype=np.float32)  # (N, 3)

    # 只保留三维 d 都不为 NaN 的样本
    mask_valid = np.isfinite(d_new).all(axis=1)
    X_new = X_new[mask_valid]
    d_new = d_new[mask_valid]

    print(f"[load_new_table_d1] 加载 Bosch 新表有效样本数：{X_new.shape[0]}")
    if X_new.shape[0] == 0:
        raise ValueError(
            f"从 {config.new_excel} 中未找到同时包含 d1_3/5/9 的有效样本，请检查新表内容。"
        )

    return X_new, d_new

# ========================= Block B.1：StageB 模型包装 =========================
# 封装 StageB 形貌网络，提供：
# - 从 StageB 的 morph_best_overall.pth 加载模型与 meta
# - 给定静态参数 X_static → 预测完整 (B,K,T) 形貌序列
# - 只取 d1 在 t=3/5/9 的值作为低保真 “物理老师” 输出

# 默认的 StageB 检查点路径（与 StageB 训练脚本保持一致）
STAGEB_DEFAULT_CKPT = os.path.join(Bcfg.save_dir, "morph_best_overall.pth")


class StageBOracle:
    """
    StageB 形貌网络封装：
    - 从 StageB 的 best_overall ckpt 加载模型 + meta（含 norm_static, time_values 等）
    - 内部构造一个“平均物理时序” phys_template，用于新 recipe 的快速预测
    - 对外提供：
        * predict_full_profile(X_static) → (B,K,T)
        * predict_d1_3_5_9(X_static)    → (B,3)  对应 d1_(3,5,9)
    """

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
    # ------------------------------------------------------------------
    # 内部：加载模型和 meta
    # ------------------------------------------------------------------
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
        """
        从 StageB 训练用的旧表中：
        - 取出 phys_seq (N,2,T)，在 N 维度做均值 → (1,2,T)，作为 phys_template
        - 取出一行 tvals_mat (T,) 作为时间刻度模板
        这样对新 recipe，只需替换静态参数，物理时序用一个“典型 profile” 近似。
        """
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
        """
        输入：
          X_static: (B,7) numpy，列顺序需与 excel_to_morph_dataset_from_old 的 static_keys 对齐：
            [APC, source_RF, LF_RF, SF6, C4F8, DEP_time, etch_time]

        输出：
          y_pred: (B,K,T) numpy，对应 families=self.family_names，时间轴与 TIME_LIST 对齐
        """
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
        """
        输入：
          X_static: (B,7) numpy
          time_ids: 要抽取的时间刻度标签，默认 ("3","5","9")

        输出：
          d_vals: (B, len(time_ids)) numpy，对应 d1_time_ids
        """
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
    """
    经典相关性分析（基于旧表 StageB 数据）：
      - 输入：
          stageB_data: load_stageB_dataset(cfg) 的返回
          cfg        : ExpConfig，全局配置（主要用 static_names / d_names）
      - 输出：
          {
            "pearson":  pd.DataFrame (index=static_names, columns=d_names),
            "spearman": Optional[pd.DataFrame]  # 若缺少 scipy 则为 None
          }
    """
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
    """
    基于 StageB 模型的 Active Subspace 估计：
      - 利用旧表 (static_norm, phys_seq, time_mat)，抽样 n_samples 个 recipe
      - 用 autograd 计算 d1_{target_time} 对静态输入（归一化空间）的梯度
      - 聚合得到 C = 1/M Σ g_i g_i^T，并做特征分解
      - 输出：
          {
            "eigenvalues":      (D,)  从大到小
            "eigenvalues_frac": (D,)  归一化能量占比
            "eigenvectors":     (D,D) 列向量为特征向量 v_k（对应 eigenvalues[k]）
            "param_importance_first": (D,) 参数在第一主方向上的绝对权重
            "param_names":      list[str]，与 cfg.static_names 一致
            "target_time":      str
          }
    """
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

        y = model(x_norm, phys, tvals)                   # (B,K,T)
        # d1 family 的 index
        idx_d1 = oracle.idx_d1
        y_d1_t = y[:, idx_d1, tidx]                      # (B,)

        # 对所有样本求平均再回传，数值更稳定
        val = y_d1_t.mean()
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
# ========================= Block C：多保真 Residual GP 代理 =========================
class MultiFidelityResidualGP:
    """
    多保真 surrogate 封装：
      - 低保真：StageBOracle.predict_d1_3_5_9(X)  → y_sim(X)
      - 高保真：新表实测 d_new(X)                → y_real(X)
      - 残差： r(X) = y_real(X) - y_sim(X)
      - 对 r_j(X) 训练 3 个独立的 Gaussian Process：
            r_j ~ GP_j(X)

    对外接口：
      - predict(X) → μ_real(X) = y_sim(X) + μ_r(X)
      - predict_with_uncertainty(X) → (μ_real(X), σ_real(X))
      - diagnostics: StageB & 多保真 surrogate 的 R² 对比
    """

    def __init__(
        self,
        cfg: "ExpConfig",
        oracle: "StageBOracle",
        X_train: np.ndarray,
        y_real_train: np.ndarray,
        jitter: float = 1e-6,
        n_restarts_optimizer: int = 5,
    ):
        """
        参数：
          cfg                 : 全局配置（只用到 d_names）
          oracle              : StageBOracle 实例，用于计算 y_sim
          X_train             : (N,7) 新表静态参数
          y_real_train        : (N,3) 新表实测 d1_3/5/9
          jitter              : GP 中的 alpha，数值稳定用
          n_restarts_optimizer: GP 超参数优化的重启次数
        """
        self.cfg = cfg
        self.oracle = oracle

        X = np.asarray(X_train, dtype=np.float64)
        Y_real = np.asarray(y_real_train, dtype=np.float64)
        assert X.shape[0] == Y_real.shape[0], "X_train 与 y_real_train 样本数不一致"

        self.X_train_raw = X
        self.y_real_train = Y_real
        self.input_dim = X.shape[1]
        self.output_dim = Y_real.shape[1]  # 3，对应 d1_3/5/9

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

        # 为每个输出维度建一个 GP
        self.gps: list[GaussianProcessRegressor] = []
        for j in range(self.output_dim):
            kernel = (
                C(1.0, (1e-3, 1e3))
                * RBF(length_scale=np.ones(self.input_dim), length_scale_bounds=(1e-2, 1e3))
                + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e0))
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=jitter,
                normalize_y=True,
                n_restarts_optimizer=n_restarts_optimizer,
                random_state=cfg.seed,
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
        """
        输入：
          X: (N,D) 或 (D,) numpy，静态参数（原始尺度）

        输出：
          μ_real: (N,3) numpy，预测的 d1_3/5/9 多保真均值
        """
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
        return mu_real

    # ------------------------------------------------------------------
    # 对外接口：多保真均值 + 不确定度（σ）
    # ------------------------------------------------------------------
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        输入：
          X: (N,D) 或 (D,) numpy

        输出：
          μ_real: (N,3)  多保真均值
          σ_real: (N,3)  来自残差 GP 的标准差（StageB 为确定性模型）
        """
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
        sigma_real = std_r  # 不确定度完全来自残差 GP 部分

        return mu_real, sigma_real


def build_multi_fidelity_surrogate(
    cfg: "ExpConfig",
    oracle: "StageBOracle",
    X_new: np.ndarray,
    d_new: np.ndarray,
    jitter: float = 1e-6,
    n_restarts_optimizer: int = 5,
) -> MultiFidelityResidualGP:
    """
    工具函数：给定配置 + StageBOracle + 新表数据，一步构建多保真 surrogate。

    输入：
      cfg       : ExpConfig
      oracle    : StageBOracle
      X_new     : (N,7) 新表静态参数
      d_new     : (N,3) 新表实测 d1_3/5/9

    输出：
      surrogate : MultiFidelityResidualGP 实例
    """
    print("[build_multi_fidelity_surrogate] 使用新表数据训练 Residual GP...")
    surrogate = MultiFidelityResidualGP(
        cfg=cfg,
        oracle=oracle,
        X_train=X_new,
        y_real_train=d_new,
        jitter=jitter,
        n_restarts_optimizer=n_restarts_optimizer,
    )
    print("[build_multi_fidelity_surrogate] 训练完成。")
    return surrogate
# ========================= Block D：候选生成（StageB + Active Subspace） =========================
def clip_and_quantize_static(cfg: "ExpConfig", X_raw: np.ndarray) -> np.ndarray:
    """
    按 cfg.static_params 中的 min/max/step 对静态参数：
      - 逐参数裁剪到 [min, max]
      - 按 step 做量化：v = min + round((v-min)/step)*step

    输入：
      X_raw: (N,D) 或 (D,) numpy

    输出：
      X_q: (N,D) numpy
    """
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

    return X_q


def sample_random_static(cfg: "ExpConfig", n: int) -> np.ndarray:
    """
    在静态参数范围内做均匀随机采样，并按 step 量化。

    输入：
      cfg : ExpConfig
      n   : 采样数量

    输出：
      X: (n,D) numpy
    """
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
    """
    给定 StageB 对 d1_3/5/9 的预测值，计算一个加权 L2 损失：

        J(x) = Σ_j w_j * (d_j - center_j)^2

    输入：
      d_pred: (N,3) numpy，对应 cfg.d_names 顺序（d1_3、d1_5、d1_9）

    输出：
      J: (N,) numpy，越小越接近目标
    """
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
    """
    对一批候选 recipe：
      - 调 StageBOracle 预测 d1_3/5/9
      - 计算 StageB 视角下的目标损失 J_sim

    输入：
      cfg        : ExpConfig
      oracle     : StageBOracle
      X          : (N,D) numpy，静态参数
      return_pred: 是否返回 d_pred

    输出：
      J_sim: (N,) numpy
      d_pred: (N,3) numpy 或 None
    """
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
    """
    对若干“有希望”的候选 recipe，用 StageB + Active Subspace 做几步梯度下降：

      目标函数：
        J(x) = Σ_j w_j * (d1_j(x) - center_j)^2

      梯度空间：
        - 在 StageB 的静态归一化空间 x_norm 上做梯度
        - 若提供 as_info（Block B.2 输出），可以把梯度投影到前 subspace_dim 个
          Active Subspace 方向上，避免在不重要方向乱抖

      每一步更新后：
        - 映射回原始尺度 x_raw
        - 裁剪 + step 量化（clip_and_quantize_static）
        - 作为下一步的起点

    输入：
      cfg         : ExpConfig
      oracle      : StageBOracle
      X_init      : (N,D) numpy，初始候选（已在范围内）
      as_info     : Active Subspace 信息 dict（可选）
      n_steps     : 梯度迭代步数
      lr          : 学习率（在归一化空间）
      subspace_dim: Active Subspace 维度（0 表示不用 AS，直接用原始梯度）

    输出：
      X_refined: (N,D) numpy，微调后的候选
    """

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

    device = oracle.device
    model = oracle.model
    mean = oracle.static_mean       # (1,D) on device
    std = oracle.static_std         # (1,D) on device
    phys_template = oracle.phys_template  # (1,2,T)
    tvals_template = oracle.tvals_template  # (T,)
    idx_d1 = oracle.idx_d1

    # d 目标中心 & 权重
    centers = torch.tensor(
        [cfg.d_constraints[name]["center"] for name in cfg.d_names],
        dtype=torch.float32,
        device=device,
    )  # (3,)
    w = torch.tensor(cfg.d_weights, dtype=torch.float32, device=device)
    w = w / (w.sum() + 1e-12)

    # 时间索引（3,5,9）
    tidx_list = [T2IDX[t] for t in ["3", "5", "9"]]

    X_cur = clip_and_quantize_static(cfg, X0)  # 先保证合法
    X_refined = np.zeros_like(X_cur, dtype=np.float64)

    for i in range(N):
        x_raw = X_cur[i : i + 1]  # (1,D)
        x_raw_t = torch.from_numpy(x_raw.astype(np.float32)).to(device)  # (1,D)

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
            loss = (w.view(1, -1) * diff2).sum()            # scalar

            loss.backward()
            g_norm = x_norm.grad.detach().cpu().numpy()[0]  # (D,)

            # 若有 Active Subspace，则把梯度投影到低维子空间
            if V_sub is not None:
                # g_proj = V * (V^T g)
                g_norm = V_sub @ (V_sub.T @ g_norm)

            # 在归一化空间做梯度下降
            x_norm_np = x_norm.detach().cpu().numpy()[0]
            x_norm_new = x_norm_np - lr * g_norm

            # 映射回原始空间
            mean_np = mean.detach().cpu().numpy()[0]
            std_np = std.detach().cpu().numpy()[0]
            x_raw_new = mean_np + x_norm_new * std_np       # (D,)

            # 裁剪 + 量化
            x_raw_new = clip_and_quantize_static(cfg, x_raw_new)[0]

            # 准备下一个 step
            x_raw_t = torch.from_numpy(x_raw_new.astype(np.float32)).to(device).view(1, -1)

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
    """
    综合候选生成流程（Block D 主函数）：

      1. 基础随机采样：
         - 按 static_params 范围随机采样 N0 = oversample_factor * cfg.n_candidates
         - 裁剪 + step 量化

      2. StageB 快速筛选：
         - 调 StageBOracle 预测 d1_3/5/9
         - 计算 StageB 目标损失 J_sim
         - 选取损失最小的前 N_keep = max(cfg.n_candidates, stageB_keep_ratio * N0)

      3. 梯度微调（小号 MFL）：
         - 对前 n_refine_max 个“最好”的点做 n_grad_steps 步梯度下降
           （在归一化空间，必要时用 Active Subspace 限制方向）

      4. 合并：
         - 把微调后的点 + 剩余未微调的点合并成最终候选池 X_candidates

    输出的 dict：
      {
        "X_candidates":      (M,D)  用于后续 Block E 的多保真 EI 打分
        "X_stageB_all":      (N0,D) 初始所有候选
        "J_stageB_all":      (N0,)  对应的 StageB 损失
        "X_stageB_top":      (N_keep,D) 筛选后保留的候选（微调前）
        "J_stageB_top":      (N_keep,)  对应损失
        "d_stageB_top":      (N_keep,3) StageB 眼中的 d1_3/5/9 预测
      }
    """
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
    """
    根据多保真 surrogate 的 μ_real, σ_real 近似构造标量损失 J 的均值和标准差。

    设：
        Y_j ~ N(μ_j, σ_j^2)  对应 d1_3/5/9
        J(Y) = Σ_j w_j * (Y_j - center_j)^2

    我们用一阶近似（delta method）估计 J 的方差：
        grad_J_j = ∂J/∂Y_j ≈ 2 * w_j * (μ_j - center_j)
        Var[J] ≈ Σ_j (grad_J_j)^2 * σ_j^2

    输入：
      mu_real:   (N,3) surrogate 均值
      sigma_real:(N,3) surrogate 标准差

    输出：
      J_mu:   (N,)  标量目标的期望
      J_sigma:(N,)  标量目标的近似标准差（>=0）
    """
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
    """
    对最小化问题的期望改进 EI：

        I(x) = max(J_best - J(x), 0)
        若 J ~ N(J_mu, J_sigma^2)，则

        Z = (J_best - J_mu) / J_sigma
        EI(x) = (J_best - J_mu) * Φ(Z) + J_sigma * φ(Z)

      若 J_sigma ≈ 0，则退化为 EI = max(J_best - J_mu, 0)。

    输入：
      J_mu   : (N,)
      J_sigma: (N,)
      J_best : 标量，目前已有高保真数据上的最优 J 值

    输出：
      EI: (N,)
    """
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
    """
    计算 StageB 分布先验权重：

      思路：
        - 使用旧表的静态输入 X_all_raw 构建“经验分布”
        - 在标准化空间（按旧表均值/方差）中计算每个候选点到
          最近旧表样本的距离 d_min
        - prior(x) = exp( -0.5 * (d_min / L)^2 )
        - 再将 prior 下限截断到 prior_floor，避免乘完之后完全为 0

    输入：
      cfg           : ExpConfig
      stageB_data   : load_stageB_dataset(cfg) 的输出
      X_candidates  : (N,D) 候选
      length_scale  : L，越大表示容忍离旧表分布更远的点
      prior_floor   : 先验权重最小值（避免完全抹掉）

    输出：
      prior: (N,) numpy，∈ [prior_floor, 1]
    """
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
    stageB_data: Optional[dict] = None,
    J_best: Optional[float] = None,
    prior_length_scale: float = 1.0,
    prior_floor: float = 0.1,
) -> dict:
    """
    对候选集 X_candidates 使用：
      - 多保真 surrogate 的 EI（最小化 J）
      - StageB 分布先验
    综合打分：

        score(x) = EI(x) * prior(x)

    输入：
      cfg            : ExpConfig
      surrogate      : MultiFidelityResidualGP
      X_candidates   : (N,D) numpy
      stageB_data    : StageB 数据（若为 None，则不加先验）
      J_best         : 当前已知的最佳 J（可为 None）
                       若为 None，则用 surrogate.y_real_train 上的最小 J 作为 J_best
      prior_length_scale: StageB 先验的距离尺度 L
      prior_floor    : 先验下限（避免 0）

    输出：
      info dict，包含：
        - X_candidates
        - mu_real       (N,3)
        - sigma_real    (N,3)
        - J_mu          (N,)
        - J_sigma       (N,)
        - EI            (N,)
        - prior         (N,)  StageB 先验
        - score         (N,)  最终分数
        - J_best        标量
    """
    X_cand = np.asarray(X_candidates, dtype=np.float64)
    if X_cand.ndim == 1:
        X_cand = X_cand[None, :]

    # 1) surrogate 的均值与不确定度
    mu_real, sigma_real = surrogate.predict_with_uncertainty(X_cand)  # (N,3), (N,3)

    # 2) 将多维输出映射到标量目标 J
    J_mu, J_sigma = scalar_objective_from_mu_sigma(cfg, mu_real, sigma_real)

    # 3) 当前已知最佳 J_best（来自高保真数据）
    if J_best is None:
        # 用 surrogate 训练集的 y_real_train 计算
        J_train = stageB_objective_from_preds(cfg, surrogate.y_real_train)  # (N_train,)
        J_best = float(J_train.min())
        print(f"[score_candidates_with_EI_and_prior] 自动推得 J_best = {J_best:.3e}")

    # 4) 计算 minimization EI
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

    # 6) 综合分数
    score = EI * prior

    info = {
        "X_candidates": X_cand,
        "mu_real": mu_real,
        "sigma_real": sigma_real,
        "J_mu": J_mu,
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
    """
    根据 Block E 的打分结果，选出前 topK 条推荐 recipe。

    输入：
      cfg        : ExpConfig
      score_info : score_candidates_with_EI_and_prior 的返回值
      topK       : 选取数量（默认 cfg.n_select）

    输出：
      {
        "X_selected": (K,D)  推荐静态参数
        "mu_real":    (K,3)  多保真预测均值
        "sigma_real": (K,3)  多保真预测不确定度
        "EI":         (K,)
        "prior":      (K,)
        "score":      (K,)
        "indices":    (K,)   在候选池中的原始索引
        "J_best":     标量   当前已知最佳 J
      }
    """
    X_cand = score_info["X_candidates"]
    score = score_info["score"]

    N = X_cand.shape[0]
    if topK is None:
        topK = cfg.n_select
    topK = min(topK, N)

    order = np.argsort(score)[::-1]  # score 越大越好
    idx_sel = order[:topK]

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
    """
    保存选点结果到 Excel + CSV：
      - sheet1: 推荐清单（多保真预测 + StageB 预测 + EI + 先验 + score）
      - sheet2: 候选池评分（用于分析 EI 分布）
      - sheet3: 新表实测（X_new + d_new）
      - sheet4: StageB_相关性（Pearson / Spearman）
      - sheet5: ActiveSubspace（特征值、能量占比、主方向权重）
      - sheet6: Surrogate_R2（StageB vs 多保真 R²）
      - sheet7: StageB筛选（StageB 眼里好的候选）

    同时导出一个 CSV：
      - 推荐recipe清单.csv：便于直接丢给设备/仿真脚本
    """
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
    """
    一键跑完：
      - Block A: 数据入口
      - Block B.1: StageBOracle 模型
      - Block B.2: StageB 相关性 + Active Subspace
      - Block C: 多保真 Residual GP surrogate
      - Block D: 候选生成（StageB + 梯度微调）
      - Block E: 多保真 EI + StageB 先验打分
      - Block F: 结果导出
    返回一个 dict，把中间关键对象都回传出来方便你 debug / 画图。
    """
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
    X_new, d_new = load_new_table_d1(cfg)

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
        target_time="9",
    )

    print("\n========== Block C：构建多保真 Residual GP ==========")
    surrogate = build_multi_fidelity_surrogate(cfg, oracle, X_new, d_new)

    print("\n========== Block D：生成候选（StageB + 梯度微调） ==========")
    cand_info = generate_candidates_with_stageB(
        cfg=cfg,
        oracle=oracle,
        as_info=as_info,
        oversample_factor=2.0,
        stageB_keep_ratio=0.5,
        n_grad_steps=3,
        grad_lr=0.5,
        subspace_dim=1,
        n_refine_max=512,
    )

    print("\n========== Block E：多保真 EI + StageB 先验打分 ==========")
    score_info = score_candidates_with_EI_and_prior(
        cfg=cfg,
        surrogate=surrogate,
        X_candidates=cand_info["X_candidates"],
        stageB_data=stageB_data,
        J_best=None,                # 自动根据新表最优 J 估计
        prior_length_scale=1.0,
        prior_floor=0.1,
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
    cfg = ExpConfig()
    ensure_dir(cfg.save_dir)
    _ = run_advanced_selection(cfg)

