import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import pairwise_distances
from skopt import gp_minimize
from skopt.space import Real
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional


# 新增：DataLoader，用来把 StageB 数据一次性读出来
from torch.utils.data import DataLoader

# -------------------------- 导入StageB/C相关工具（关键对接） --------------------------
from physio_util import load_new_excel_as_sparse_morph, excel_to_morph_dataset_from_old
from stageC_finetune_joint_on_new_pycharm_new import Cfg as CfgC
from stageB_train_morph_on_old_pycharm import Cfg as Bcfg  # 新增：拿 StageB 的 old_excel/sheet_name


# -------------------------- 1. 配置类（仅保留“基于现有数据选点”的参数） --------------------------
class ExpConfig:
    """
    实验配置：基于 Bosch_new.xlsx 中已有实测/仿真数据，
    用 GP 做一次性虚拟实验选点 + 参数影响分析。
    """
    # 1. 静态输入参数（顺序与 physio_util.static_keys 严格一致）
    static_params = {
        "APC": {"min": 10, "max": 100, "step": 5, "unit": "mT"},      # 对应 apc
        "source_RF": {"min": 500, "max": 3500, "step": 100, "unit": "W"},  # 对应 source_rf
        "LF_RF": {"min": 25, "max": 150, "step": 5, "unit": "W"},     # 对应 lf_rf
        "SF6": {"min": 50, "max": 500, "step": 50, "unit": "sccm"},   # 对应 sf6
        "C4F8": {"min": 50, "max": 500, "step": 50, "unit": "sccm"},  # 对应 c4f8
        "DEP_time": {"min": 0.4, "max": 4.0, "step": 0.4, "unit": "s"},   # 对应 dep_time
        "etch_time": {"min": 0.4, "max": 4.0, "step": 0.4, "unit": "s"}   # 对应 etch_time
    }
    static_names = list(static_params.keys())  # 顺序必须与 static_keys 一致

    # 2. d 目标约束（对应 family="d1"，time="3/5/9"，与 StageC 映射一致）
    d_constraints = {
        "d1_3": {"min": 0, "max": 0.05, "unit": "μm", "center": 0.01},  # (d1,3)
        "d1_5": {"min": 0, "max": 0.05, "unit": "μm", "center": 0.01},  # (d1,5)
        "d1_9": {"min": 0, "max": 0.05, "unit": "μm", "center": 0.01}   # (d1,9)
    }
    d_names = list(d_constraints.keys())
    # 多目标权重：可根据工艺关注点调整
    d_weights = [0.3, 0.3, 0.4]

    # 3. 实验分组（Real 模式核心策略）
    experiments = [
        # I2：用 GP + 随机采样做多输出 BO 风格选点
        {"id": "I2", "dim": "短期改进", "desc": "GP多输出（随机采样）", "seed": 42},
        # I4：用 GP + LHS + 权重调优（更强调 d1_5、d1_9）
        {"id": "I4", "dim": "短期改进", "desc": "GP多输出（LHS+权重调优）", "seed": 42},
        # 下面三个作为中期融合占位（暂不运行）
        {"id": "M1", "dim": "中期融合", "desc": "[占位]单独MFL", "seed": 42},
        {"id": "M2", "dim": "中期融合", "desc": "[占位]单独StageC", "seed": 42},
        {"id": "M3", "dim": "中期融合", "desc": "[占位]融合组", "seed": 42}
    ]

    # 4. 路径配置（与 StageC 共享数据路径）
    save_dir = "./stageC_results"
    excel_name = "stageC_real_selection_results.xlsx"
    real_meas_path = CfgC.new_excel     # TODO: 如需对接 MFL 反向模型
    # 新增：StageB 仿真数据路径（从 StageB 的 Cfg 里拿）
    sim_excel_path = Bcfg.old_excel
    sim_sheet_name = getattr(Bcfg, "sheet_name", None)
    stageC_ckpt = "./ckpt/stageC_best.pth"
    mfl_ckpt = "./ckpt/mfl_reverse.pth"
    # 5. 实验参数
    n_candidates = 2000  # 候选池大小（虚拟实验可按算力调大/调小）
    n_test = 10          # 每个策略最终推荐多少条 recipe
    n_rounds = 1         # 当前场景：一次性推荐即可，设为 1
    fig_dpi = 300
    # 6. StageB 先验相关参数（新加）
    use_stageB_prior = True  # 是否启用 StageB 仿真空间先验
    sim_dist_scale = 300.0  # “离仿真样本多远算远” 的长度尺度（单位是静态参数空间的欧式距离，可之后调）
    sim_prior_weight = 0.6  # 0~1，越大越偏向 StageB 覆盖区域


# -------------------------- 2. 基础工具函数 --------------------------
def ensure_dir(path: str):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)

def load_stageB_static(config: ExpConfig) -> Optional[np.ndarray]:
    """
    从旧表（StageB 使用的 Bosch_old.xlsx）中读取仿真样本的静态参数（物理单位）.
    """
    print("\n尝试加载 StageB 仿真数据（旧表）用于先验约束 ...")

    ds, meta_old = excel_to_morph_dataset_from_old(
        config.sim_excel_path,
        sheet_name=config.sim_sheet_name
    )
    if len(ds) == 0:
        print("  StageB 数据集为空，跳过先验约束。")
        return None

    # ds.tensors[0] 就是 static_norm
    static_norm = ds.tensors[0].detach().cpu().numpy().astype(np.float32)  # (N, 7) 实际上

    # 从 meta 中取出 mean/std（注意是 Tensor，要转成 numpy）
    mean = meta_old["norm_static"]["mean"]
    std = meta_old["norm_static"]["std"]
    if isinstance(mean, torch.Tensor):
        mean = mean.cpu().numpy()
    else:
        mean = np.asarray(mean)
    if isinstance(std, torch.Tensor):
        std = std.cpu().numpy()
    else:
        std = np.asarray(std)

    # 反归一化，回到物理单位
    static_raw = static_norm * std + mean   # (N, 7)
    X_sim = static_raw[:, :len(config.static_names)].astype(np.float32)

    print(f"  加载 StageB 仿真样本数：{X_sim.shape[0]}")
    return X_sim

def sample_candidates(config: ExpConfig, exp: Dict, sample_type: str = "random") -> np.ndarray:
    """
    生成候选 recipe（按设备步长量化，匹配真实调节能力）
    - sample_type="random"：均匀随机
    - sample_type="LHS"   ：Latin Hypercube Sampling
    """
    n = config.n_candidates
    dim = len(config.static_names)
    X = np.zeros((n, dim), dtype=np.float32)

    for i, name in enumerate(config.static_names):
        p = config.static_params[name]
        min_val, max_val = p["min"], p["max"]
        if sample_type == "random":
            X[:, i] = np.random.uniform(min_val, max_val, n)
        elif sample_type == "LHS":
            partitions = np.linspace(min_val, max_val, n + 1)
            for j in range(n):
                X[j, i] = np.random.uniform(partitions[j], partitions[j + 1])
            np.random.shuffle(X[:, i])
        else:
            raise ValueError(f"未知的采样方式：{sample_type}")

    # 按设备步长量化
    for i, name in enumerate(config.static_names):
        step = config.static_params[name]["step"]
        X[:, i] = np.round(X[:, i] / step) * step

    return X


# -------------------------- 3. 代理模型接口（GP + 预留 StageC） --------------------------
def surrogate_predict(config: ExpConfig, X: np.ndarray, exp: Dict,
                      mode: str = "gp", aux: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    统一代理模型接口：
    - mode="gp"     ：用 GP 多输出代理
    - mode="stageC"：预留口，直接走 StageC 模型前向（后续可接）
    """
    if mode == "gp":
        gp_models: List[GaussianProcessRegressor] = aux["gp_models"]
        n = X.shape[0]
        n_targets = len(gp_models)

        mu = np.zeros((n, n_targets), dtype=np.float32)
        sigma = np.zeros((n, n_targets), dtype=np.float32)

        for j, gp in enumerate(gp_models):
            mu_j, sigma_j = gp.predict(X, return_std=True)
            mu[:, j] = mu_j
            sigma[:, j] = sigma_j

        return mu, sigma

    elif mode == "stageC":
        model = aux.get("stageC_model")
        return stageC_forward(model, X)

    else:
        raise ValueError(f"当前仅支持 gp/stageC 代理模型，不支持：{mode}")


# -------------------------- 4. 数据加载 & 模型训练 --------------------------
def load_real_measurements(config: ExpConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 Bosch_new.xlsx 读取实测/仿真数据（与 StageC 的解析逻辑对齐）

    返回：
    - X_meas: (N, 7) 静态参数 [APC, source_RF, LF_RF, SF6, C4F8, DEP_time, etch_time]
    - d_meas: (N, 3) 实测/仿真 d 值 [d1_3, d1_5, d1_9]
    """
    recs = load_new_excel_as_sparse_morph(
        config.real_meas_path,
        height_family="h1"  # 与 StageB/C 保持一致
    )

    X_list, d_list = [], []
    for rec in recs:
        static = rec["static"]  # shape: (7,)
        X_list.append(static)

        tg = rec["targets"]    # dict: (family, tid) -> value
        d_vals = []
        for tid in ["3", "5", "9"]:
            key = ("d1", tid)
            if key in tg and tg[key] is not None:
                d_vals.append(float(tg[key]))
            else:
                d_vals.append(np.nan)
        d_list.append(d_vals)

    X_meas = np.array(X_list, dtype=np.float32)
    d_meas = np.array(d_list, dtype=np.float32)
    mask = np.isfinite(d_meas).all(axis=1)
    X_meas = X_meas[mask]
    d_meas = d_meas[mask]

    print(f"加载 Bosch_new 中有效样本数：{X_meas.shape[0]}")
    if X_meas.shape[0] == 0:
        raise ValueError("未加载到有效实测数据，请检查 CfgC.new_excel 指向的文件。")

    return X_meas, d_meas


def load_stageC_model(config: ExpConfig) -> torch.nn.Module:
    """
    预留：加载 StageC 模型（当前不启用，只做接口占位）
    真实实现需包含：
    - TemporalRegressor 主干
    - per-family head（针对 "d1" 族）
    - 校准层（calib）
    """
    class DummyStageC(nn.Module):
        def forward(self, x):
            w = torch.randn(x.shape[1], 3)
            return x @ w, torch.ones(x.shape[0], 3) * 0.3

    model = DummyStageC()
    # TODO: 如需使用 StageC 做 oracle，请在此处实例化真实 TemporalRegressor 并加载 ckpt
    return model


def stageC_forward(model: torch.nn.Module, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    预留：StageC 模型前向预测
    真实实现示意：
    1. 静态参数归一化（meta_old.norm_static）
    2. 调 StageA 物理模型，生成 phys_enh
    3. 输入 TemporalRegressor，得到全时间 scallop 序列
    4. 提取 (d1,3)/(d1,5)/(d1,9) 对应的预测值
    """
    raise NotImplementedError(
        "stageC_forward 尚未实现。\n"
        "如需用 StageC 做仿真 oracle，请在此函数中对接 StageA+TemporalRegressor 的前向。"
    )


def train_gp_surrogate(config: ExpConfig,
                       X_meas: np.ndarray,
                       d_meas: np.ndarray) -> List[GaussianProcessRegressor]:
    """
    基于 Bosch_new 中的 (X_meas, d_meas) 训练 GP 代理：
    - 为 d1_3 / d1_5 / d1_9 各训练一个独立的 GP
    """
    n_targets = d_meas.shape[1]
    models: List[GaussianProcessRegressor] = []

    print("\n开始训练 GP 代理模型：")
    for j in range(n_targets):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(X_meas.shape[1])) \
                 + WhiteKernel(noise_level=1e-3)
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=3,
            alpha=1e-6,
            random_state=42 + j
        )
        gp.fit(X_meas, d_meas[:, j])
        y_pred = gp.predict(X_meas)
        r2 = r2_score(d_meas[:, j], y_pred)
        print(f"  - GP[{config.d_names[j]}] 训练完成，R²≈ {r2:.3f}，核函数：{gp.kernel_}")
        models.append(gp)

    return models


# -------------------------- 5. 选点核心逻辑（EI，无多样性项） --------------------------
def calculate_score(config: ExpConfig,
                    pred_d: np.ndarray,
                    sigma_d: np.ndarray,
                    X_cand: np.ndarray,
                    X_meas: np.ndarray,
                    y_meas: np.ndarray,
                    X_sim: Optional[np.ndarray] = None) -> np.ndarray:

    """
    基于 GP 预测的：
    - pred_d: 预测 d 值 (N, 3)
    - sigma_d: 预测不确定性 (N, 3)

    计算每个候选 recipe 的综合得分：
    - 目标：尽量靠近目标区间中心（多目标加权）
    - EI：基于当前已观测的最优目标距离
    - 多样性：当前版本不启用（div_weight = 1），避免“往参数边界乱跑”
    """
    n = pred_d.shape[0]
    score = np.zeros(n, dtype=np.float32)

    # 1. 定义目标函数：距离区间中心越小越好
    target_scores = np.zeros_like(pred_d, dtype=np.float32)
    for j, d_name in enumerate(config.d_names):
        min_d = config.d_constraints[d_name]["min"]
        max_d = config.d_constraints[d_name]["max"]
        center_d = config.d_constraints[d_name]["center"]

        target_scores[:, j] = np.abs(pred_d[:, j] - center_d)

        # 预测值已经明显超出目标区间的，先乘一个惩罚因子
        out_of_bounds = (pred_d[:, j] < min_d) | (pred_d[:, j] > max_d)
        target_scores[out_of_bounds, j] *= 5.0

    # 2. 计算基于当前实测数据的最佳目标值（per d）
    target_best = np.array([
        np.abs(y_meas[:, j] - config.d_constraints[config.d_names[j]]["center"]).min()
        for j in range(3)
    ], dtype=np.float32)

    # 3. 对每个 d 计算 EI（越大越好）
    ei_per_d = np.zeros_like(pred_d, dtype=np.float32)
    for j in range(3):
        mu = target_scores[:, j]     # 当前预测样本的“损失”
        sigma = sigma_d[:, j]        # 模型对该样本的“不确定性”
        improvement = target_best[j] - mu  # 相对当前最好还能改善多少（负数表示更差）

        z = improvement / (sigma + 1e-8)
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        ei_per_d[:, j] = np.maximum(ei, 0.0)  # 负改进视作 0

    # 4. 不再使用“离已测点越远越好”的多样性项，避免偏向范围边界
    div_weight = np.ones(n, dtype=np.float32)

    # 5. 综合得分：多维 EI 按权重加权
    ei_weighted = np.dot(ei_per_d, np.array(config.d_weights, dtype=np.float32))
    score = ei_weighted * div_weight

    # 6. StageB 仿真空间先验：离 StageB 样本越近，权重越大
    if config.use_stageB_prior and (X_sim is not None) and (X_sim.shape[0] > 0):
        # 计算每个候选点到 StageB 仿真样本的最小欧氏距离
        dist_mat = pairwise_distances(X_cand, X_sim, metric="euclidean")
        min_dist = dist_mat.min(axis=1)  # (N,)

        # 距离 → [0,1] 的先验权重：越近越接近 1，越远越接近 0
        L = float(config.sim_dist_scale)
        prior = np.exp(- (min_dist / (L + 1e-8)) ** 2)  # 高斯型衰减

        # 和 EI 结合： (1-w) * 1 + w * prior
        w = float(config.sim_prior_weight)
        prior_weight = (1.0 - w) + w * prior

        score = score * prior_weight

    # 7. 硬过滤：预测值严重超出物理合理区间的候选直接丢弃
    hard_out = np.zeros(n, dtype=bool)
    for j, d_name in enumerate(config.d_names):
        min_d = config.d_constraints[d_name]["min"]
        max_d = config.d_constraints[d_name]["max"]
        margin = 0.2 * (max_d - min_d)  # 允许 20% 的 buffer

        too_low = pred_d[:, j] < (min_d - margin)
        too_high = pred_d[:, j] > (max_d + margin)
        hard_out |= (too_low | too_high)

    score[hard_out] = 0.0
    return score


def judge_d_qualified(config: ExpConfig,
                      measured_d: np.ndarray) -> Tuple[List[bool], List[List[bool]]]:
    """判断 d 是否在目标区间内（可用于真实实测 or 仿真结果的达标统计）"""
    n = measured_d.shape[0]
    full_qualified = []
    single_qualified = []

    for i in range(n):
        single = []
        for j, name in enumerate(config.d_names):
            min_d = config.d_constraints[name]["min"]
            max_d = config.d_constraints[name]["max"]
            single.append(min_d <= measured_d[i, j] <= max_d)
        full_qualified.append(all(single))
        single_qualified.append(single)

    return full_qualified, single_qualified


# -------------------------- 6. 一次性主动选点（n_rounds 默认 1） --------------------------
def active_selection_loop(config: ExpConfig,
                          strategy_id: str,
                          exp: Dict) -> List[Dict]:
    """
    一次性主动选点循环（当前 n_rounds=1）：

    流程：
    1. 从 Bosch_new.xlsx 载入现有实测/仿真数据
    2. 训练 GP 代理（多输出）
    3. 在参数空间采样候选 recipe（Random / LHS）
    4. 利用 GP 预测 (d1_3, d1_5, d1_9) 及不确定性
    5. 根据 EI 打分，选出 top-K 个推荐 recipe
    6. 输出推荐清单（用于补充 StageC 实测或虚拟实验）
    """
    # 固定随机种子
    np.random.seed(exp["seed"] + 1000)

    # 1. 加载新表已有实测/仿真数据
    X_meas, d_meas = load_real_measurements(config)

    # 2. 可选：加载 StageB 仿真样本的静态参数，用作先验约束
    X_sim = None
    if config.use_stageB_prior:
        try:
            X_sim = load_stageB_static(config)
        except Exception as e:
            print(f"加载 StageB 仿真数据失败，将仅使用新表实测进行选点。错误信息：{e}")
            X_sim = None

    all_round_results = []

    for round_idx in range(config.n_rounds):
        print(f"\n===== 主动选点第 {round_idx + 1}/{config.n_rounds} 轮（策略 {strategy_id}） =====")

        # 2. 选择代理模型（当前：仅 GP）
        if strategy_id in ["I2", "I4"]:
            gp_models = train_gp_surrogate(config, X_meas, d_meas)
            surrogate_mode = "gp"
            aux = {"gp_models": gp_models}
        else:
            raise NotImplementedError(f"当前仅实现 I2/I4 两个策略，{strategy_id} 暂未支持。")

        # 3. 生成候选池：I4 用 LHS，I2 用随机
        sample_type = "LHS" if strategy_id == "I4" else "random"
        X_cand = sample_candidates(config, exp, sample_type=sample_type)

        # 4. 预测 + 打分
        pred_d, sigma_d = surrogate_predict(config, X_cand, exp,
                                            mode=surrogate_mode, aux=aux)
        score = calculate_score(
            config,
            pred_d, sigma_d,
            X_cand,
            X_meas, y_meas=d_meas,
            X_sim=X_sim
        )

        # 5. 选出 top-K 推荐 recipe
        top_idx = np.argsort(-score)[:config.n_test]
        X_next = X_cand[top_idx]
        pred_d_next = pred_d[top_idx]

        # 6. 统计“预测上的达标率”与偏离中心程度
        pred_full_q, pred_single_q = judge_d_qualified(config, pred_d_next)
        pred_full_rate = sum(pred_full_q) / len(pred_full_q) * 100.0

        avg_center_dev = []
        for j, d_name in enumerate(config.d_names):
            center = config.d_constraints[d_name]["center"]
            dev = np.abs(pred_d_next[:, j] - center)
            avg_center_dev.append(float(dev.mean()))

        print(f"  本轮推荐 {len(X_next)} 个 recipe")
        print(f"  预测全 d 达标率 ≈ {pred_full_rate:.1f}%")
        print(
            f"  预测平均偏离中心 (μm)："
            f"d1_3≈{avg_center_dev[0]:.1f}, d1_5≈{avg_center_dev[1]:.1f}, d1_9≈{avg_center_dev[2]:.1f}"
        )

        # 7. 记录结果（当前不自动追加实测，视为一次性推荐）
        round_result = {
            "round": round_idx + 1,
            "strategy_id": strategy_id,
            "X_recommended": X_next,
            "pred_d": pred_d_next,
            "pred_full_qualified": pred_full_q,
            "pred_single_qualified": pred_single_q,
            "pred_full_rate": pred_full_rate,
            "avg_center_dev": avg_center_dev,
            "X_measured": X_meas.copy(),
            "d_measured": d_meas.copy()
        }
        all_round_results.append(round_result)

        # 当前场景：没有新实测补充，不更新 X_meas / d_meas，下一轮只是重复分析

    return all_round_results


# -------------------------- 7. 实验执行与结果保存 --------------------------
def run_single_experiment(config: ExpConfig, exp: Dict) -> Dict:
    """执行单个策略（如 I2 / I4）的选点实验"""
    print(f"\n{'=' * 60}")
    print(f"选点实验组：{exp['id']} | {exp['dim']} | {exp['desc']}")
    print(f"{'=' * 60}")

    np.random.seed(exp["seed"])
    orig_weights = config.d_weights.copy()

    try:
        # I4：多 d 权重调优（更强调 d1_5 和 d1_9）
        if exp["id"] == "I4":
            config.d_weights = [0.2, 0.4, 0.4]
            print(f"  I4 组 d 权重调整为：{config.d_weights}")

        round_results = active_selection_loop(config, exp["id"], exp)

        return {
            "exp_info": exp,
            "round_results": round_results,
            "final_weights": config.d_weights
        }
    finally:
        config.d_weights = orig_weights


def save_results(config: ExpConfig, all_results: List[Dict]):
    """保存选点结果：策略汇总 + 推荐详情 + 参数影响分析"""
    ensure_dir(config.save_dir)

    excel_path = os.path.join(config.save_dir, config.excel_name)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        # 1) 策略汇总 sheet
        summary_rows = []
        for res in all_results:
            exp = res["exp_info"]
            for rr in res["round_results"]:
                summary_rows.append({
                    "实验组ID": exp["id"],
                    "策略描述": exp["desc"],
                    "轮次": rr["round"],
                    "推荐数量": len(rr["X_recommended"]),
                    "预测全d达标率(%)": rr["pred_full_rate"],
                    "d1_3平均偏离(μm)": rr["avg_center_dev"][0],
                    "d1_5平均偏离(μm)": rr["avg_center_dev"][1],
                    "d1_9平均偏离(μm)": rr["avg_center_dev"][2],
                    "权重配置": str(res["final_weights"])
                })
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="策略汇总", index=False)

        # 2) 推荐详情 sheet
        detail_rows = []
        for res in all_results:
            exp = res["exp_info"]
            for rr in res["round_results"]:
                for i, x in enumerate(rr["X_recommended"]):
                    row = {
                        "实验组ID": exp["id"],
                        "轮次": rr["round"],
                        "推荐序号": i + 1,
                        "预测全d达标": "是" if rr["pred_full_qualified"][i] else "否"
                    }
                    for j, name in enumerate(config.static_names):
                        row[name] = float(x[j])
                    for j, d_name in enumerate(config.d_names):
                        row[f"预测{d_name}(μm)"] = float(rr["pred_d"][i, j])
                        row[f"{d_name}达标"] = "是" if rr["pred_single_qualified"][i][j] else "否"
                    detail_rows.append(row)
        pd.DataFrame(detail_rows).to_excel(writer, sheet_name="推荐详情", index=False)

        # 3) 参数-目标相关系数 sheet（基于已有实测/仿真数据）
        if all_results and all_results[0]["round_results"]:
            ref_rr = all_results[0]["round_results"][0]
            X_meas = ref_rr["X_measured"]
            d_meas = ref_rr["d_measured"]

            df_x = pd.DataFrame(X_meas, columns=config.static_names)
            df_d = pd.DataFrame(d_meas, columns=config.d_names)
            corr = df_x.join(df_d).corr().loc[config.static_names, config.d_names]

            corr.to_excel(writer, sheet_name="参数-目标相关系数")

    # 4) 可直接导入设备/仿真脚本的推荐清单 CSV
    csv_rows = []
    for res in all_results:
        exp = res["exp_info"]
        for rr in res["round_results"]:
            for i, x in enumerate(rr["X_recommended"]):
                row = {
                    "实验组": exp["id"],
                    "轮次": rr["round"],
                    "序号": i + 1,
                    **{name: float(x[j]) for j, name in enumerate(config.static_names)},
                    **{f"预测{d_name}": float(rr["pred_d"][i, j]) for j, d_name in enumerate(config.d_names)},
                    "预测全达标": bool(rr["pred_full_qualified"][i])
                }
                csv_rows.append(row)
    csv_path = os.path.join(config.save_dir, "推荐recipe清单.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\n结果已保存至：{config.save_dir}")
    print(f"  - 详细结果：{os.path.basename(excel_path)}")
    print(f"  - 推荐清单：{os.path.basename(csv_path)}")

    # 控制台打印一次参数-目标相关系数（便于快速检查）
    if all_results and all_results[0]["round_results"]:
        ref_rr = all_results[0]["round_results"][0]
        X_meas = ref_rr["X_measured"]
        d_meas = ref_rr["d_measured"]
        df_x = pd.DataFrame(X_meas, columns=config.static_names)
        df_d = pd.DataFrame(d_meas, columns=config.d_names)
        corr = df_x.join(df_d).corr().loc[config.static_names, config.d_names]
        print("\n参数-目标相关系数（基于当前 Bosch_new 数据）：")
        print(corr.round(3))


# -------------------------- 8. 主函数 --------------------------
def main():
    """
    主函数：基于 Bosch_new.xlsx 现有数据，运行 I2 / I4 两个策略，
    进行一次性 recipe 推荐 + 参数影响分析。
    """
    config = ExpConfig()
    ensure_dir(config.save_dir)

    real_strategies = ["I2", "I4"]
    exp_list = [e for e in config.experiments if e["id"] in real_strategies]

    all_results = []
    for exp in exp_lis
        res = run_single_experiment(config, exp)
        all_results.append(res)

    save_results(config, all_results)
    print("\n选点实验全部完成。")
    print("提示：生成的『推荐recipe清单.csv』可用于：")
    print("  - 作为一次性补充 StageC 实测的候选 recipe；")
    print("  - 或在虚拟仿真环境中批量跑 profile，用于进一步分析。")


if __name__ == "__main__":
    main()
