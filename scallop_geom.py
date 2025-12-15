# -*- coding: utf-8 -*-
"""
scallop_geom.py

基于 physio_util.load_new_excel_as_sparse_morph 读入的新表 targets，
使用简化的“圆弧凸透镜”几何模型，构造 9 个 scallop 的几何参数，并在 2D 网格上生成
level set φ(x,z)，可以配合可视化检查扇贝纹截面。

依赖：
    - numpy
    - matplotlib（仅用于 plot_scallops，可选）
    - physio_util.load_new_excel_as_sparse_morph（可选）

本版参数使用原则：

  - 宽度 w：用 w1, w3, w5, w9 四个点在周期索引 {1,3,5,9} 上做锚点，
    对 i=1..9 做线性插值，得到每一段自己的 w_i。

  - 高度 h（弦长）：只相信中间的 h3, h5，认为第 9 段测不准；
    先用 (3,h3)、(5,h5) 在周期索引上插值出 1..8 段的 h_i，
    然后用 zmin 兜底第 9 段：
        h9* = |zmin| - sum_{i=1..8} h_i
    若 h9* <= 0，则回退为插值值或实测 h9。

  - 矢高 d：用 d3, d5, d9 三个点在周期索引 {3,5,9} 上插值出 1..9 段的 d_i，
    第 9 段用实测 d9，不做额外放大。

  - 半径 R_i：采用圆弓形几何关系
        R_i = h_i^2 / (8 d_i) + d_i / 2
"""

from typing import Dict, Tuple, Any, Optional

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    plt = None
    _HAS_PLT = False

# ---------------------------------------------------------------------------
# 1. 从 targets（与 physio_util 一致的字典）构造 9 个周期的几何参数
# ---------------------------------------------------------------------------


def build_9_cycle_params_from_targets(
    tg: Dict[Tuple[str, str], float],
    height_family: str = "h1",
    allow_negative_h9: bool = False,
) -> Dict[str, np.ndarray]:
    """
    根据单个样本的 targets 字典（与 physio_util.load_new_excel_as_sparse_morph 一致）
    构造 9 个 scallop 周期的几何参数。

    参数
    ----
    tg : dict
        键为 (family, time_id) 的字典，例如：
            ("zmin", "9_2")
            ("w", "1"),("w","3"),("w","5"),("w","9")
            (height_family, "3/5/9")
            ("d1", "3/5/9")
        值为浮点数，单位均为 μm（physio_util 中已进行 nm→μm 转换）。
    height_family : {"h0","h1"}, 默认 "h1"
        使用哪一族高度作为 scallop 的弦长 h。
    allow_negative_h9 : bool, 默认 False
        是否允许 h9* 为非正。一般不允许，若 h9*<=0 则回退为插值值/实测 h9。

    返回
    ----
    params : dict
        包含以下字段（均为形状 (9,) 的 numpy 数组，单位 μm）：
            - "h"      : 每个周期的弦长
            - "d"      : 每个周期的矢高
            - "w"      : 每个周期的宽度
            - "R"      : 每个周期的圆半径
            - "z_top"  : 每段弦上端 z（z 向下为正，顶面为 0）
            - "z_bot"  : 每段弦下端 z
            - "z_c"    : 每段弦中点 z
            - "x_c_L"  : 左侧圆心 x 坐标
            - "x_c_R"  : 右侧圆心 x 坐标
        以及：
            - "total_depth" : |zmin|，总深度（标量）
    """

    def _get(key: Tuple[str, str], name: str) -> float:
        if key not in tg:
            raise KeyError(f"targets 中缺少 {name} 参数: key={key}")
        return float(tg[key])

    # ---- 基本量：zmin & 测量的 w/h/d ----
    # zmin(9_2) 在 physio_util 中已经是负值：tg[("zmin","9_2")] = -abs(...)
    zmin = _get(("zmin", "9_2"), "总深度 zmin(9_2)")
    total_depth = abs(zmin)

    # 宽度锚点
    w1 = _get(("w", "1"), "w1")
    w3 = _get(("w", "3"), "w3")
    w5 = _get(("w", "5"), "w5")
    w9 = _get(("w", "9"), "w9")

    # 高度（弦长）锚点：认为 h3/h5 可靠，h9 仅作为回退
    h3 = _get((height_family, "3"), f"{height_family}_3 (第三个 scallop 高度)")
    h5 = _get((height_family, "5"), f"{height_family}_5 (第五个 scallop 高度)")
    h9_meas: Optional[float] = tg.get((height_family, "9"), None)
    if h9_meas is not None:
        h9_meas = float(h9_meas)

    # 矢高锚点：d3, d5, d9 都用上
    d3 = _get(("d1", "3"), "d1_3 (第三个 scallop 深度)")
    d5 = _get(("d1", "5"), "d1_5 (第五个 scallop 深度)")
    d9 = _get(("d1", "9"), "d1_9 (最后一个 scallop 深度)")

    # ---- 周期索引：1..9 ----
    idx = np.arange(1.0, 10.0, dtype=float)  # [1,2,...,9]

    # ---- Step 1: 每段宽度 w[i]：用 w1,w3,w5,w9 做分段线性插值 ----
    xp_w = np.array([1.0, 3.0, 5.0, 9.0], dtype=float)
    fp_w = np.array([w1, w3, w5, w9], dtype=float)
    # np.interp 会在 [1,9] 内做线性插值；idx 正好覆盖 1..9
    w = np.interp(idx, xp_w, fp_w)

    # ---- Step 2: 1..8 段的弦长 h[i]
    # 只用 (3,h3)、(5,h5) 这两个点插出 1..8 段，认为顶部类似 3 段，5 以下类似 5 段
    xp_h = np.array([3.0, 5.0], dtype=float)
    fp_h = np.array([h3, h5], dtype=float)
    # left=h3, right=h5：1,2 用 h3，6,7,8 用 h5
    h_raw = np.interp(idx, xp_h, fp_h, left=h3, right=h5)

    h = h_raw.copy()

    # 先不管 h9，先固定 1..8 段
    H_1to8 = float(h[:8].sum())

    # ---- Step 3: 用 zmin 兜底 h9 ----
    h9_star = total_depth - H_1to8

    if (not allow_negative_h9) and (h9_star <= 0.0):
        # 若 zmin 不够大导致 h9*<=0，则回退：
        #   优先用实测 h9，其次用插值的 h_raw[8]
        if h9_meas is not None and h9_meas > 0.0:
            h9 = h9_meas
        else:
            h9 = float(h_raw[8])
    else:
        h9 = h9_star

    h[8] = h9  # 第 9 段弦长

    # ---- Step 4: d[i]：用 (3,d3),(5,d5),(9,d9) 插出全部 1..9 段 ----
    xp_d = np.array([3.0, 5.0, 9.0], dtype=float)
    fp_d = np.array([d3, d5, d9], dtype=float)
    # 对 idx=1..9 做插值：1,2 用 d3；6,7,8 在 d5-d9 之间插；9 用 d9。
    d = np.interp(idx, xp_d, fp_d, left=d3, right=d9)

    # ---- Step 5: 计算每一段的 z 位置：顶面 z=0，z 向下为正 ----
    z_top = np.zeros(9, dtype=float)
    z_bot = np.zeros(9, dtype=float)
    z_c = np.zeros(9, dtype=float)

    acc = 0.0
    for i in range(9):
        z_top[i] = acc
        acc += h[i]
        z_bot[i] = acc
        z_c[i] = 0.5 * (z_top[i] + z_bot[i])

    # ---- Step 6: 圆弧几何：R = h^2/(8d) + d/2；圆心位置由 w 决定 ----
    R = np.zeros(9, dtype=float)
    x_c_L = np.zeros(9, dtype=float)
    x_c_R = np.zeros(9, dtype=float)

    for i in range(9):
        hi = float(h[i])
        di = float(d[i])
        if di <= 0:
            raise ValueError(f"第 {i + 1} 段 d<=0，不合法: d={di}")
        # 圆弓形几何关系：弦长 hi、矢高 di → 半径
        Ri = hi * hi / (8.0 * di) + di / 2.0
        R[i] = Ri

        wi = float(w[i])

        # 注意：这里仍使用“w 是两侧凸透镜某种对称距离”的设定，
        # 具体是：圆心在硅里，R 向左右伸出，真正的壁在圆的一部分上；
        # x_c_L/x_c_R 只是圆的位置，具体哪一半作为侧壁由 level set 决定。
        x_c_R[i] = wi / 2.0 + Ri
        x_c_L[i] = -wi / 2.0 - Ri

    params = {
        "h": h,
        "d": d,
        "w": w,
        "R": R,
        "z_top": z_top,
        "z_bot": z_bot,
        "z_c": z_c,
        "x_c_L": x_c_L,
        "x_c_R": x_c_R,
        "total_depth": total_depth,
    }
    return params


# ---------------------------------------------------------------------------
# 2. 在网格上构造 level set φ(x,z)
# ---------------------------------------------------------------------------


def build_levelset_from_params(
    params: Dict[str, np.ndarray],
    x: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """
    在给定的 (x,z) 网格上，基于几何参数构造 level set φ(x,z)。

    当前版本仍采用「左半平面用左圆、右半平面用右圆」的全圆方式：
      - 左半平面 (x<=0) 只使用左侧圆弧；
      - 右半平面 (x>=0) 只使用右侧圆弧；
    φ=0 是所有圆弧的外包络。后续如果要更精细地区分“左半圆/右半圆 + 底部碗形”，
    可以在此基础上继续改。
    """
    X, Z = np.meshgrid(x, z, indexing="xy")
    phi = np.full_like(X, np.inf, dtype=float)

    R = params["R"]
    z_c = params["z_c"]
    x_c_L = params["x_c_L"]
    x_c_R = params["x_c_R"]

    # 左右半平面的掩码
    mask_L = X <= 0
    mask_R = X >= 0

    for i in range(9):
        Ri = R[i]
        zci = z_c[i]

        # 左侧圆
        xci_L = x_c_L[i]
        phi_L = np.sqrt((X - xci_L) ** 2 + (Z - zci) ** 2) - Ri

        # 右侧圆
        xci_R = x_c_R[i]
        phi_R = np.sqrt((X - xci_R) ** 2 + (Z - zci) ** 2) - Ri

        # 左半边只看左圆
        phi[mask_L] = np.minimum(phi[mask_L], phi_L[mask_L])
        # 右半边只看右圆
        phi[mask_R] = np.minimum(phi[mask_R], phi_R[mask_R])

    return phi


# ---------------------------------------------------------------------------
# 3. 可视化：画出 φ=0 的截面轮廓（可选）
# ---------------------------------------------------------------------------


def plot_scallops(
    params,
    x_range=None,
    z_range=None,
    nx: int = 400,
    nz: int = 400,
    show: bool = True,
):
    """
    使用等高线 φ(x,z)=0 画出 9 个 scallops 的左右侧壁轮廓。

    如果 x_range 或 z_range 为 None，则自动根据圆心和半径估一个合理范围。
    """
    if not _HAS_PLT:
        raise RuntimeError("matplotlib 未安装，无法使用 plot_scallops")

    # --- 自动决定绘图范围 ---
    if x_range is None:
        # 左侧最外面在 x_c_L - R，右侧最外面在 x_c_R + R
        x_min = float((params["x_c_L"] - params["R"]).min())
        x_max = float((params["x_c_R"] + params["R"]).max())
        pad = 0.05 * (x_max - x_min)
        x_range = (x_min - pad, x_max + pad)

    if z_range is None:
        z_min = 0.0
        z_max = float(params["total_depth"]) * 1.05
        z_range = (z_min, z_max)

    x = np.linspace(x_range[0], x_range[1], nx)
    z = np.linspace(z_range[0], z_range[1], nz)
    phi = build_levelset_from_params(params, x, z)

    plt.figure(figsize=(4, 8))
    plt.contour(x, z, phi, levels=[0.0])
    plt.gca().invert_yaxis()
    plt.xlabel("x (μm)")
    plt.ylabel("z (μm)")
    plt.title("Scallop cross-section (φ = 0)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    if show:
        plt.show()


# ---------------------------------------------------------------------------
# 4. 与 physio_util 打通的辅助函数（可选）
# ---------------------------------------------------------------------------


def build_params_from_excel(
    excel_path: str,
    index: int = 0,
    height_family: str = "h1",
    allow_negative_h9: bool = False,
) -> Dict[str, np.ndarray]:
    """
    从新 Excel 表中读取数据（使用 physio_util.load_new_excel_as_sparse_morph），
    直接构造第 index 个样本的 9-cycle scallop 几何参数。
    """
    try:
        from physio_util import load_new_excel_as_sparse_morph
    except ImportError as e:
        raise ImportError(
            "无法导入 physio_util.load_new_excel_as_sparse_morph，"
            "请确认 physio_util.py 在 Python 路径中。"
        ) from e

    recs = load_new_excel_as_sparse_morph(excel_path, height_family=height_family)
    if not (0 <= index < len(recs)):
        raise IndexError(f"index={index} 超出范围，当前样本数为 {len(recs)}")

    tg = recs[index]["targets"]
    params = build_9_cycle_params_from_targets(
        tg,
        height_family=height_family,
        allow_negative_h9=allow_negative_h9,
    )
    return params


# ---------------------------------------------------------------------------
# 5. 简单自测入口（可选）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    print("scallop_geom.py 自测示例：")

    example_excel = r"D:\data\pycharm\bosch\Bosch.xlsx"
    if example_excel is not None and os.path.exists(example_excel):
        params = build_params_from_excel(example_excel, index=0, height_family="h1")
        print("9 段 h:", params["h"])
        print("9 段 d:", params["d"])
        print("9 段 w:", params["w"])
        print("总深度 |zmin|:", params["total_depth"])
        if _HAS_PLT:
            plot_scallops(params, x_range=None, z_range=None)
    else:
        print("未设置 example_excel 或文件不存在，只测试几何函数不画图。")
