import numpy as np
import pandas as pd
import torch
from physio_util import (
    excel_to_morph_dataset_from_old,
    load_new_excel_as_sparse_morph,
    FAMILIES, TIME_LIST, F2IDX, T2IDX
)

OLD_EXCEL = r"D:\data\pycharm\bosch\case.xlsx"
NEW_EXCEL = r"D:\data\pycharm\bosch\Bosch.xlsx"
SHEET = "case"  # 你 StageB/C 用的

def get_old_static_and_targets_phys():
    ds, meta = excel_to_morph_dataset_from_old(OLD_EXCEL, sheet_name=SHEET)
    static_norm, phys_seq, targets_norm, mask, time_mat = ds.tensors

    # static 反标准化
    mean = meta["norm_static"]["mean"].to(static_norm)
    std  = meta["norm_static"]["std"].to(static_norm)
    static_raw = (static_norm * std + mean).cpu().numpy()  # (N,7)

    # targets 反 family-wise 标准化
    fam_mean = meta["norm_target"]["mean"].view(1, -1, 1).to(targets_norm)
    fam_std  = meta["norm_target"]["std"].view(1, -1, 1).to(targets_norm)
    targets_phys = (targets_norm * fam_std + fam_mean).cpu().numpy()  # (N,K,T)
    mask_np = mask.cpu().numpy().astype(bool)
    return static_raw, targets_phys, mask_np

def get_new_static_and_sparse_targets():
    recs = load_new_excel_as_sparse_morph(NEW_EXCEL, height_family="h1")
    X_new = np.stack([r["static"] for r in recs], axis=0).astype(np.float32)  # (M,7)
    # sparse dict -> dense (M,K,T)
    M = len(recs); K=len(FAMILIES); T=len(TIME_LIST)
    y = np.zeros((M,K,T), np.float32)
    m = np.zeros((M,K,T), bool)
    for i,r in enumerate(recs):
        for (fam,tid), val in r["targets"].items():
            if fam in F2IDX and tid in T2IDX:
                y[i, F2IDX[fam], T2IDX[tid]] = float(val)
                m[i, F2IDX[fam], T2IDX[tid]] = True
    return X_new, y, m

def match_rows(A, B, tol=1e-6):
    # 返回：B 中每一行在 A 中匹配到的索引（-1 表示没匹配）
    idx = -np.ones((B.shape[0],), dtype=int)
    # 用“rounded tuple”做 hash（离散步进数据很好用）
    A_key = {tuple(np.round(a, 6)): i for i,a in enumerate(A)}
    for j,b in enumerate(B):
        k = tuple(np.round(b, 6))
        if k in A_key:
            idx[j] = A_key[k]
    return idx

def main():
    old_X, old_y, old_m = get_old_static_and_targets_phys()
    new_X, new_y, new_m = get_new_static_and_sparse_targets()

    map_new_to_old = match_rows(old_X, new_X)
    overlap_new = np.where(map_new_to_old >= 0)[0]
    print("Overlap count:", len(overlap_new))

    rows=[]
    for j in overlap_new:
        i = map_new_to_old[j]
        # 只在“新表有实测”的点上比对
        for k,fam in enumerate(FAMILIES):
            for t,tid in enumerate(TIME_LIST):
                if not new_m[j,k,t]:
                    continue
                if not old_m[i,k,t]:
                    continue
                meas = float(new_y[j,k,t])
                sim  = float(old_y[i,k,t])
                denom = max(abs(meas), 1e-8)
                rows.append({
                    "new_idx": j, "old_idx": i,
                    "family": fam, "time": tid,
                    "meas": meas, "sim": sim,
                    "abs_err": abs(sim-meas),
                    "rel_err_%": abs(sim-meas)/denom*100.0
                })

    df = pd.DataFrame(rows)
    df.to_excel("meas_vs_sim_detail.xlsx", index=False)

    # 按参数汇总误差率
    agg = df.groupby(["family","time"]).agg(
        n=("abs_err","count"),
        mae=("abs_err","mean"),
        mape=("rel_err_%","mean"),
        p90=("rel_err_%", lambda x: np.percentile(x, 90)),
    ).reset_index().sort_values(["family","time"])
    agg.to_excel("meas_vs_sim_summary.xlsx", index=False)
    print(agg)

if __name__ == "__main__":
    main()
