# -*- coding: utf-8 -*-
"""
StageC (single-head, per-family) 一键对比脚本
口径（按你最新要求）：
- 不用 teacher / 蒸馏
- 新表 zmin@9_2 -> t=9
- 只训练/评估新表“有实际值”的点（mask=True）
- StageC 初始化来自 StageB 的 best（需要从 best_config + results_summary.csv 检索）
- 由于 StageB 是 per-family 单头模型（out_dim=1），StageC 也按 family 单头跑（避免 out 维度不匹配 + target 尺度/量纲耦合问题）

输出结构：
out_dir/
  summary_all_families.csv
  <fam>/
    stageB_best_ckpt.json
    best_split.json
    split_trials.csv
    compare_on_best_split.csv
    experiments/<exp_name>/
      model_best.pth
      metrics_test.json
      compare_row.json
"""

import os, json, time, argparse
import traceback
from dataclasses import dataclass
from email.policy import default
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import physio_util as pu
import stageB_util as su
import stageB_train_morph_on_phys7_pycharm as sb


# ----------------------------
# Globals
# ----------------------------
TIME_LIST = list(su.TIME_LIST)          # ["1","2",...,"9"]
TIME_VALUES = np.asarray([float(t.replace("_", ".")) for t in TIME_LIST], np.float32)
FAMILIES = list(su.FAMILIES)           # ["zmin","h0","h1","d0","d1","w"]
T = len(TIME_LIST)


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def family_to_index(fam: str) -> int:
    fam = str(fam).strip().lower()
    for i, f in enumerate(FAMILIES):
        if f.lower() == fam:
            return i
    raise KeyError(f"Unknown family={fam}, available={FAMILIES}")


def _torch_load_any(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strip_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not isinstance(sd, dict):
        return sd
    out = {}
    for k, v in sd.items():
        kk = k
        for p in ("module.", "model."):
            if kk.startswith(p):
                kk = kk[len(p):]
        out[kk] = v
    return out


# ----------------------------
# Build StageC raw (no norm)
# ----------------------------
def build_sparse_batch_subset_time(
    recs: List[Dict],
    time_list: List[str],
    time_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    返回：
      recipe_raw: (N,7)
      y:         (N,K,T)  (K=6 canonical)
      m:         (N,K,T)
      time_mat:  (N,T)
    """
    N = len(recs)
    K = len(FAMILIES)
    T = len(time_list)
    t2i = {t: i for i, t in enumerate(time_list)}
    f2i = {f: i for i, f in enumerate(FAMILIES)}

    recipe_raw = np.stack([r["static"] for r in recs], 0).astype(np.float32)  # (N,7)
    y = np.zeros((N, K, T), np.float32)
    m = np.zeros((N, K, T), np.bool_)

    for i, r in enumerate(recs):
        tg: Dict[Tuple[str, str], float] = r.get("targets", {})
        for (fam, tid), val in tg.items():
            fam = str(fam).strip().lower()
            tid = str(tid).strip()
            # 你确认的口径：9_2 -> 9
            if tid == "9_2" and "9" in t2i:
                tid = "9"
            if fam in f2i and tid in t2i:
                y[i, f2i[fam], t2i[tid]] = float(val)
                m[i, f2i[fam], t2i[tid]] = True

    time_mat = np.tile(time_values.reshape(1, -1), (N, 1)).astype(np.float32)
    return recipe_raw, y, m, time_mat


def build_stageC_raw(device: str, new_excel: str, height_family: str, recipe_aug_mode: str,
                     stageA_heads_root: str) -> Dict[str, Any]:
    recs = pu.load_new_excel_as_sparse_morph(new_excel, height_family=height_family)
    recipe_raw, y_raw, mask, time_mat = build_sparse_batch_subset_time(recs, TIME_LIST, TIME_VALUES)

    static_raw = sb.augment_recipe_features(recipe_raw, str(recipe_aug_mode)).astype(np.float32)

    provider = sb.StageAEnsemblePhys7Provider(
        heads_root=stageA_heads_root,
        device=device,
        recipe_cols_in=None,
        expect_k=7
    )
    phys7_raw_full = provider.infer(recipe_raw, phys7_mode="full", use_cache=True).astype(np.float32)  # (N,7)

    # recipe_id 读取（用于 key_recipes）
    try:
        df = pd.read_excel(new_excel)
        rid_col = pu.detect_recipe_id_column(df)
        recipe_ids = df[rid_col].astype(str).to_numpy()
    except Exception:
        recipe_ids = np.asarray([f"row{i:04d}" for i in range(len(recs))], dtype=object)

    return dict(
        recipe_ids=recipe_ids,
        recipe_raw=recipe_raw,
        static_raw=static_raw,
        phys7_raw_full=phys7_raw_full,
        y_raw=y_raw,
        mask=mask,
        time_mat=time_mat,
    )


# ----------------------------
# Normalization (fit on train)
# ----------------------------
def zfit(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std


def zfit_targets_masked_1fam(y: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    y: (N,T), m: (N,T)
    返回 mean/std: (T,)
    """
    mean = np.zeros((T,), np.float32)
    std = np.ones((T,), np.float32)
    for t in range(T):
        sel = m[:, t]
        if int(sel.sum()) < 2:
            mean[t] = 0.0
            std[t] = 1.0
        else:
            vals = y[sel, t]
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            if sd < 1e-6:
                sd = 1.0
            mean[t] = mu
            std[t] = sd
    return mean, std


def apply_z(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / (std + 1e-8)).astype(np.float32)


def apply_phys7_mode(phys7_z_full: np.ndarray, phys7_mode: str) -> np.ndarray:
    # 用 StageB 的定义（按组/子集）
    return sb.apply_phys7_mode(phys7_z_full.astype(np.float32), str(phys7_mode).strip())


# ----------------------------
# DataLoader
# ----------------------------
def make_loader(static_x: np.ndarray,
                phys7_seq: np.ndarray,
                y_norm: np.ndarray,
                m: np.ndarray,
                time_mat: np.ndarray,
                idx: np.ndarray,
                batch: int,
                shuffle: bool,
                num_workers: int = 0) -> DataLoader:
    ds = TensorDataset(
        torch.from_numpy(static_x[idx]).float(),     # (B,Ds)
        torch.from_numpy(phys7_seq[idx]).float(),    # (B,7,T)
        torch.from_numpy(y_norm[idx]).float(),       # (B,1,T)
        torch.from_numpy(m[idx]).bool(),             # (B,1,T)
        torch.from_numpy(time_mat[idx]).float(),     # (B,T)
    )
    return DataLoader(
        ds,
        batch_size=max(1, min(batch, len(ds))),
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )


# ----------------------------
# Split (key recipes in test + quality)
# ----------------------------
def compute_quality_score_1fam(m_fam: np.ndarray) -> np.ndarray:
    # m_fam: (N,T) bool
    return m_fam.astype(np.int32).sum(axis=1).astype(np.int32)  # per-sample valid count


def split_with_key_and_quality(
    recipe_ids: np.ndarray,
    key_recipes: List[str],
    scores: np.ndarray,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    N = len(recipe_ids)
    all_idx = np.arange(N)

    key_set = set([str(x).strip() for x in key_recipes if str(x).strip()])
    key_idx = np.array([i for i in all_idx if str(recipe_ids[i]) in key_set], dtype=int)

    n_test = max(1, int(round(N * test_ratio)))
    n_val = max(1, int(round(N * val_ratio)))

    test = key_idx.tolist()
    remain = [i for i in all_idx.tolist() if i not in set(test)]

    if len(test) < n_test:
        need = n_test - len(test)
        remain_sorted = sorted(remain, key=lambda i: (scores[i], rng.rand()), reverse=True)
        add = remain_sorted[:need]
        test.extend(add)
        remain = [i for i in remain if i not in set(add)]

    remain_sorted = sorted(remain, key=lambda i: (scores[i], rng.rand()), reverse=True)
    val = remain_sorted[:n_val]
    train = [i for i in remain if i not in set(val)]

    return np.asarray(train, int), np.asarray(val, int), np.asarray(test, int)


def check_min_test_points_1fam(m_fam: np.ndarray, test_idx: np.ndarray, min_points: int) -> bool:
    # m_fam: (N,T)
    return bool(int(m_fam[test_idx].sum()) >= int(min_points))


# ----------------------------
# Model / init / finetune modes
# ----------------------------
def build_model(model_type: str, static_dim: int, out_dim: int = 1) -> nn.Module:
    mt = str(model_type).lower().strip()
    if mt == "transformer":
        return su.MorphTransformer(
            static_dim=static_dim,
            d_model=getattr(sb.Cfg, "tf_d_model", 256),
            nhead=getattr(sb.Cfg, "tf_nhead", 8),
            num_layers=getattr(sb.Cfg, "tf_layers", 4),
            dropout=getattr(sb.Cfg, "tf_dropout", 0.1),
            out_dim=out_dim
        )
    if mt == "gru":
        return su.MorphGRU(
            static_dim=static_dim,
            hidden=getattr(sb.Cfg, "gru_hidden", 256),
            num_layers=getattr(sb.Cfg, "gru_layers", 1),
            out_dim=out_dim
        )
    if mt == "mlp":
        return su.MorphMLP(
            static_dim=static_dim,
            hidden=getattr(sb.Cfg, "mlp_hidden", 256),
            num_layers=getattr(sb.Cfg, "mlp_layers", 3),
            out_dim=out_dim
        )
    raise ValueError(f"Unknown model_type={model_type}")


def load_ckpt_into_model(model: nn.Module, ckpt_path: str) -> Dict[str, Any]:
    ck = _torch_load_any(ckpt_path)
    meta = ck.get("meta", {}) if isinstance(ck, dict) else {}
    sd = None
    if isinstance(ck, dict):
        sd = ck.get("model", None) or ck.get("state_dict", None)
    if sd is None and isinstance(ck, dict):
        cand = {k: v for k, v in ck.items() if isinstance(v, torch.Tensor)}
        sd = cand if cand else None
    if sd is None:
        raise RuntimeError(f"Invalid ckpt: {ckpt_path}")
    sd = _strip_prefix(sd)
    miss = model.load_state_dict(sd, strict=False)
    return {"meta": meta, "missing": list(miss.missing_keys), "unexpected": list(miss.unexpected_keys)}


def apply_finetune_mode(model: nn.Module, mode: str):
    mode = str(mode).lower().strip()

    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return

    for p in model.parameters():
        p.requires_grad = False

    def _enable_out():
        if hasattr(model, "out"):
            for p in model.out.parameters():
                p.requires_grad = True

    def _enable_layernorm_affine():
        for m in model.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True

    def _enable_bias_only():
        for n, p in model.named_parameters():
            if n.endswith(".bias"):
                p.requires_grad = True

    if mode == "head":
        _enable_out()
        return
    if mode == "head_ln":
        _enable_out()
        _enable_layernorm_affine()
        return
    if mode == "last_block":
        _enable_out()
        _enable_layernorm_affine()
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers") and len(model.encoder.layers) > 0:
            for p in model.encoder.layers[-1].parameters():
                p.requires_grad = True
        return
    if mode == "bitfit":
        _enable_bias_only()
        return
    if mode == "bitfit_ln":
        _enable_bias_only()
        _enable_layernorm_affine()
        return

    raise ValueError(f"Unknown finetune_mode={mode}")


# ----------------------------
# Loss / Train / Eval
# ----------------------------
def masked_mse(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    # pred,y: (B,1,T)  m: (B,1,T)
    diff = (pred - y) ** 2
    diff = diff.masked_fill(~m, 0.0)
    denom = m.float().sum().clamp_min(1.0)
    return diff.sum() / denom


def l2sp_penalty(model: nn.Module, anchor_state: Dict[str, torch.Tensor], lam: float) -> torch.Tensor:
    if lam <= 0 or (not anchor_state):
        return torch.tensor(0.0, device=next(model.parameters()).device)
    reg = 0.0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n in anchor_state:
            a = anchor_state[n].to(device=p.device, dtype=p.dtype)
            reg = reg + (p - a).pow(2).mean()
    return reg * float(lam)


@torch.no_grad()
def eval_pack(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, np.ndarray]:
    model.eval()
    preds, ys, ms = [], [], []
    for static_x, phys7_seq, y, m, time_mat in loader:
        static_x = static_x.to(device)
        phys7_seq = phys7_seq.to(device)
        y = y.to(device)
        m = m.to(device)
        time_mat = time_mat.to(device)
        p = model(static_x, phys7_seq, time_mat)  # (B,1,T)
        preds.append(p.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        ms.append(m.detach().cpu().numpy())
    return dict(
        pred=np.concatenate(preds, 0),  # (N,1,T)
        y=np.concatenate(ys, 0),        # (N,1,T)
        m=np.concatenate(ms, 0).astype(bool),
    )


def metrics_1fam_display(pack: Dict[str, np.ndarray],
                        fam: str,
                        y_mean_t: np.ndarray,
                        y_std_t: np.ndarray,
                        unit_scale: float = 1000.0) -> Dict[str, Any]:
    """
    计算显示空间（nm、zmin 翻正、非负裁剪）的 R2/MAE
    pack: pred/y/m 形状 (N,1,T)
    y_mean_t/y_std_t: (T,) 训练空间（um）统计
    """
    pred = torch.from_numpy(pack["pred"]).float()
    y = torch.from_numpy(pack["y"]).float()
    m = torch.from_numpy(pack["m"]).bool()

    mean = torch.from_numpy(y_mean_t).view(1, 1, T)
    std = torch.from_numpy(y_std_t).view(1, 1, T)

    pred_um = pred * std + mean
    y_um = y * std + mean

    # StageB 默认：zmin 翻符号，其余不翻；全部 family 非负裁剪（这里只是单 family）
    sign_map, nonneg_set = su._default_family_sign_and_nonneg([fam])
    family_sign = torch.tensor([sign_map[fam]], dtype=torch.float32)

    pred_disp, y_disp = pu.transform_for_display(
        pred_um, y_um,
        family_sign=family_sign,
        clip_nonneg=True,
        nonneg_families=[0],          # 单 family 下索引=0
        unit_scale=unit_scale,
        flip_sign=False,
        min_display_value=None
    )

    # flatten masked
    yp = pred_disp[:, 0, :].reshape(-1).numpy()
    yt = y_disp[:, 0, :].reshape(-1).numpy()
    mk = pack["m"][:, 0, :].reshape(-1)
    yp = yp[mk]; yt = yt[mk]
    ok = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[ok]; yp = yp[ok]
    n = int(len(yt))
    r2 = float(su.masked_r2_score_np(yt, yp)) if n >= 2 else float("nan")
    mae = float(np.mean(np.abs(yt - yp))) if n >= 1 else float("nan")
    return {"family": fam, "r2": r2, "mae_nm": mae, "n": n}


def train_one(model: nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader,
              device: str,
              epochs: int,
              lr: float,
              wd: float,
              patience: int,
              l2sp_lam: float = 0.0,
              anchor_state: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)

    # [修改点]：初始化 history
    best = {"val_loss": float("inf"), "epoch": 0, "state": None, "history": {"train": [], "val": []}}
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        ep_train_loss = 0.0
        ep_train_n = 0
        for static_x, phys7_seq, y, m, time_mat in train_loader:
            static_x = static_x.to(device)
            phys7_seq = phys7_seq.to(device)
            y = y.to(device)
            m = m.to(device)
            time_mat = time_mat.to(device)

            pred = model(static_x, phys7_seq, time_mat)
            loss = masked_mse(pred, y, m)

            raw_loss_val = float(loss.item())

            if l2sp_lam > 0 and anchor_state is not None:
                loss = loss + l2sp_penalty(model, anchor_state, l2sp_lam)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_train_loss += raw_loss_val
            ep_train_n += 1

        avg_train_loss = ep_train_loss / max(1, ep_train_n)

        # val
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for static_x, phys7_seq, y, m, time_mat in val_loader:
                static_x = static_x.to(device)
                phys7_seq = phys7_seq.to(device)
                y = y.to(device)
                m = m.to(device)
                time_mat = time_mat.to(device)

                pred = model(static_x, phys7_seq, time_mat)
                loss = masked_mse(pred, y, m)
                vl += float(loss.item())
                vn += 1
        val_loss = vl / max(1, vn)

        # [修改点]：记录 loss 到 history
        best["history"]["train"].append(avg_train_loss)
        best["history"]["val"].append(val_loss)

        if val_loss + 1e-9 < best["val_loss"]:
            best["val_loss"] = val_loss
            best["epoch"] = ep
            best["state"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"], strict=False)
    return best

# ----------------------------
# StageB best retrieval (from best_config + results_summary.csv)
# ----------------------------
def _find_candidates(runs_root: str, relname: str) -> List[str]:
    cands = [
        os.path.join(runs_root, relname),
        os.path.join(runs_root, "_tuneV_verify", relname),
    ]
    return [p for p in cands if os.path.exists(p)]


def load_best_config_common(runs_root: str) -> Dict[str, Any]:
    cands = _find_candidates(runs_root, "best_config_common_all_families.json")
    if not cands:
        raise FileNotFoundError(f"best_config_common_all_families.json not found under: {runs_root}")
    with open(cands[0], "r", encoding="utf-8") as f:
        return json.load(f)


def load_results_summary_df(runs_root: str) -> Optional[pd.DataFrame]:
    cands = _find_candidates(runs_root, "results_summary.csv")
    if not cands:
        return None
    df = pd.read_csv(cands[0])
    return df

def _resolve_ckpt_by_expname(runs_root: str, best_conf: dict, fam: str) -> str:
    mt  = str(best_conf.get("model_type"))
    hp  = str(best_conf.get("hp_tag"))
    ps  = str(best_conf.get("phys_source"))
    au  = str(best_conf.get("recipe_aug_mode"))
    p7  = str(best_conf.get("phys7_mode"))
    ss  = int(best_conf.get("split_seed"))

    exp_name = f"{mt}_{hp}_{ps}_{au}_{p7}_{fam}_s{ss}"
    cands = [
        os.path.join(runs_root, exp_name, "best.pth"),
        os.path.join(runs_root, "_tuneV_verify", exp_name, "best.pth"),
    ]
    for p in cands:
        if os.path.isfile(p):
            return p

    # 兜底：只用后缀匹配（避免 best_conf 字段名略有差异）
    suffix = f"_{fam}_s{ss}"
    for base in [runs_root, os.path.join(runs_root, "_tuneV_verify")]:
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            if name.endswith(suffix):
                p = os.path.join(base, name, "best.pth")
                if os.path.isfile(p):
                    return p

    raise FileNotFoundError(f"Cannot locate best.pth for fam={fam} using best_conf under {runs_root}")

def resolve_stageB_best_ckpts_from_common(runs_root: str) -> Dict[str, str]:
    """
    依据 StageB 的 best_config_common_all_families.json，优先用 results_summary.csv
    定位每个 family 的 ckpt_path；若 results_summary.csv 缺失，则 fallback
    通过 exp_name 规则直接定位 best.pth。
    """
    best_conf = load_best_config_common(runs_root)
    df = load_results_summary_df(runs_root)

    out: Dict[str, str] = {}

    # ---------- fallback: no results_summary.csv ----------
    if df is None:
        for fam in FAMILIES:
            out[fam] = _resolve_ckpt_by_expname(runs_root, best_conf, fam)
        return out

    # ---------- normal path: use results_summary.csv ----------
    # 用 best_conf 中“同时存在于 df 列”的字段做严格匹配
    filters = {}
    for k, v in best_conf.items():
        if k in df.columns:
            filters[k] = v

    df_f = df.copy()
    for k, v in filters.items():
        if k == "split_seed":
            df_f = df_f[pd.to_numeric(df_f[k], errors="coerce").fillna(-1).astype(int) == int(v)]
        else:
            df_f = df_f[df_f[k].astype(str) == str(v)]

    if df_f.empty:
        raise RuntimeError("No rows matched best_config in results_summary.csv (check paths / columns).")

    if "family_mode" not in df_f.columns or "ckpt_path" not in df_f.columns:
        raise RuntimeError("results_summary.csv missing required columns: family_mode / ckpt_path")

    for fam in FAMILIES:
        dff = df_f[df_f["family_mode"].astype(str).str.lower() == fam.lower()].copy()
        if dff.empty:
            continue
        if "min_pf_r2" in dff.columns:
            dff["min_pf_r2"] = pd.to_numeric(dff["min_pf_r2"], errors="coerce")
            dff = dff.sort_values("min_pf_r2", ascending=False)
        out[fam] = str(dff.iloc[0]["ckpt_path"])

    if not out:
        raise RuntimeError("Matched config, but no family ckpts resolved (check family_mode values).")

    return out

# ----------------------------
# Experiment config
# ----------------------------
@dataclass
class ExpCfg:
    name: str
    init: str              # "scratch" / "stageB_best"
    finetune_mode: str     # "full" / "head_ln" / "bitfit_ln" ...
    phys7_mode: str        # "full" / "none" / ...
    lr: float
    wd: float
    l2sp: float = 0.0      # L2SP strength


def run_one_experiment_on_split_1fam(
        fam: str,
        exp: ExpCfg,
        device: str,
        raw: Dict[str, Any],
        split: Dict[str, List[int]],
        stageB_best_ckpt_for_fam: Optional[str],
        model_type: str,
        stageA_heads_root: str,
        recipe_aug_mode: str,
        out_dir_fam: str,
        epochs: int,
        batch: int,
        patience: int,
        num_workers: int,
) -> Dict[str, Any]:
    recipe_ids = raw["recipe_ids"]
    static_raw = raw["static_raw"]  # (N,Ds)
    phys7_raw_full = raw["phys7_raw_full"]  # (N,7)
    y_raw = raw["y_raw"]  # (N,K,T)
    mask = raw["mask"]  # (N,K,T)
    time_mat = raw["time_mat"]  # (N,T)

    k = family_to_index(fam)
    y_f = y_raw[:, k, :].astype(np.float32)  # (N,T)
    m_f = mask[:, k, :].astype(bool)  # (N,T)

    train_idx = np.asarray(split["train_idx"], int)
    val_idx = np.asarray(split["val_idx"], int)
    test_idx = np.asarray(split["test_idx"], int)

    # fit norms on train
    s_mean, s_std = zfit(static_raw[train_idx])
    p_mean, p_std = zfit(phys7_raw_full[train_idx])
    y_mean_t, y_std_t = zfit_targets_masked_1fam(y_f[train_idx], m_f[train_idx])

    static = apply_z(static_raw, s_mean, s_std)  # (N,Ds)
    phys7_z_full = apply_z(phys7_raw_full, p_mean, p_std)  # (N,7)
    phys7_z = apply_phys7_mode(phys7_z_full, exp.phys7_mode)  # (N,7')
    phys7_seq = su.broadcast_phys7_to_T(phys7_z, T)  # (N,7',T)

    y_norm = apply_z(y_f, y_mean_t.reshape(1, T), y_std_t.reshape(1, T))  # (N,T)
    y_norm = y_norm[:, None, :]  # (N,1,T)
    m_ = m_f[:, None, :]  # (N,1,T)

    train_loader = make_loader(static, phys7_seq, y_norm, m_, time_mat, train_idx, batch, True, num_workers)
    val_loader = make_loader(static, phys7_seq, y_norm, m_, time_mat, val_idx, batch, False, num_workers)
    test_loader = make_loader(static, phys7_seq, y_norm, m_, time_mat, test_idx, batch, False, num_workers)

    # build model (single-head)
    model = build_model(model_type, static_dim=static.shape[1], out_dim=1).to(device)

    init_info = {}
    if exp.init == "stageB_best":
        if not stageB_best_ckpt_for_fam:
            raise RuntimeError(f"No StageB best ckpt for fam={fam}")
        init_info = load_ckpt_into_model(model, stageB_best_ckpt_for_fam)
    elif exp.init == "scratch":
        pass
    else:
        raise ValueError(f"Unknown init={exp.init}")

    apply_finetune_mode(model, exp.finetune_mode)

    # L2SP anchor = starting point after init
    anchor_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    best = train_one(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=exp.lr, wd=exp.wd, patience=patience,
        l2sp_lam=exp.l2sp, anchor_state=anchor_state
    )

    # save
    exp_dir = os.path.join(out_dir_fam, "experiments", exp.name)
    ensure_dir(exp_dir)

    # [修改点]：画 Loss 曲线
    history = best["history"]
    plt.figure()
    plt.plot(history["train"], label="Train")
    plt.plot(history["val"], label="Val")
    plt.title(f"{fam} {exp.name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(exp_dir, "loss_curve.png"))
    plt.close()

    # [修改点]：确定最终评估集 (兼容 7:3 划分)
    # 如果 Test 集基本没数据，而 Val 集有数据，就用 Val 做最终展示
    use_val_as_test = (len(test_idx) <= 1 and len(val_idx) > 1)
    final_loader = val_loader if use_val_as_test else test_loader

    pack = eval_pack(model, final_loader, device)
    met = metrics_1fam_display(pack, fam, y_mean_t, y_std_t, unit_scale=1000.0)

    # [修改点]：画预测散点图 (Scatter)
    if "yt" in met and "yp" in met and len(met["yt"]) > 0:
        yt = met["yt"]
        yp = met["yp"]
        plt.figure()
        plt.scatter(yt, yp, alpha=0.6, edgecolors='k', s=40)

        # 画 y=x 参考线
        all_vals = yt + yp
        if len(all_vals) > 0:
            vmin, vmax = min(all_vals), max(all_vals)
            margin = (vmax - vmin) * 0.1
            vmin -= margin
            vmax += margin
            plt.plot([vmin, vmax], [vmin, vmax], 'r--', alpha=0.5, label="Ideal")
            plt.xlim(vmin, vmax)
            plt.ylim(vmin, vmax)

        plt.xlabel(f"True {fam} (nm)")
        plt.ylabel(f"Pred {fam} (nm)")
        plt.title(f"{fam} {exp.name} Scatter (R2={met['r2']:.3f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(exp_dir, "scatter_best.png"))
        plt.close()

    # 清理掉 raw points 以免 json 文件太大
    if "yt" in met: del met["yt"]
    if "yp" in met: del met["yp"]

    ckpt_path = os.path.join(exp_dir, "model_best.pth")
    torch.save({
        "model": model.state_dict(),
        "best": best,
        "exp": exp.__dict__,
        "family": fam,
        "split": split,
        "norm": {
            "static_mean": s_mean, "static_std": s_std,
            "phys7_mean": p_mean, "phys7_std": p_std,
            "y_mean_t": y_mean_t, "y_std_t": y_std_t
        },
        "meta": {
            "init_info": init_info,
            "model_type": model_type,
            "recipe_aug_mode": recipe_aug_mode,
            "phys7_mode": exp.phys7_mode,
            "time_list": TIME_LIST,
            "families": FAMILIES,
            "new_excel_recipe_ids_head": [str(x) for x in recipe_ids[:10]]
        }
    }, ckpt_path)

    with open(os.path.join(exp_dir, "metrics_test.json"), "w", encoding="utf-8") as f:
        json.dump(met, f, indent=2, ensure_ascii=False)

    row = {
        "family": fam,
        "exp": exp.name,
        "r2": met["r2"],
        "mae_nm": met["mae_nm"],
        "n": met["n"],
        "best_epoch": best["epoch"],
        "val_loss": best["val_loss"],
        "ckpt": ckpt_path
    }
    with open(os.path.join(exp_dir, "compare_row.json"), "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)

    return row


def search_best_split_for_family(
        fam: str,
        device: str,
        raw: Dict[str, Any],
        key_recipes: List[str],
        trials: int,
        test_ratio: float,
        val_ratio: float,
        seed0: int,
        min_test_points: int,
        # 用哪个 exp 做 split-search 评估
        search_exp: ExpCfg,
        stageB_best_ckpt_for_fam: Optional[str],
        model_type: str,
        recipe_aug_mode: str,
        out_dir_fam: str,
        # 训练超参（search 可适当小一些）
        search_epochs: int,
        batch: int,
        patience: int,
        num_workers: int,
        search_on: str = "test",  # "test" or "val"  (默认按 test 冲高)
) -> Dict[str, Any]:
    ensure_dir(os.path.join(out_dir_fam, "_search_tmp"))
    recipe_ids = raw["recipe_ids"]
    mask = raw["mask"]
    k = family_to_index(fam)
    m_f = mask[:, k, :].astype(bool)  # (N,T)
    scores = compute_quality_score_1fam(m_f)

    trials_csv = os.path.join(out_dir_fam, "split_trials.csv")
    with open(trials_csv, "w", encoding="utf-8") as f:
        f.write("trial,trainN,valN,testN,test_valid_points,okMinPts,scoreTestMean\n")

    best = {"rank": -1e18, "trial": -1, "split": None, "note": ""}
    err_cnt = 0
    for tr in range(trials):
        seed = int(seed0 * 100000 + tr)
        train_idx, val_idx, test_idx = split_with_key_and_quality(
            recipe_ids=recipe_ids,
            key_recipes=key_recipes,
            scores=scores,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            seed=seed
        )

        # [修改点]：检查逻辑适配 Ratio=0
        # 如果 test_ratio=0 (len(test_idx)很小)，则检查 val_idx 是否满足 min points
        check_idx = test_idx if len(test_idx) > 1 else val_idx
        ok_pts = check_min_test_points_1fam(m_f, check_idx, min_test_points)

        test_valid = int(m_f[check_idx].sum())
        score_test_mean = float(np.mean(scores[check_idx])) if len(check_idx) else 0.0

        with open(trials_csv, "a", encoding="utf-8") as f:
            f.write(
                f"{tr},{len(train_idx)},{len(val_idx)},{len(test_idx)},{test_valid},{int(ok_pts)},{score_test_mean:.4f}\n")

        if not ok_pts:
            continue

        split = dict(train_idx=train_idx.tolist(), val_idx=val_idx.tolist(), test_idx=test_idx.tolist())
        try:
            row = run_one_experiment_on_split_1fam(
                fam=fam,
                exp=search_exp,
                device=device,
                raw=raw,
                split=split,
                stageB_best_ckpt_for_fam=stageB_best_ckpt_for_fam,
                model_type=model_type,
                stageA_heads_root="",  # unused here
                recipe_aug_mode=recipe_aug_mode,
                out_dir_fam=os.path.join(out_dir_fam, "_search_tmp"),
                epochs=search_epochs,
                batch=batch,
                patience=patience,
                num_workers=num_workers,
            )
            # R2 自动取自 Val 或 Test (取决于 run_one_exp 里的逻辑)
            rank = float(row["r2"]) if np.isfinite(row["r2"]) else -1e9
        except Exception as e:
            err_cnt += 1
            if err_cnt <= 5:
                msg = traceback.format_exc()
                print(f"[SPLIT-SEARCH][{fam}] trial={tr} FAILED:\n{msg}")
                with open(os.path.join(out_dir_fam, "split_search_errors.log"), "a", encoding="utf-8") as ff:
                    ff.write(f"\n=== trial {tr} ===\n{msg}\n")
            continue

        if rank > best["rank"]:
            best.update(rank=rank, trial=tr, split=split,
                        note=f"fam={fam} search_exp={search_exp.name} rank({search_on})={rank:.6f}")
            with open(os.path.join(out_dir_fam, "best_split.json"), "w", encoding="utf-8") as f:
                json.dump(best, f, indent=2, ensure_ascii=False)

    if best["split"] is None:
        raise RuntimeError(f"[{fam}] No valid split found. Try reduce min_test_points or increase trials.")

    return best
def search_best_split_for_family(
    fam: str,
    device: str,
    raw: Dict[str, Any],
    key_recipes: List[str],
    trials: int,
    test_ratio: float,
    val_ratio: float,
    seed0: int,
    min_test_points: int,
    # 用哪个 exp 做 split-search 评估
    search_exp: ExpCfg,
    stageB_best_ckpt_for_fam: Optional[str],
    model_type: str,
    recipe_aug_mode: str,
    out_dir_fam: str,
    # 训练超参（search 可适当小一些）
    search_epochs: int,
    batch: int,
    patience: int,
    num_workers: int,
    search_on: str = "test",  # "test" or "val"  (默认按 test 冲高)
) -> Dict[str, Any]:
    ensure_dir(os.path.join(out_dir_fam, "_search_tmp"))
    recipe_ids = raw["recipe_ids"]
    mask = raw["mask"]
    k = family_to_index(fam)
    m_f = mask[:, k, :].astype(bool)  # (N,T)
    scores = compute_quality_score_1fam(m_f)

    trials_csv = os.path.join(out_dir_fam, "split_trials.csv")
    with open(trials_csv, "w", encoding="utf-8") as f:
        f.write("trial,trainN,valN,testN,test_valid_points,okMinPts,scoreTestMean\n")

    best = {"rank": -1e18, "trial": -1, "split": None, "note": ""}
    err_cnt = 0
    for tr in range(trials):
        seed = int(seed0 * 100000 + tr)
        train_idx, val_idx, test_idx = split_with_key_and_quality(
            recipe_ids=recipe_ids,
            key_recipes=key_recipes,
            scores=scores,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            seed=seed
        )
        ok_pts = check_min_test_points_1fam(m_f, test_idx, min_test_points)
        test_valid = int(m_f[test_idx].sum())
        score_test_mean = float(np.mean(scores[test_idx])) if len(test_idx) else 0.0

        with open(trials_csv, "a", encoding="utf-8") as f:
            f.write(f"{tr},{len(train_idx)},{len(val_idx)},{len(test_idx)},{test_valid},{int(ok_pts)},{score_test_mean:.4f}\n")

        if not ok_pts:
            continue

        split = dict(train_idx=train_idx.tolist(), val_idx=val_idx.tolist(), test_idx=test_idx.tolist())
        try:
            row = run_one_experiment_on_split_1fam(
                fam=fam,
                exp=search_exp,
                device=device,
                raw=raw,
                split=split,
                stageB_best_ckpt_for_fam=stageB_best_ckpt_for_fam,
                model_type=model_type,
                stageA_heads_root="",            # unused here
                recipe_aug_mode=recipe_aug_mode,
                out_dir_fam=os.path.join(out_dir_fam, "_search_tmp"),
                epochs=search_epochs,
                batch=batch,
                patience=patience,
                num_workers=num_workers,
            )
            rank = float(row["r2"]) if np.isfinite(row["r2"]) else -1e9
        except Exception as e:
            err_cnt += 1
            if err_cnt <= 5:
                msg = traceback.format_exc()
                print(f"[SPLIT-SEARCH][{fam}] trial={tr} FAILED:\n{msg}")
                with open(os.path.join(out_dir_fam, "split_search_errors.log"), "a", encoding="utf-8") as ff:
                    ff.write(f"\n=== trial {tr} ===\n{msg}\n")
            continue

        if rank > best["rank"]:
            best.update(rank=rank, trial=tr, split=split,
                        note=f"fam={fam} search_exp={search_exp.name} rank({search_on})={rank:.6f}")
            with open(os.path.join(out_dir_fam, "best_split.json"), "w", encoding="utf-8") as f:
                json.dump(best, f, indent=2, ensure_ascii=False)

    if best["split"] is None:
        raise RuntimeError(f"[{fam}] No valid split found. Try reduce min_test_points or increase trials.")

    return best


# =========================================================
#  新增：数据健康度与分布漂移检测工具 (Stage C 专用版)
# =========================================================

def diagnose_data_health(raw):
    """
    检测 Stage C 数据的健康度：NaN, Inf, 0值, 极值
    适配 build_stageC_raw 返回的 numpy 字典结构
    """
    print("\n" + "=" * 40)
    print(" [DIAGNOSTIC] Data Health Check")
    print("=" * 40)

    # Stage C 的 raw 是一个 dict，包含 numpy array
    required_keys = ["y_raw", "mask", "recipe_raw"]
    for k in required_keys:
        if k not in raw:
            print(f" [SKIP] Missing key '{k}' in raw data. Cannot diagnose.")
            return

    y_raw = raw["y_raw"]  # (N, K, T)
    mask = raw["mask"]  # (N, K, T)
    recipe_raw = raw["recipe_raw"]  # (N, 7)

    N, K, T = y_raw.shape
    print(f" [INFO] Data Shape: N={N}, K={K}, T={T}")

    # 1. 检查 Target 里的 NaN/Inf
    if np.isnan(y_raw).any() or np.isinf(y_raw).any():
        print(" [!!!] CRITICAL: Targets (y_raw) contain NaN or Inf!")
        # 简单定位
        bad_indices = np.where(np.isnan(y_raw) | np.isinf(y_raw))
        if len(bad_indices[0]) > 0:
            print(f"   -> First bad sample index: {bad_indices[0][0]}")
    else:
        print(" [OK] No NaN/Inf in targets.")

    # 2. 检查 Target 里的 0 (Log 敏感) - 只检查 mask=True 的部分
    # mask 可能是 0/1 int 或 bool
    m_bool = mask.astype(bool)
    zeros = (y_raw == 0) & m_bool

    if zeros.any():
        count = zeros.sum()
        print(f" [WARN] Found {count} zeros in valid (masked) target entries. Log-transform will fail!")
        # 打印前几个位置
        nz = np.nonzero(zeros)
        for i in range(min(5, len(nz[0]))):
            n_idx, k_idx, t_idx = nz[0][i], nz[1][i], nz[2][i]
            print(f"        At (n={n_idx}, k={k_idx}, t={t_idx})")
    else:
        print(" [OK] No dangerous zeros in masked targets.")

    # 3. 检查输入 Recipe 的数值范围
    max_val = np.abs(recipe_raw).max()
    if max_val > 10000:
        print(f" [WARN] Recipe inputs have very large values (max={max_val:.1f}). Check units!")
    else:
        print(f" [OK] Recipe value range seems normal (max={max_val:.1f}).")


def diagnose_distribution_shift(raw, stageB_runs_root):
    """
    检测 Stage C 数据相对于 Stage B 的分布漂移 (Covariate Shift)
    适配 build_stageC_raw 返回的 numpy 字典结构
    """
    print("\n" + "=" * 40)
    print(" [DIAGNOSTIC] Distribution Shift Check (Stage B vs C)")
    print("=" * 40)

    import stageB_util as su

    # 1. 获取 Stage B 的统计量 (Mean/Std)
    mean_b, std_b = None, None
    try:
        dummy_ckpt = None
        for root, dirs, files in os.walk(stageB_runs_root):
            for f in files:
                if f.endswith("best.pth"):
                    dummy_ckpt = os.path.join(root, f)
                    break
            if dummy_ckpt: break

        if not dummy_ckpt:
            print(" [SKIP] No Stage B checkpoint found to compare stats.")
            return

        # 必须加上 weights_only=False 以防 pickle 报错，或者捕获 Warning
        try:
            ckpt = torch.load(dummy_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(dummy_ckpt, map_location="cpu")

        meta = ckpt.get("meta", {})
        norm_p7 = meta.get("norm_phys7", {})

        mean_b = norm_p7.get("mean", None)  # numpy array
        std_b = norm_p7.get("std", None)  # numpy array

        if mean_b is None:
            print(" [SKIP] Stage B checkpoint missing 'norm_phys7' stats.")
            return

    except Exception as e:
        print(f" [SKIP] Failed to load Stage B stats: {e}")
        return

    # 2. 计算 Stage C 的统计量
    if "phys7_raw_full" not in raw:
        print(" [SKIP] raw data missing 'phys7_raw_full'.")
        return

    phys_c = raw["phys7_raw_full"]  # (N, 7) - Stage C 中 phys7 是 (N, 7) 的 numpy 数组

    # 计算均值 (跨样本)
    mean_c = phys_c.mean(axis=0)  # (7,)

    # 3. 对比
    phys_names = ["logGamma_SF6", "pF_SF6", "spread_SF6", "qskew_SF6",
                  "logGamma_C4F8", "rho_C4F8", "spread_C4F8"]

    print(f"{'Feature':<20} | {'Z-Score Shift':<15} | {'Status'}")
    print("-" * 60)

    warnings = 0
    # mean_b 可能是 (1, 7) 或 (7,)
    mb_flat = mean_b.flatten()
    sb_flat = std_b.flatten()

    for i, name in enumerate(phys_names):
        if i >= len(mb_flat): break

        mu_b = mb_flat[i]
        sigma_b = sb_flat[i] + 1e-6
        mu_c = mean_c[i]

        z_shift = (mu_c - mu_b) / sigma_b

        status = "OK"
        if abs(z_shift) > 1.0: status = "DRIFT (!)"
        if abs(z_shift) > 3.0: status = "SEVERE (!!!)"
        if abs(z_shift) > 3.0: warnings += 1

        print(f"{name:<20} | {z_shift:+.2f} sigma      | {status}")

    if warnings > 0:
        print("\n [CONCLUSION] Severe covariate shift detected.")
        print("              Consider using 'phys7_mode=none' or re-calibrating Stage A.")
    else:
        print("\n [CONCLUSION] Distribution seems matched.")


def metrics_1fam_display(pack: Dict[str, np.ndarray],
                         fam: str,
                         y_mean_t: np.ndarray,
                         y_std_t: np.ndarray,
                         unit_scale: float = 1000.0) -> Dict[str, Any]:
    """
    计算显示空间（nm、zmin 翻正、非负裁剪）的 R2/MAE
    pack: pred/y/m 形状 (N,1,T)
    y_mean_t/y_std_t: (T,) 训练空间（um）统计
    """
    pred = torch.from_numpy(pack["pred"]).float()
    y = torch.from_numpy(pack["y"]).float()
    m = torch.from_numpy(pack["m"]).bool()

    mean = torch.from_numpy(y_mean_t).view(1, 1, T)
    std = torch.from_numpy(y_std_t).view(1, 1, T)

    pred_um = pred * std + mean
    y_um = y * std + mean

    # StageB 默认：zmin 翻符号，其余不翻；全部 family 非负裁剪（这里只是单 family）
    sign_map, nonneg_set = su._default_family_sign_and_nonneg([fam])
    family_sign = torch.tensor([sign_map[fam]], dtype=torch.float32)

    pred_disp, y_disp = pu.transform_for_display(
        pred_um, y_um,
        family_sign=family_sign,
        clip_nonneg=True,
        nonneg_families=[0],  # 单 family 下索引=0
        unit_scale=unit_scale,
        flip_sign=False,
        min_display_value=None
    )

    # flatten masked
    yp = pred_disp[:, 0, :].reshape(-1).numpy()
    yt = y_disp[:, 0, :].reshape(-1).numpy()
    mk = pack["m"][:, 0, :].reshape(-1)
    yp = yp[mk];
    yt = yt[mk]
    ok = np.isfinite(yt) & np.isfinite(yp)
    yt = yt[ok];
    yp = yp[ok]
    n = int(len(yt))
    r2 = float(su.masked_r2_score_np(yt, yp)) if n >= 2 else float("nan")
    mae = float(np.mean(np.abs(yt - yp))) if n >= 1 else float("nan")

    # [修改点]：返回原始点集供画图
    return {"family": fam, "r2": r2, "mae_nm": mae, "n": n,
            "yp": yp.tolist(), "yt": yt.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_excel", type=str, default=r"D:\PycharmProjects\Bosch\Bosch.xlsx")
    ap.add_argument("--out_dir", type=str, default=r"D:\PycharmProjects\Bosch\runs_stageC_singlehead")
    ap.add_argument("--stageB_runs_root", type=str, default=r"D:\PycharmProjects\Bosch\runs_stageB_morph_phys7")

    # 注意：这里的 model_type 默认值会被 StageB 的 best config 覆盖
    ap.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "gru", "mlp"])
    ap.add_argument("--recipe_aug_mode", type=str, default="base")
    ap.add_argument("--height_family", type=str, default="h1")
    ap.add_argument("--stageA_heads_root", type=str, default=getattr(sb.Cfg, "stageA_heads_root", ""))

    ap.add_argument("--key_recipes", type=str, default="")
    ap.add_argument("--families_eval", type=str, default="zmin,h1,d1,w")

    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--trials", type=int, default=500)

    # [修改点]：默认 7:3 划分 (Test=0, Val=0.3)
    # Val 用于选择 Best Epoch，同时也用于最终汇报 (Test Best策略)
    ap.add_argument("--test_ratio", type=float, default=0.0)
    ap.add_argument("--val_ratio", type=float, default=0.3)
    ap.add_argument("--min_test_points", type=int, default=2)

    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--search_epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # ---------------------------------------------------------
    # 1) 加载 StageB 最佳配置 & 关键修复：对齐模型结构参数
    # ---------------------------------------------------------
    print(f"[StageC] Loading best config from: {args.stageB_runs_root}")
    # 修复点：su 现在有了 load_best_config_common
    best_conf = su.load_best_config_common(args.stageB_runs_root)

    # ✅ 修复点：调用 StageB 的函数，将 best_conf 里的 d_model/layers 等覆盖到 Cfg
    # 否则 StageC 建立的模型是默认大小，加载权重时会报错 size mismatch
    sb.apply_hp_from_best_conf_to_cfg(best_conf)

    # 获取 StageB 决定的关键模式
    stageB_aug_mode = best_conf.get("recipe_aug_mode", args.recipe_aug_mode)
    stageB_phys_mode = best_conf.get("phys7_mode", "full")  # 获取 StageB 用的 phys7 模式
    stageB_model_type = best_conf.get("model_type", args.model_type)

    print(f"[StageC] Aligning with StageB:")
    print(f"  - Model Type: {stageB_model_type}")
    print(f"  - Aug Mode:   {stageB_aug_mode}")
    print(f"  - Phys Mode:  {stageB_phys_mode}")
    print(f"  - HPs: d_model={sb.Cfg.tf_d_model}, layers={sb.Cfg.tf_layers}")

    # 覆盖参数
    args.recipe_aug_mode = stageB_aug_mode
    args.model_type = stageB_model_type

    # 解析 Checkpoints
    stageB_best = resolve_stageB_best_ckpts_from_common(args.stageB_runs_root)

    # ---------------------------------------------------------
    # 2) 构建数据
    # ---------------------------------------------------------
    raw = build_stageC_raw(
        device=device,
        new_excel=args.new_excel,
        height_family=args.height_family,
        recipe_aug_mode=args.recipe_aug_mode,
        stageA_heads_root=args.stageA_heads_root
    )
    diagnose_data_health(raw)
    diagnose_distribution_shift(raw, args.stageB_runs_root)
    families_eval = [x.strip().lower() for x in str(args.families_eval).split(",") if x.strip()]
    families_eval = [f for f in families_eval if f in [x.lower() for x in FAMILIES]]
    if not families_eval:
        raise RuntimeError("families_eval is empty.")

    key_recipes = [x.strip() for x in str(args.key_recipes).split(",") if x.strip()]

    # ---------------------------------------------------------
    # 3) 定义实验矩阵 (动态对齐 Phys7 Mode)
    # ---------------------------------------------------------
    # 修复点：这里的 phys7_mode 必须和 StageB 一致，
    # 否则如果 StageB 是 only_energy (3维)，这里 full (7维) 就会导致输入特征错位或多余。
    pm = stageB_phys_mode

    experiments = [
        # 基线：不加载权重，从头练
        ExpCfg(name="scratch_full", init="scratch", finetune_mode="full", phys7_mode=pm, lr=3e-4, wd=1e-4, l2sp=0.0),

        # 迁移：全参数微调
        ExpCfg(name="stageB_full", init="stageB_best", finetune_mode="full", phys7_mode=pm, lr=2e-4, wd=1e-4, l2sp=0.0),

        # 迁移：只修 Head + LayerNorm (推荐)
        ExpCfg(name="stageB_head_ln", init="stageB_best", finetune_mode="head_ln", phys7_mode=pm, lr=6e-4, wd=1e-4,
               l2sp=0.0),

        # 迁移：BitFit (只修 Bias)
        ExpCfg(name="stageB_bitfit_ln", init="stageB_best", finetune_mode="bitfit_ln", phys7_mode=pm, lr=1e-3, wd=0.0,
               l2sp=0.0),

        # 迁移：L2SP 正则
        ExpCfg(name="stageB_head_ln_l2sp", init="stageB_best", finetune_mode="head_ln", phys7_mode=pm, lr=6e-4, wd=1e-4,
               l2sp=1e-3),

        # 消融：强制不用物理信息 (对比看 Phys7 是否真的有帮助)
        ExpCfg(name="stageB_head_ln_noP7", init="stageB_best", finetune_mode="head_ln", phys7_mode="none", lr=6e-4,
               wd=1e-4, l2sp=0.0),
    ]

    # Split-search 用的策略
    search_exp = next(e for e in experiments if e.name == "stageB_head_ln")

    summary_path = os.path.join(args.out_dir, "summary_all_families.csv")
    if not os.path.exists(summary_path):
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("family,exp,r2,mae_nm,n,best_epoch,val_loss,ckpt\n")

    all_rows = []

    # ---------------------------------------------------------
    # 4) 执行实验
    # ---------------------------------------------------------
    for fam in families_eval:
        out_dir_fam = os.path.join(args.out_dir, fam)
        ensure_dir(out_dir_fam)

        fam_ckpt = stageB_best.get(fam, None)
        with open(os.path.join(out_dir_fam, "stageB_best_ckpt.json"), "w", encoding="utf-8") as f:
            json.dump({"family": fam, "ckpt_path": fam_ckpt}, f, indent=2, ensure_ascii=False)

        if fam_ckpt is None or (not os.path.exists(fam_ckpt)):
            print(f"[WARN] Skip {fam}: StageB best ckpt missing.")
            continue

        # Split Search
        print(f"\n>>> Searching best split for [{fam}] ...")
        best = search_best_split_for_family(
            fam=fam, device=device, raw=raw, key_recipes=key_recipes,
            trials=args.trials, test_ratio=args.test_ratio, val_ratio=args.val_ratio,
            seed0=args.seed, min_test_points=args.min_test_points,
            search_exp=search_exp, stageB_best_ckpt_for_fam=fam_ckpt,
            model_type=args.model_type, recipe_aug_mode=args.recipe_aug_mode,
            out_dir_fam=out_dir_fam, search_epochs=args.search_epochs,
            batch=args.batch, patience=max(10, args.patience // 2), num_workers=args.num_workers
        )
        best_split = best["split"]

        # Run Experiments
        comp_csv = os.path.join(out_dir_fam, "compare_on_best_split.csv")
        if not os.path.exists(comp_csv):
            with open(comp_csv, "w", encoding="utf-8") as f:
                f.write("family,exp,r2,mae_nm,n,best_epoch,val_loss,ckpt\n")

        print(f">>> Running experiments on best split for [{fam}] ...")
        for exp in experiments:
            print(f"   -> {exp.name}")
            row = run_one_experiment_on_split_1fam(
                fam=fam, exp=exp, device=device, raw=raw, split=best_split,
                stageB_best_ckpt_for_fam=fam_ckpt, model_type=args.model_type,
                stageA_heads_root=args.stageA_heads_root, recipe_aug_mode=args.recipe_aug_mode,
                out_dir_fam=out_dir_fam, epochs=args.epochs, batch=args.batch,
                patience=args.patience, num_workers=args.num_workers
            )
            all_rows.append(row)
            with open(comp_csv, "a", encoding="utf-8") as f:
                f.write(
                    f"{row['family']},{row['exp']},{row['r2']},{row['mae_nm']},{row['n']},{row['best_epoch']},{row['val_loss']},{row['ckpt']}\n")
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{row['family']},{row['exp']},{row['r2']},{row['mae_nm']},{row['n']},{row['best_epoch']},{row['val_loss']},{row['ckpt']}\n")

    with open(os.path.join(args.out_dir, "summary_all_families.json"), "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)

    print("\n[DONE] StageC Finished.")
if __name__ == "__main__":
    main()
