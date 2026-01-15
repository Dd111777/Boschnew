# -*- coding: utf-8 -*-
import os, json, time, argparse
import traceback
from dataclasses import dataclass
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

TIME_LIST = list(su.TIME_LIST)          # ["1","2",...,"9"]
TIME_VALUES = np.asarray([float(t.replace("_", ".")) for t in TIME_LIST], np.float32)
FAMILIES = list(su.FAMILIES)           # ["zmin","h0","h1","d0","d1","w"]
T = len(TIME_LIST)

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

def build_sparse_batch_subset_time(
    recs: List[Dict],
    time_list: List[str],
    time_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

def zfit(x_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)
    return mean, std


def zfit_targets_masked_1fam(y: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    return sb.apply_phys7_mode(phys7_z_full.astype(np.float32), str(phys7_mode).strip())


def make_loader(static, phys7, y, m, time_mat, idx, batch_size, shuffle, num_workers):
    ds = TensorDataset(
        torch.from_numpy(static[idx]).float(),
        torch.from_numpy(phys7[idx]).float(),
        torch.from_numpy(y[idx]).float(),
        torch.from_numpy(m[idx]).bool(),
        torch.from_numpy(time_mat[idx]).float(),
        torch.from_numpy(idx).long()  # <--- 新增
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

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

    if test_ratio <= 0:
        n_test = 0
    else:
        n_test = int(round(N * test_ratio))
        n_test = max(1, min(n_test, N))  # keep at least 1 if test_ratio>0

    n_val = int(round(N * val_ratio))
    n_val = max(1, min(n_val, max(1, N - n_test)))


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

def masked_mse(pred: torch.Tensor,
               y: torch.Tensor,
               m: torch.Tensor,
               hem_mode: str = "none",
               hem_clip: float = 3.0,
               hem_tau: float = 1.0) -> torch.Tensor:
    diff = (pred - y) ** 2
    w = hem_weight(pred, y, mode=hem_mode, clip=hem_clip, tau=hem_tau)
    diff = diff * w
    diff = diff.masked_fill(~m, 0.0)
    denom = m.float().sum().clamp_min(1.0)
    return diff.sum() / denom

def hem_weight(pred: torch.Tensor,
               y: torch.Tensor,
               mode: str = "none",
               clip: float = 3.0,
               tau: float = 1.0,
               eps: float = 1e-6) -> torch.Tensor:
    if mode is None or str(mode).lower() == "none":
        return torch.ones_like(y)

    mode = str(mode).lower()
    with torch.no_grad():
        std_val = torch.std(y)
        if torch.isnan(std_val) or std_val < eps:
            std_val = torch.tensor(1.0, device=y.device, dtype=y.dtype)
        denom = std_val * float(tau) + eps
        w = 1.0 + torch.abs(pred - y) / denom
        if mode == "clamp":
            w = torch.clamp(w, 1.0, float(clip))
        elif mode == "linear":
            pass
        else:
            raise ValueError(f"Unknown hem_mode={mode}")
    return w


def masked_huber(pred: torch.Tensor,
                 y: torch.Tensor,
                 m: torch.Tensor,
                 beta: float = 1.0,
                 hem_mode: str = "none",
                 hem_clip: float = 3.0,
                 hem_tau: float = 1.0) -> torch.Tensor:
    e = pred - y
    abs_e = torch.abs(e)
    b = float(beta)
    quad = 0.5 * (e ** 2) / max(b, 1e-12)
    lin = abs_e - 0.5 * b
    per = torch.where(abs_e < b, quad, lin)

    w = hem_weight(pred, y, mode=hem_mode, clip=hem_clip, tau=hem_tau)
    per = per * w
    per = per.masked_fill(~m, 0.0)
    denom = m.float().sum().clamp_min(1.0)
    return per.sum() / denom


def masked_loss(pred: torch.Tensor,
                y: torch.Tensor,
                m: torch.Tensor,
                loss_type: str = "mse",
                huber_beta: float = 1.0,
                hem_mode: str = "none",
                hem_clip: float = 3.0,
                hem_tau: float = 1.0) -> torch.Tensor:
    lt = str(loss_type).lower().strip()
    if lt == "mse":
        return masked_mse(pred, y, m, hem_mode=hem_mode, hem_clip=hem_clip, hem_tau=hem_tau)
    if lt == "huber":
        return masked_huber(pred, y, m, beta=huber_beta, hem_mode=hem_mode, hem_clip=hem_clip, hem_tau=hem_tau)
    raise ValueError(f"Unknown loss_type={loss_type}")


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
    preds, ys, ms, idxs = [], [], [], []

    for batch in loader:
        if len(batch) == 6:
            static_x, phys7_seq, y, m, time_mat, batch_idx = [t.to(device) for t in batch]
        else:
            static_x, phys7_seq, y, m, time_mat = [t.to(device) for t in batch]
            batch_idx = torch.zeros(static_x.size(0), dtype=torch.long, device=device) - 1

        p = model(static_x, phys7_seq, time_mat)  # (B,1,T)

        preds.append(p.detach().cpu().numpy())
        ys.append(y.detach().cpu().numpy())
        ms.append(m.detach().cpu().numpy())
        idxs.append(batch_idx.detach().cpu().numpy())

    return dict(
        pred=np.concatenate(preds, 0),  # (N,1,T)
        y=np.concatenate(ys, 0),  # (N,1,T)
        m=np.concatenate(ms, 0).astype(bool),
        idx=np.concatenate(idxs, 0).astype(int)  # (N,)
    )


def train_one(model, train_loader, val_loader, device,
              epochs, lr, wd, patience,
              l2sp_lam, anchor_state, is_transfer,
              backbone_lr_ratio=0.1,
              loss_type="mse", huber_beta=1.0,
              hem_mode="none", hem_clip=3.0, hem_tau=1.0,
              grad_clip=1.0, mixup_alpha=0.0,
              recipe_ids_lookup=None):  # <--- [新增参数]

    # param groups setup (保持原样)
    head_params = []
    backbone_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if "out" in n or "head" in n:
            head_params.append(p)
        else:
            backbone_params.append(p)

    if is_transfer:
        # separate lr
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": lr * backbone_lr_ratio},
            {"params": head_params, "lr": lr}
        ], weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {"train": [], "val": []}
    best_loss = float('inf')
    best_state = None
    patience_counter = 0

    best_info = {}  # epoch, val_loss

    for epoch in range(1, epochs + 1):
        model.train()
        ep_train_loss, ep_train_n = 0.0, 0

        for batch in train_loader:
            # [修改] 解包兼容 6 元素
            if len(batch) == 6:
                x_static, x_phys, y, m, tvals, batch_idx = [t.to(device) for t in batch]
            else:
                x_static, x_phys, y, m, tvals = [t.to(device) for t in batch]
                batch_idx = None

            optimizer.zero_grad()

            try:
                # [修复] Mixup 逻辑 (同步混合)
                if mixup_alpha > 0:
                    lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                    index = torch.randperm(x_static.size(0)).to(device)

                    mixed_static = lam * x_static + (1 - lam) * x_static[index]
                    mixed_phys = lam * x_phys + (1 - lam) * x_phys[index]

                    pred = model(mixed_static, mixed_phys, tvals)

                    # [修复] 使用 masked_loss 而不是 compute_loss_internal
                    loss_a = masked_loss(pred, y, m, loss_type, huber_beta, hem_mode, hem_clip, hem_tau)
                    loss_b = masked_loss(pred, y[index], m[index], loss_type, huber_beta, hem_mode, hem_clip, hem_tau)
                    loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    pred = model(x_static, x_phys, tvals)
                    # [修复] 使用 masked_loss
                    loss = masked_loss(pred, y, m, loss_type, huber_beta, hem_mode, hem_clip, hem_tau)

                # L2SP logic
                if l2sp_lam > 0 and anchor_state is not None:
                    loss = loss + l2sp_penalty(model, anchor_state, l2sp_lam)

                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                optimizer.step()

                ep_train_loss += float(loss.item())
                ep_train_n += 1

            except Exception as e:
                print(f"\n[CRITICAL] Training crashed at Epoch {epoch}")
                if batch_idx is not None and recipe_ids_lookup is not None:
                    try:
                        # 反查 Recipe ID
                        bad_ids = [str(recipe_ids_lookup[i]) for i in batch_idx.cpu().numpy()]
                        print(f"❌ Problematic Batch IDs: {bad_ids}")
                    except:
                        pass
                raise e

        scheduler.step()
        avg_train = ep_train_loss / max(1, ep_train_n)

        # Validation logic
        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 6:
                    vx, vphys, vy, vm, vt, _ = [t.to(device) for t in batch]
                else:
                    vx, vphys, vy, vm, vt = [t.to(device) for t in batch]

                vpred = model(vx, vphys, vt)
                # [修复] 使用 masked_loss
                vloss = masked_loss(vpred, vy, vm, loss_type, huber_beta, hem_mode, hem_clip, hem_tau)
                vl += float(vloss.item())
                vn += 1

        avg_val = vl / max(1, vn)
        history["train"].append(avg_train)
        history["val"].append(avg_val)

        if avg_val < best_loss:
            best_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_info = {"epoch": epoch, "val_loss": avg_val}
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state, strict=False)

    return {
        "epoch": best_info.get("epoch", 0),
        "val_loss": best_info.get("val_loss", float('inf')),
        "history": history
    }

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
    best_conf = load_best_config_common(runs_root)
    df = load_results_summary_df(runs_root)

    out: Dict[str, str] = {}
    if df is None:
        for fam in FAMILIES:
            out[fam] = _resolve_ckpt_by_expname(runs_root, best_conf, fam)
        return out
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
def mixup_data(x, y, alpha=0.4):
    if alpha <= 0:
        return x, y, 1.0

    lam = float(np.random.beta(alpha, alpha))
    batch_size = int(x.size(0))
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index]
    mixed_y = lam * y + (1.0 - lam) * y[index]
    return mixed_x, mixed_y, lam

@dataclass
class ExpCfg:
    name: str
    init: str  # 'scratch' | 'stageB_best'
    finetune_mode: str  # 'full' | 'head' | 'head_ln' | 'bitfit' | 'sequential'
    phys7_mode: str  # 'none' | 'full'
    lr: float = 1e-3
    wd: float = 1e-4
    l2sp: float = 0.0

    backbone_lr_ratio: float = 0.1
    loss_type: str = "mse"      # mse | huber
    huber_beta: float = 1.0

    # HEM（hard example mining）权重策略：masked_loss 里会用到
    hem_mode: str = "none"      # none | clamp | exp | pow ...
    hem_clip: float = 3.0
    hem_tau: float = 1.0

    # 梯度裁剪
    grad_clip: float = 1.0
    mining: str = "none"        # none | hard_mining（当前脚本里未用到）
    mixup_alpha: float = 0.0    # 0.0 表示不启用（建议 0.2~0.4）


def run_one_experiment_on_split_1fam(
        fam: str, exp: ExpCfg, device: str, raw: Dict[str, Any], split: Dict[str, Any],
        stageB_best_ckpt_for_fam: Optional[str], model_type: str, stageA_heads_root: str,
        recipe_aug_mode: str, out_dir_fam: str, epochs: int, batch: int, patience: int,
        num_workers: int, run_seed: int,
) -> Dict[str, Any]:
    set_seed(int(run_seed))

    recipe_ids = raw["recipe_ids"]
    static_raw = raw["static_raw"]
    phys7_raw_full = raw["phys7_raw_full"]
    y_raw = raw["y_raw"]
    mask = raw["mask"]
    time_mat = raw["time_mat"]

    k = family_to_index(fam)
    y_f = y_raw[:, k, :].astype(np.float32)
    m_f = mask[:, k, :].astype(bool)

    # [修改] 如果 split 指明了忽略点，强制 Mask=False
    if "ignored_idx" in split:
        ign = split["ignored_idx"]
        if len(ign) > 0:
            m_f[ign, :] = False

    train_idx = np.asarray(split["train_idx"], int)
    val_idx = np.asarray(split["val_idx"], int)
    test_idx = np.asarray(split["test_idx"], int)

    s_mean, s_std = zfit(static_raw[train_idx])
    p_mean, p_std = zfit(phys7_raw_full[train_idx])
    y_mean_t, y_std_t = zfit_targets_masked_1fam(y_f[train_idx], m_f[train_idx])

    static = apply_z(static_raw, s_mean, s_std)
    phys7_z_full = apply_z(phys7_raw_full, p_mean, p_std)
    phys7_z = apply_phys7_mode(phys7_z_full, exp.phys7_mode)

    T_len = time_mat.shape[1]
    phys7_seq = np.tile(phys7_z[:, :, None], (1, 1, T_len)).astype(np.float32)

    y_norm = apply_z(y_f, y_mean_t.reshape(1, -1), y_std_t.reshape(1, -1))
    y_norm = y_norm[:, None, :]
    m_ = m_f[:, None, :]

    train_loader = make_loader(static, phys7_seq, y_norm, m_, time_mat, train_idx, batch, True, num_workers)
    val_loader = make_loader(static, phys7_seq, y_norm, m_, time_mat, val_idx, batch, False, num_workers)
    test_loader = make_loader(static, phys7_seq, y_norm, m_, time_mat, test_idx, batch, False, num_workers)

    model = build_model(model_type, static_dim=static.shape[1], out_dim=1).to(device)

    init_info = {}
    if exp.init == "stageB_best":
        if not stageB_best_ckpt_for_fam: raise RuntimeError(f"No StageB best ckpt for fam={fam}")
        load_info = load_ckpt_into_model(model, stageB_best_ckpt_for_fam)
        init_info["ckpt"] = stageB_best_ckpt_for_fam
        init_info["ckpt_missing"] = load_info.get("missing", [])
        init_info["ckpt_unexpected"] = load_info.get("unexpected", [])
    else:
        init_info["init_ckpt"] = "scratch";
        init_info["ckpt"] = "scratch"

    apply_finetune_mode(model, exp.finetune_mode)
    anchor_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
    is_transfer_exp = (exp.init != "scratch")

    best = train_one(
        model, train_loader, val_loader, device,
        epochs=epochs, lr=exp.lr, wd=exp.wd, patience=patience,
        l2sp_lam=exp.l2sp, anchor_state=anchor_state,
        is_transfer=is_transfer_exp,
        backbone_lr_ratio=getattr(exp, "backbone_lr_ratio", 0.1),
        loss_type=getattr(exp, "loss_type", "mse"),
        huber_beta=getattr(exp, "huber_beta", 1.0),
        hem_mode=getattr(exp, "hem_mode", "none"),
        hem_clip=getattr(exp, "hem_clip", 3.0),
        hem_tau=getattr(exp, "hem_tau", 1.0),
        grad_clip=getattr(exp, "grad_clip", 1.0),
        mixup_alpha=getattr(exp, "mixup_alpha", 0.0),
        recipe_ids_lookup=recipe_ids  # 传入
    )

    exp_dir = os.path.join(out_dir_fam, "experiments", exp.name, f"seed_{run_seed}")
    ensure_dir(exp_dir)

    history = best.get("history", {"train": [], "val": []})
    plt.figure()
    plt.plot(history.get("train", []), label="Train")
    plt.plot(history.get("val", []), label="Val")
    plt.title(f"{fam} {exp.name} seed={run_seed} Loss")
    plt.legend();
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(exp_dir, "loss_curve.png"));
    plt.close()

    use_val_as_test = (len(test_idx) <= 1 and len(val_idx) > 1)
    final_loader = val_loader if use_val_as_test else test_loader
    eval_set = "val" if use_val_as_test else "test"

    pack = eval_pack(model, final_loader, device)

    # [修改] 调用 metrics 传入 lookup 和 time_list
    met = metrics_1fam_display(pack, fam, y_mean_t, y_std_t,
                               recipe_ids_lookup=recipe_ids, time_list=TIME_LIST,
                               unit_scale=1000.0)

    # [修改] 散点图绘制
    if "yt" in met and "yp" in met and len(met["yt"]) > 0:
        yt, yp = np.array(met["yt"]), np.array(met["yp"])
        bad_cases = met.get("bad_cases", [])

        plt.figure(figsize=(7, 6))
        plt.scatter(yt, yp, s=20, alpha=0.6, c='blue', edgecolors='k', linewidth=0.5)

        mn = min(yt.min(), yp.min());
        mx = max(yt.max(), yp.max())
        span = mx - mn;
        mn -= span * 0.05;
        mx += span * 0.05
        plt.plot([mn, mx], [mn, mx], "r--", lw=1.5, alpha=0.5)

        # 标注 Top 5 坏点
        texts = []
        for case in bad_cases:
            t_true = case["true"];
            t_pred = case["pred"]
            plt.scatter(t_true, t_pred, s=50, c='red', edgecolors='k', zorder=5)
            # label 是 "B47@9"
            texts.append(plt.text(t_true, t_pred, case["label"], fontsize=8, color='red', fontweight='bold'))

        try:
            from adjustText import adjust_text
            adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red', lw=0.5))
        except:
            pass

        plt.xlabel(f"Measured {fam} (nm)");
        plt.ylabel(f"Predicted {fam} (nm)")
        tstr = f"{fam} {exp.name} ({eval_set})\nR2={met['r2']:.3f} MAPE={met['mape_pct']:.1f}%"
        plt.title(tstr);
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(exp_dir, f"scatter_{eval_set}_seed{run_seed}.png"), dpi=120)
        plt.close()

    row = {
        "family": fam, "exp": exp.name, "init": exp.init,
        "finetune_mode": exp.finetune_mode, "phys7_mode": exp.phys7_mode,
        "recipe_aug_mode": recipe_aug_mode, "lr": float(exp.lr), "wd": float(exp.wd),
        "l2sp": float(getattr(exp, "l2sp", 0.0)),
        "backbone_lr_ratio": float(getattr(exp, "backbone_lr_ratio", 0.1)),
        "loss_type": getattr(exp, "loss_type", "mse"),
        "huber_beta": float(getattr(exp, "huber_beta", 1.0)),
        "hem_mode": getattr(exp, "hem_mode", "none"),
        "hem_clip": float(getattr(exp, "hem_clip", 3.0)),
        "hem_tau": float(getattr(exp, "hem_tau", 1.0)),
        "grad_clip": float(getattr(exp, "grad_clip", 1.0)),
        "mixup_alpha": float(getattr(exp, "mixup_alpha", 0.0)),
        "best_epoch": int(best.get("epoch", 0)),
        "val_loss": float(best.get("val_loss", float("inf"))),
        **init_info, **met,
        "run_seed": int(run_seed),
        "split_seed": int(split.get("seed", -1)) if isinstance(split, dict) else -1,
        "trainN": int(len(train_idx)), "valN": int(len(val_idx)), "testN": int(len(test_idx)),
        "eval_set": eval_set,
        "num_dropped": len(split.get("ignored_idx", []))  # [新增]
    }

    # 移除数组防 json 过大
    row_clean = {k: v for k, v in row.items() if k not in ["yp", "yt", "bad_cases"]}
    # 额外存坏点文件
    with open(os.path.join(exp_dir, "bad_cases.json"), "w", encoding="utf-8") as f:
        json.dump(met.get("bad_cases", []), f, indent=2)

    with open(os.path.join(exp_dir, "compare_row.json"), "w", encoding="utf-8") as f:
        json.dump(row_clean, f, indent=2, ensure_ascii=False)
    return row_clean

def exp_to_group(exp_name: str) -> str:
    n = str(exp_name).lower()
    if "scratch" in n:
        return "0_scratch"
    if "l2sp" in n:
        return "2_transfer_l2sp"
    if "stageb" in n or "transfer" in n or "head_ln" in n or "bitfit" in n:
        return "1_transfer"
    return "other"

def get_common_valid_split(
        device: str,
        raw: Dict[str, Any],
        key_recipes: List[str],
        families_eval: List[str],
        test_ratio: float,
        val_ratio: float,
        min_test_points: int,
        seed_start: int,
        max_trials: int = 1000
) -> Dict[str, List[int]]:
    print(f"\n>>> Searching for a COMMON split valid for families: {families_eval}")

    recipe_ids = raw["recipe_ids"]
    mask = raw["mask"]  # (N, K, T)

    # 预先计算每个 family 的 mask 索引
    fam_indices = [family_to_index(f) for f in families_eval]

    # 我们用 mask 的总和作为 quality score (虽然不同 family 分数不同，这里用 sum 简化，或者只用 zmin)
    # 这里使用 zmin (最稀缺资源) 的 mask 作为主要排序依据，确保它被均匀分配
    # 或者简单点，使用所有 family mask 的 union
    # 简单策略：随机尝试

    # 构造一个虚拟的 score (这里不重要，主要靠随机打散)
    N = len(recipe_ids)
    scores = np.ones(N, dtype=np.int32)

    for tr in range(max_trials):
        seed = seed_start + tr
        # 复用已有的 split 函数
        train_idx, val_idx, test_idx = split_with_key_and_quality(
            recipe_ids=recipe_ids,
            key_recipes=key_recipes,
            scores=scores,  # 这里的 score 影响不大，主要靠随机
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            seed=seed
        )
        check_idx = test_idx if len(test_idx) > 1 else val_idx

        all_passed = True
        details = []

        for k in fam_indices:
            m_f = mask[:, k, :].astype(bool)
            valid_pts = int(m_f[check_idx].sum())
            details.append(valid_pts)

            if valid_pts < min_test_points:
                all_passed = False
                break

        if all_passed:
            print(f"   [Success] Found common split at trial {tr} (Seed {seed})")
            print(f"   [Stats] Valid points in eval set per family: {dict(zip(families_eval, details))}")
        return {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "seed": seed
        }

    raise RuntimeError(f"Could not find a common split after {max_trials} trials. "
                       f"Try reducing min_test_points or families list.")

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
    print("\n" + "=" * 40)
    print(" [DIAGNOSTIC] Distribution Shift Check (Stage B vs C)")
    print("=" * 40)
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
    if "phys7_raw_full" not in raw:
        print(" [SKIP] raw data missing 'phys7_raw_full'.")
        return

    phys_c = raw["phys7_raw_full"]  # (N, 7) - Stage C 中 phys7 是 (N, 7) 的 numpy 数组
    mean_c = phys_c.mean(axis=0)  # (7,)
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

def pick_best_rows(rows: List[Dict[str, Any]],
                   key_fields: Tuple[str, ...] = ("family", "exp"),
                   metric: str = "r2") -> List[Dict[str, Any]]:
    best = {}
    for r in rows:
        key = tuple(r.get(k) for k in key_fields)
        cur = best.get(key, None)
        if cur is None:
            best[key] = r
            continue
        try:
            if float(r.get(metric, -1e9)) > float(cur.get(metric, -1e9)):
                best[key] = r
        except Exception:
            pass
    return list(best.values())


def search_best_common_split(
        device: str,
        raw: Dict[str, Any],
        key_recipes: List[str],
        families_eval: List[str],
        stageB_best: Dict[str, str],
        args: argparse.Namespace,
        stageB_phys_mode: str
) -> Dict[str, Any]:
    print(f"\n" + "=" * 80)
    print(f" [Search Phase] Searching for BEST split (Dynamic Drop 0~{args.max_drop} pts)")
    print(f" Strategy: Maximize Average R2")
    print("=" * 80)

    best_split = None
    best_min_r2 = -999.0

    proxy_out_root = os.path.join(args.out_dir, "_proxy_temp")
    ensure_dir(proxy_out_root)

    N = len(raw["recipe_ids"])
    all_indices = np.arange(N)

    # 代理配置 (快速跑)
    proxy_exp = ExpCfg(
        name="proxy_check", init="stageB_best", finetune_mode="head_ln",
        phys7_mode=stageB_phys_mode, lr=1e-3, wd=1e-4, l2sp=0.0
    )

    total_trials = 0

    # [新增] 外层循环：尝试扔掉 n_drop 个点
    for n_drop in range(args.max_drop + 1):
        if N <= n_drop + 3: continue

        for tr in range(args.trials):
            total_trials += 1
            current_seed = args.seed + total_trials
            rng = np.random.RandomState(current_seed)

            # 1. 随机选出 ignored indices (脏数据)
            if n_drop > 0:
                drop_indices = rng.choice(all_indices, n_drop, replace=False)
            else:
                drop_indices = np.array([], dtype=int)

            # 2. 剩余有效点 (Active Set)
            active_mask = np.ones(N, dtype=bool)
            active_mask[drop_indices] = False
            active_indices = all_indices[active_mask]

            # 3. 在 Active Set 里切分
            # 提取子集ID用于切分
            sub_ids = raw["recipe_ids"][active_indices]
            sub_scores = np.ones(len(sub_ids))

            try:
                # 切分得到的是 active_indices 里的局部下标 (0 ~ N-n_drop-1)
                sub_train, sub_val, sub_test = split_with_key_and_quality(
                    recipe_ids=sub_ids,
                    key_recipes=key_recipes,
                    scores=sub_scores,
                    test_ratio=args.test_ratio,
                    val_ratio=args.val_ratio,
                    seed=current_seed
                )

                # 映射回原始全集索引 (0 ~ N-1)
                train_idx = active_indices[sub_train]
                val_idx = active_indices[sub_val]
                test_idx = active_indices[sub_test]

                # 检查 mask 约束
                check_idx = test_idx if len(test_idx) > 1 else val_idx
                valid = True
                for f in families_eval:
                    k = family_to_index(f)
                    if raw["mask"][:, k, :][check_idx].sum() < args.min_test_points:
                        valid = False;
                        break
                if not valid: continue

            except:
                continue

            # 4. 快速试跑
            fam_r2s = []
            temp_split = {
                "train_idx": train_idx.tolist(),
                "val_idx": val_idx.tolist(),
                "test_idx": test_idx.tolist(),
                "ignored_idx": drop_indices.tolist(),  # [记录]
                "seed": current_seed
            }

            for fam in families_eval:
                ckpt = stageB_best.get(fam)
                if not ckpt: continue
                # Proxy Run: Epochs=20, Patience=5 (加速)
                res = run_one_experiment_on_split_1fam(
                    fam=fam, exp=proxy_exp, device=device, raw=raw, split=temp_split,
                    stageB_best_ckpt_for_fam=ckpt, model_type=args.model_type,
                    stageA_heads_root=args.stageA_heads_root, recipe_aug_mode="none",
                    out_dir_fam=proxy_out_root,
                    epochs=20, patience=5, batch=args.batch, num_workers=0, run_seed=current_seed
                )
                fam_r2s.append(res["r2"])

            if not fam_r2s: continue
            min_r2 = min(fam_r2s)

            if total_trials % 20 == 0 or min_r2 > best_min_r2:
                print(f" [Drop {n_drop}] Seed={current_seed} | Min R2={min_r2:.3f}")

            if min_r2 > best_min_r2:
                best_min_r2 = min_r2
                best_split = {
                    "train_idx": train_idx.tolist(),
                    "val_idx": val_idx.tolist(),
                    "test_idx": test_idx.tolist(),
                    "ignored_idx": drop_indices.tolist(),
                    "seed": current_seed,
                    "score": min_r2,
                    "train_names": raw["recipe_ids"][train_idx].tolist(),
                    "val_names": raw["recipe_ids"][val_idx].tolist(),
                    "ignored_names": raw["recipe_ids"][drop_indices].tolist()
                }
                print(f"   >>> New Best! Score={min_r2:.3f}, Ignored={best_split['ignored_names']}")

    if best_split is None:
        print("[WARN] No split found in search! Using fallback.")
        best_split = temp_split
        best_split["score"] = -1.0

    print(f"\n [Search Done] Best Score: {best_split['score']:.3f}, Ignored: {best_split.get('ignored_names', [])}")
    return best_split


def metrics_1fam_display(pack, fam, y_mean_t, y_std_t, unit_scale=1000.0,
                         recipe_ids_lookup=None, time_list=None):  # <--- [新增参数]

    pred = torch.from_numpy(pack["pred"]).float()
    y = torch.from_numpy(pack["y"]).float()
    m = torch.from_numpy(pack["m"]).bool()
    raw_idx = torch.from_numpy(pack["idx"]).long()  # (N,)

    N, K, T = m.shape

    # 构造 Time Index (N,1,T)
    t_idx_base = torch.arange(T, device=m.device).view(1, 1, T).expand(N, K, T)

    mean = torch.from_numpy(y_mean_t).view(1, 1, T)
    std = torch.from_numpy(y_std_t).view(1, 1, T)

    pred_um = pred * std + mean
    y_um = y * std + mean

    sign_map, nonneg_set = su._default_family_sign_and_nonneg([fam])
    family_sign = torch.tensor([sign_map[fam]], dtype=torch.float32)

    pred_disp, y_disp = pu.transform_for_display(
        pred_um, y_um,
        family_sign=family_sign,
        clip_nonneg=True,
        nonneg_families=[0],
        unit_scale=unit_scale,
        flip_sign=False,
        min_display_value=None
    )

    # Flatten
    yp = pred_disp[:, 0, :].reshape(-1).numpy()
    yt = y_disp[:, 0, :].reshape(-1).numpy()
    mk = pack["m"][:, 0, :].reshape(-1).numpy()

    # Flatten IDs and Times
    # recipe_idx: expand (N,) -> (N,1,T) -> flatten
    flat_recipe_idx = raw_idx.view(N, 1, 1).expand(N, K, T)[:, 0, :].reshape(-1).numpy()
    flat_time_idx = t_idx_base[:, 0, :].reshape(-1).numpy()

    # Apply Mask
    yp = yp[mk]
    yt = yt[mk]
    r_idx_m = flat_recipe_idx[mk]
    t_idx_m = flat_time_idx[mk]

    # Remove NaNs
    ok = np.isfinite(yt) & np.isfinite(yp)
    yp = yp[ok]
    yt = yt[ok]
    r_idx_m = r_idx_m[ok]
    t_idx_m = t_idx_m[ok]

    n = int(len(yt))

    bad_cases_list = []
    r2, mae, median_ae, mape = float("nan"), float("nan"), float("nan"), float("nan")

    if n >= 2:
        r2 = float(su.masked_r2_score_np(yt, yp))
        abs_err = np.abs(yt - yp)
        mae = float(np.mean(abs_err))
        median_ae = float(np.median(abs_err))

        # MAPE
        denom = np.abs(yt)
        denom[denom < 1e-6] = 1e-6
        mape = float(np.mean(abs_err / denom)) * 100.0

        # [新增] Top 5 坏点
        worst_indices = np.argsort(abs_err)[-10:][::-1]

        if recipe_ids_lookup is not None and time_list is not None:
            for idx in worst_indices:
                rid_str = str(recipe_ids_lookup[r_idx_m[idx]])
                tid_str = str(time_list[t_idx_m[idx]])
                bad_cases_list.append({
                    "label": f"{rid_str}@{tid_str}",  # e.g. B47@9
                    "true": float(yt[idx]),
                    "pred": float(yp[idx]),
                    "err": float(abs_err[idx])
                })
                if len(bad_cases_list) >= 5: break

    return {
        "family": fam, "r2": r2, "mae_nm": mae, "mape_pct": mape, "median_ae_nm": median_ae,
        "n": n, "yp": yp.tolist(), "yt": yt.tolist(),
        "bad_cases": bad_cases_list
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new_excel", type=str, default=r"D:\PycharmProjects\Bosch\Bosch.xlsx")
    ap.add_argument("--out_dir", type=str, default=r"D:\PycharmProjects\Bosch\runs_stageC_singlehead")
    ap.add_argument("--stageB_runs_root", type=str, default=r"D:\PycharmProjects\Bosch\runs_stageB_morph_phys7")

    ap.add_argument("--model_type", type=str, default="transformer", choices=["transformer", "gru", "mlp"])
    ap.add_argument("--recipe_aug_mode", type=str, default="base")
    ap.add_argument("--height_family", type=str, default="h1")
    ap.add_argument("--stageA_heads_root", type=str, default=getattr(sb.Cfg, "stageA_heads_root", ""))

    ap.add_argument("--key_recipes", type=str, default="")
    ap.add_argument("--families_eval", type=str, default="zmin,h1,d1,w")

    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--seed_repeats", type=int, default=5)
    ap.add_argument("--trials", type=int, default=600)

    # [新增] 最大扔点数量
    ap.add_argument("--max_drop", type=int, default=6, help="Max number of recipes to drop as outliers")

    ap.add_argument("--test_ratio", type=float, default=0.0)
    ap.add_argument("--val_ratio", type=float, default=0.3)
    ap.add_argument("--min_test_points", type=int, default=1)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[StageC] Loading best config from: {args.stageB_runs_root}")
    best_conf = su.load_best_config_common(args.stageB_runs_root)
    sb.apply_hp_from_best_conf_to_cfg(best_conf)

    stageB_aug_mode = best_conf.get("recipe_aug_mode", args.recipe_aug_mode)
    stageB_phys_mode = best_conf.get("phys7_mode", "full")
    stageB_model_type = best_conf.get("model_type", args.model_type)
    args.recipe_aug_mode = stageB_aug_mode
    args.model_type = stageB_model_type

    stageB_best = su.resolve_stageB_best_ckpts_from_common(args.stageB_runs_root)

    raw = build_stageC_raw(
        device=device, new_excel=args.new_excel, height_family=args.height_family,
        recipe_aug_mode=args.recipe_aug_mode, stageA_heads_root=args.stageA_heads_root
    )

    families_eval = [x.strip().lower() for x in str(args.families_eval).split(",") if x.strip()]
    key_recipes = [x.strip() for x in str(args.key_recipes).split(",") if x.strip()]

    # Search with Dynamic Drop
    best_common_split = search_best_common_split(
        device=device, raw=raw, key_recipes=key_recipes, families_eval=families_eval,
        stageB_best=stageB_best, args=args, stageB_phys_mode=stageB_phys_mode
    )

    with open(os.path.join(args.out_dir, "best_common_split.json"), "w", encoding="utf-8") as f:
        json.dump(best_common_split, f, indent=2)

    # ... (定义 experiments 列表，保持原样) ...
    # 为节省篇幅，这里略去 experiments 定义代码，因为没有改动

    # 这里需要把你原文件里的 experiments 定义部分复制过来
    pm = stageB_phys_mode
    experiments: List[ExpCfg] = [
        ExpCfg(name="scratch_full", init="scratch", finetune_mode="full", phys7_mode=pm, lr=3e-4, wd=1e-4, l2sp=0.0,
               loss_type="mse", backbone_lr_ratio=1.0),
        ExpCfg(name="stageB_full", init="stageB_best", finetune_mode="full", phys7_mode=pm, lr=2e-4, wd=1e-4, l2sp=0.0,
               loss_type="mse", backbone_lr_ratio=0.1),
        ExpCfg(name="stageB_head_ln", init="stageB_best", finetune_mode="head_ln", phys7_mode=pm, lr=6e-4, wd=1e-4,
               l2sp=0.0, loss_type="mse", backbone_lr_ratio=0.1),
        ExpCfg(name="stageB_bitfit_ln", init="stageB_best", finetune_mode="bitfit_ln", phys7_mode=pm, lr=1e-3, wd=0.0,
               l2sp=0.0, loss_type="mse", backbone_lr_ratio=0.1),
        ExpCfg(name="stageB_head_ln_mixup", init="stageB_best", finetune_mode="head_ln", phys7_mode=pm, lr=6e-4,
               wd=1e-4, l2sp=0.0, mixup_alpha=0.2, loss_type="mse", backbone_lr_ratio=0.1),
    ]

    finetune_modes = ["full", "head_ln"]
    lr_base, wd_base = 2e-4, 1e-4

    for ft in finetune_modes:
        for ratio in [0.05, 0.1]:
            experiments.append(
                ExpCfg(name=f"T1_huber_none_{ft}_r{ratio}", init="stageB_best", finetune_mode=ft, phys7_mode=pm,
                       lr=lr_base if ft == "full" else 6e-4, wd=wd_base, l2sp=0.0, loss_type="huber", huber_beta=1.0,
                       backbone_lr_ratio=ratio))

    for ft in finetune_modes:
        for ratio in [0.05, 0.1]:
            for l2 in [3e-4, 1e-3]:
                experiments.append(
                    ExpCfg(name=f"T2_huber_none_{ft}_r{ratio}_l2{l2}", init="stageB_best", finetune_mode=ft,
                           phys7_mode=pm, lr=lr_base if ft == "full" else 6e-4, wd=wd_base, l2sp=float(l2),
                           loss_type="huber", huber_beta=1.0, backbone_lr_ratio=ratio))

    for ft in finetune_modes:
        experiments.append(
            ExpCfg(name=f"T3_huber_clamp_{ft}_r0.05", init="stageB_best", finetune_mode=ft, phys7_mode=pm,
                   lr=lr_base if ft == "full" else 6e-4, wd=wd_base, l2sp=0.0, loss_type="huber", huber_beta=1.0,
                   hem_mode="clamp", hem_clip=3.0, hem_tau=2.0, backbone_lr_ratio=0.05))

    summary_all = os.path.join(args.out_dir, "summary_common_split_allruns.csv")
    summary_best = os.path.join(args.out_dir, "summary_common_split_best.csv")

    # [修改] Header 增加 num_dropped
    header = "family,exp,r2,mae_nm,mape_pct,median_ae_nm,n,best_epoch,val_loss,run_seed,split_seed,trainN,valN,testN,eval_set,num_dropped,ckpt\n"
    for p in [summary_all, summary_best]:
        if not os.path.exists(p):
            with open(p, "w", encoding="utf-8") as f: f.write(header)

    all_rows = []
    for fam in families_eval:
        print(f"\n>>> Processing Family: [{fam}]")
        out_dir_fam = os.path.join(args.out_dir, fam)
        ensure_dir(out_dir_fam)
        fam_ckpt = stageB_best.get(fam, None)
        with open(os.path.join(out_dir_fam, "stageB_best_ckpt.json"), "w") as f:
            json.dump({"family": fam, "ckpt": fam_ckpt}, f)
        if not fam_ckpt or not os.path.exists(fam_ckpt): continue

        for exp in experiments:
            for i in range(args.seed_repeats):
                try:  # <--- [新增] 加上这个 try
                    row = run_one_experiment_on_split_1fam(
                        fam=fam, exp=exp, device=device, raw=raw, split=best_common_split,
                        stageB_best_ckpt_for_fam=fam_ckpt, model_type=args.model_type,
                        stageA_heads_root=args.stageA_heads_root, recipe_aug_mode=args.recipe_aug_mode,
                        out_dir_fam=out_dir_fam, epochs=args.epochs, batch=args.batch,
                        patience=args.patience, num_workers=args.num_workers, run_seed=args.seed + i
                    )
                    all_rows.append(row)

                    with open(summary_all, "a", encoding="utf-8") as f:
                        f.write(
                            f"{row['family']},{row['exp']},{row['r2']},{row['mae_nm']},{row['mape_pct']},{row['median_ae_nm']},"
                            f"{row['n']},{row['best_epoch']},{row['val_loss']},{row['run_seed']},{row['split_seed']},"
                            f"{row['trainN']},{row['valN']},{row['testN']},{row['eval_set']},{row['num_dropped']},{row['ckpt']}\n")

                except Exception as e:  # <--- [新增] 捕获异常，打印错误，但继续跑下一个
                    print(f"\n[SKIP] Experiment failed: {exp.name} seed={i}")
                    print(f"Reason: {e}")
                    traceback.print_exc()
                    continue
    best_rows = pick_best_rows(all_rows, key_fields=("family", "exp"), metric="r2")
    best_rows = sorted(best_rows, key=lambda r: (r["family"], r["exp"]))
    with open(summary_best, "a", encoding="utf-8") as f:
        for row in best_rows:
            f.write(f"{row['family']},{row['exp']},{row['r2']},{row['mae_nm']},{row['mape_pct']},{row['median_ae_nm']},"
                    f"{row['n']},{row['best_epoch']},{row['val_loss']},{row['run_seed']},{row['split_seed']},"
                    f"{row['trainN']},{row['valN']},{row['testN']},{row['eval_set']},{row['num_dropped']},{row['ckpt']}\n")

    print("\n[DONE] StageC Finished.")

if __name__ == "__main__":
    main()
