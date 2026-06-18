# eval_retrieval_tsne.py (patched: optional DEDUP + TRACE PRINTS)
from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

try:
    import umap  # type: ignore
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False


# =========================
# Config
# =========================
@dataclass
class RetrievalTSNEConfig:
    tsne_perplexity: float = 30.0
    tsne_random_state: int = 42
    umap_random_state: int = 42
    save_npz: bool = True

    # batch control
    # - int: use first N batches
    # - "all": use all batches
    # If int is larger than available, iteration naturally ends => "all".
    align1_batches: Union[int, str] = 1
    align2_batches: Union[int, str] = 1

    # Optional: keep cross_inputs or not
    run_cross_inputs: bool = True

    # NEW: token dedup before plotting/retrieval
    # Dedup key = (dataset_index, timestep)
    # Keep first seen (earlier batch / earlier sample / earlier timestep)
    enable_dedup: bool = True


def _mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def _compute_retrieval_metrics(sim: np.ndarray) -> Dict[str, float]:
    """
    sim: [K, K], sim[i,j] is score query i vs target j
    GT is diagonal i==j (same token order after flatten)
    """
    k = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    pos = np.zeros(k, dtype=np.int32)
    for i in range(k):
        pos[i] = int(np.where(order[i] == i)[0][0])
    r1 = float(np.mean(pos == 0))
    mrr = float(np.mean(1.0 / (pos + 1)))
    return {"R@1": r1, "MRR": mrr}


def _movement_labels_from_x(x_: torch.Tensor) -> torch.Tensor:
    """
    x_: [B, T, C] normalized lookback window
    Return movement labels per timestep for each sample: [B, T] in {-1, +1}
    Rule:
      movement[t] = sign(mean(x[t]) - mean(x[t-1]))
      movement[0] = +1
      sign==0 -> +1
    """
    B, T, C = x_.shape
    y = x_.mean(dim=-1)  # [B,T]
    labels = torch.ones((B, T), dtype=torch.long, device=x_.device)  # init +1
    if T >= 2:
        diff = y[:, 1:] - y[:, :-1]          # [B,T-1]
        s = torch.sign(diff)                 # {-1,0,+1}
        s = torch.where(s == 0, torch.ones_like(s), s)
        labels[:, 1:] = s.to(torch.long)
    return labels


def _normalize_tokens_td(x_td: torch.Tensor) -> np.ndarray:
    """
    x_td: [K, D]
    return [K, D] numpy L2-normalized per token
    """
    x_td = F.normalize(x_td, dim=1)
    return x_td.detach().cpu().numpy()


def _align_embed_dim_for_retrieval(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    a: [K,Da], b:[K,Db] -> truncate dim to d=min(Da,Db) for dot sim
    """
    d = min(a.shape[1], b.shape[1])
    return a[:, :d], b[:, :d]


def _safe_tsne_perplexity(requested: float, n_samples: int) -> float:
    if n_samples <= 1:
        return 1.0
    p = float(requested)
    p = max(1.0, min(p, float(n_samples - 1)))
    upper = max(1.0, (n_samples - 1) / 3.0)
    p = min(p, upper)
    return float(p)


def _run_tsne_2d(X: np.ndarray, cfg: RetrievalTSNEConfig) -> np.ndarray:
    n = int(X.shape[0])
    perplexity = _safe_tsne_perplexity(cfg.tsne_perplexity, n)
    tsne = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        random_state=int(cfg.tsne_random_state),
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(X)


def _run_umap_2d(X: np.ndarray, cfg: RetrievalTSNEConfig) -> Optional[np.ndarray]:
    if not _HAS_UMAP:
        return None
    reducer = umap.UMAP(
        n_components=2,
        random_state=int(cfg.umap_random_state),
    )
    return reducer.fit_transform(X)


def _scatter_time_2d(Z2: np.ndarray, t: np.ndarray, out_png: str, title: str):
    """
    Time coloring with a colorbar that explicitly indicates timestep 1..T.
    """
    plt.figure()
    sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=t, s=10)
    plt.title(title)
    cb = plt.colorbar(sc)
    cb.set_label("timestep (1..T)")
    if t.size >= 2:
        cb.set_ticks([t.min(), t.max()])
        cb.set_ticklabels([str(int(t.min())), str(int(t.max()))])
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def _scatter_movement_2d(Z2: np.ndarray, mv: np.ndarray, out_png: str, title: str):
    """
    Movement coloring with explicit legend: -1 / +1
    mv must be in {-1, +1}
    """
    plt.figure()
    mv = mv.astype(np.int64)
    m1 = (mv == -1)
    p1 = (mv == 1)

    if m1.any():
        plt.scatter(Z2[m1, 0], Z2[m1, 1], s=10, label="movement=-1")
    if p1.any():
        plt.scatter(Z2[p1, 0], Z2[p1, 1], s=10, label="movement=+1")

    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


# =========================
# Text embedding (executor-side)
# =========================
@torch.no_grad()
def _prepare_text_emb(exp, dataset, index, target_len: int) -> torch.Tensor:
    """
    returns z_text: [B, target_len, Dtxt]
    """
    hist_text = None
    if hasattr(dataset, "get_text_pair"):
        try:
            hist_text, _ = dataset.get_text_pair(index)
        except Exception:
            hist_text = None
    if hist_text is None:
        tmp = dataset.get_text(index)
        try:
            hist_text = tmp[:, 1]
        except Exception:
            hist_text = tmp

    if getattr(exp, "use_bertopic_finbert", False):
        txt = exp._encode_with_bertopic_finbert(hist_text)
    else:
        txt = exp._encode_text_list(hist_text)

    txt = exp.mlp(txt)  # [B, T?, Dtxt]
    txt = exp._time_align_text_emb(txt, target_len=target_len)  # -> [B, target_len, Dtxt]
    return txt


def _eval_linear_dropout(in_f: int, out_f: int, p: float, device, dtype) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(in_f), int(out_f), bias=True),
        nn.Dropout(float(p))
    ).to(device=device, dtype=dtype)


def _get_layer_linear_in_out(m) -> Optional[Tuple[int, int]]:
    """
    Support nn.Linear or nn.Sequential(Linear, Dropout)
    """
    if m is None:
        return None
    if isinstance(m, nn.Linear):
        return int(m.in_features), int(m.out_features)
    if isinstance(m, nn.Sequential) and len(m) >= 1 and isinstance(m[0], nn.Linear):
        return int(m[0].in_features), int(m[0].out_features)
    return None


# =========================
# align1 token extraction (STRICTLY mimic your model's transpose chain)
# =========================
@torch.no_grad()
def _extract_align1_tokens(
    exp, model, batch_x, batch_x_mark, index, dataset
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Return:
      ts_tokens : [B, L, Dtxt]
      txt_tokens: [B, L, Dtxt]
      movement  : [B, L] in {-1,+1}
    """
    device = exp.device
    model.eval()
    exp.mlp.eval()

    x_enc = batch_x.float().to(device)     # [B, W, N]
    x_mark = batch_x_mark.float().to(device)

    means = x_enc.mean(1, keepdim=True).detach()
    x_ = x_enc - means
    stdev = torch.sqrt(torch.var(x_, dim=1, keepdim=True, unbiased=False) + 1e-5)
    x_ = x_ / stdev                         # [B, W, N]

    enc = model.enc_embedding(x_, x_mark)        # [B, N, d_model]
    enc, _ = model.encoder(enc, attn_mask=None)  # [B, N, d_model]
    enc_out = F.normalize(enc, dim=-1)           # [B, N, d_model]

    B = int(enc_out.shape[0])
    N = int(enc_out.shape[1])
    d_model = int(enc_out.shape[2])

    L = int(getattr(model, "seq_len", None) or getattr(getattr(model, "configs", object()), "seq_len", 40))

    z_text = _prepare_text_emb(exp, dataset, index, target_len=L).to(device)  # [B, L, Dtxt]
    B2, L2, Dtxt = z_text.shape # 32,40,12
    assert B2 == B and L2 == L, f"text align mismatch: got {z_text.shape}, expected [B={B}, L={L}, Dtxt]"

    p = float(getattr(getattr(model, "configs", object()), "dropout_DA", 0.0)) if hasattr(model, "configs") else 0.0

    proj_ts1 = getattr(model, "_proj_ts1", None)
    proj_ts2 = getattr(model, "_proj_ts2", None)

    used1 = "eval_proj_ts1"
    used2 = "eval_proj_ts2"

    if proj_ts1 is not None:
        io = _get_layer_linear_in_out(proj_ts1)
        if io is not None and io[0] == N and io[1] == Dtxt:
            proj_ts1 = proj_ts1.to(device)
            used1 = "checkpoint:_proj_ts1"
        else:
            proj_ts1 = None
    if proj_ts1 is None:
        proj_ts1 = _eval_linear_dropout(N, Dtxt, p, device=device, dtype=enc_out.dtype)
        proj_ts1.eval()

    if proj_ts2 is not None:
        io = _get_layer_linear_in_out(proj_ts2)
        if io is not None and io[0] == d_model and io[1] == L:
            proj_ts2 = proj_ts2.to(device)
            used2 = "checkpoint:_proj_ts2"
        else:
            proj_ts2 = None
    if proj_ts2 is None:
        proj_ts2 = _eval_linear_dropout(d_model, L, p, device=device, dtype=enc_out.dtype)
        proj_ts2.eval()

    tmp = proj_ts1(enc_out.transpose(1, 2))                 # [B, d_model, Dtxt]
    h_ts_align1 = proj_ts2(tmp.transpose(1, 2)).transpose(1, 2)  # [B, L, Dtxt]
    h_txt_align1 = z_text                                   # [B, L, Dtxt]

    mv_full = _movement_labels_from_x(x_)                    # [B, W]
    if mv_full.shape[1] != L:
        if mv_full.shape[1] > L:
            mv = mv_full[:, :L]
        else:
            pad = torch.ones((B, L - mv_full.shape[1]), dtype=mv_full.dtype, device=mv_full.device)
            mv = torch.cat([mv_full, pad], dim=1)
    else:
        mv = mv_full

    diag = {
        "align1_used_proj_ts1": used1,
        "align1_used_proj_ts2": used2,
        "B": int(B),
        "N_vars": int(N),
        "L_time": int(L),
        "d_model": int(d_model),
        "d_txt": int(Dtxt),
        "x_time_len_W": int(x_.shape[1]),
        "mv_shape": [int(mv.shape[0]), int(mv.shape[1])],
        "ts_shape": [int(h_ts_align1.shape[0]), int(h_ts_align1.shape[1]), int(h_ts_align1.shape[2])],
        "txt_shape": [int(h_txt_align1.shape[0]), int(h_txt_align1.shape[1]), int(h_txt_align1.shape[2])],
    }
    return h_ts_align1, h_txt_align1, mv, diag


# =========================
# align2 token extraction
# =========================
@torch.no_grad()
def _extract_align2_tokens(
    exp, model, batch_x, index, dataset
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return:
      ts_tokens:  [B,T,align_dim]
      txt_tokens: [B,T,align_dim]
      movement:   [B,T] in {-1,+1}
    """
    if not hasattr(model, "shared_proj2_time"):
        raise RuntimeError("model.shared_proj2_time not found. Please ensure Model registers it in __init__.")

    device = exp.device
    model.eval()
    exp.mlp.eval()

    x_enc = batch_x.float().to(device)  # [B,T,C]

    means = x_enc.mean(1, keepdim=True).detach()
    x_ = x_enc - means
    stdev = torch.sqrt(torch.var(x_, dim=1, keepdim=True, unbiased=False) + 1e-5)
    x_ = x_ / stdev

    T = int(x_.shape[1])
    z_text = _prepare_text_emb(exp, dataset, index, target_len=T).to(device)  # [B,T,Dtxt]

    model.shared_proj2_time.to(device)
    h_ts_t2, h_txt_t2 = model.shared_proj2_time(x_, z_text)  # [B,T,align_dim]

    mv = _movement_labels_from_x(x_)  # [B,T] in {-1,+1}
    return h_ts_t2, h_txt_t2, mv


# =========================
# Token collectors
# =========================
def _collect_tokens_no_dedup(
    ts_btd: torch.Tensor,   # [B,T,D]
    txt_btd: torch.Tensor,  # [B,T,D]
    mv_bt: torch.Tensor,    # [B,T]
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    NO DEDUP. Flatten all tokens.
    """
    B, T, D = ts_btd.shape
    K = int(B * T)

    ts_kd = ts_btd.reshape(K, D)
    txt_kd = txt_btd.reshape(K, D)

    t_one = np.arange(1, T + 1, dtype=np.float32)
    t_k = np.tile(t_one, B)  # [B*T]

    mv_k = mv_bt.reshape(K).detach().cpu().numpy().astype(np.int64)
    return ts_kd, txt_kd, t_k, mv_k


def _collect_tokens_with_dedup(
    ts_btd: torch.Tensor,          # [B,T,D]
    txt_btd: torch.Tensor,         # [B,T,D]
    mv_bt: torch.Tensor,           # [B,T]
    index_any,                     # dataset indices from loader
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    DEDUP key: (dataset_index, timestep)
    Keep first seen.
    """
    if torch.is_tensor(index_any):
        idx_list = [int(x) for x in index_any.detach().cpu().tolist()]
    else:
        idx_list = [int(x) for x in index_any]

    B, T, D = ts_btd.shape
    store: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor, int, int]] = {}

    # order is: batch iteration order -> sample order -> time order
    for bi in range(B):
        sid = idx_list[bi] if bi < len(idx_list) else bi
        for ti in range(T):
            key = (sid, int(ti))
            if key in store:
                continue
            store[key] = (
                ts_btd[bi, ti].detach(),
                txt_btd[bi, ti].detach(),
                int(mv_bt[bi, ti].item()),
                int(ti + 1),
            )

    keys = sorted(store.keys(), key=lambda x: (x[0], x[1]))
    ts_list, txt_list, mv_list, t_list = [], [], [], []
    for k in keys:
        ts_v, txt_v, mv_v, t1 = store[k]
        ts_list.append(ts_v.unsqueeze(0))
        txt_list.append(txt_v.unsqueeze(0))
        mv_list.append(mv_v)
        t_list.append(t1)

    ts_kd = torch.cat(ts_list, dim=0) if ts_list else ts_btd.reshape(0, D)
    txt_kd = torch.cat(txt_list, dim=0) if txt_list else txt_btd.reshape(0, D)
    t_k = np.asarray(t_list, dtype=np.float32)
    mv_k = np.asarray(mv_list, dtype=np.int64)
    return ts_kd, txt_kd, t_k, mv_k


def _iter_n_batches(loader, n_or_all: Union[int, str]):
    if isinstance(n_or_all, str) and n_or_all.lower() == "all":
        for b in loader:
            yield b
        return

    n = int(n_or_all)
    if n <= 0:
        n = 1
    for i, b in enumerate(loader):
        if i >= n:
            break
        yield b


# =========================
# One-space plotting + retrieval
# =========================
def _save_space(
    out_dir: str,
    tag: str,                # IMPORTANT: stable file prefix, e.g. "align1"/"align2"
    ts_kd: torch.Tensor,
    txt_kd: torch.Tensor,
    t_k: np.ndarray,
    mv_k: np.ndarray,
    cfg: RetrievalTSNEConfig,
    *,
    color_by: str,           # "time" or "movement"
    title_prefix: str,       # ONLY for plot titles (NOT for filenames)
):
    """
    Save retrieval json + t-SNE/UMAP plots + optional npz.

    IMPORTANT (compat with eval_suite.py gallery):
      - Filenames MUST be stable:
          tsne_{tag}_{TS/TXT}_{time/movement}.png
          umap_{tag}_{TS/TXT}_{time/movement}.png
      - title_prefix is used ONLY in figure titles.
    """
    _mkdir(out_dir)

    print(f"[retrieval_tsne][{tag}] pre-trim shapes: "
          f"ts={tuple(ts_kd.shape)} txt={tuple(txt_kd.shape)} "
          f"t={int(t_k.shape[0])} mv={int(mv_k.shape[0])}")

    K = min(ts_kd.shape[0], txt_kd.shape[0], int(t_k.shape[0]), int(mv_k.shape[0]))
    print(f"[retrieval_tsne][{tag}] K chosen (after min) = {K}")

    ts_kd = ts_kd[:K]
    txt_kd = txt_kd[:K]
    t_k = t_k[:K]
    mv_k = mv_k[:K]

    # normalize for retrieval + visualization
    ts_np = _normalize_tokens_td(ts_kd)
    txt_np = _normalize_tokens_td(txt_kd)

    # retrieval metrics
    A, B = _align_embed_dim_for_retrieval(ts_np, txt_np)
    sim_ts2txt = A @ B.T
    sim_txt2ts = B @ A.T

    retrieval = {
        "K": int(K),
        "dim_ts": int(ts_np.shape[1]),
        "dim_txt": int(txt_np.shape[1]),
        "dim_used_for_sim": int(A.shape[1]),
        "TS->Text": _compute_retrieval_metrics(sim_ts2txt),
        "Text->TS": _compute_retrieval_metrics(sim_txt2ts),
    }
    with open(os.path.join(out_dir, f"retrieval_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(retrieval, f, indent=2, ensure_ascii=False)

    # embeddings -> 2D
    Z_ts = _run_tsne_2d(ts_np, cfg)
    Z_txt = _run_tsne_2d(txt_np, cfg)

    Zu_ts = _run_umap_2d(ts_np, cfg)
    Zu_txt = _run_umap_2d(txt_np, cfg)

    # IMPORTANT: filenames use stable prefix = tag (align1/align2),
    # so eval_suite gallery can find them.
    file_prefix = tag

    if color_by == "time":
        _scatter_time_2d(
            Z_ts, t_k,
            os.path.join(out_dir, f"tsne_{file_prefix}_TS_time.png"),
            f"{title_prefix} TS (t-SNE; color=time)"
        )
        _scatter_time_2d(
            Z_txt, t_k,
            os.path.join(out_dir, f"tsne_{file_prefix}_TXT_time.png"),
            f"{title_prefix} TXT (t-SNE; color=time)"
        )
        if Zu_ts is not None and Zu_txt is not None:
            _scatter_time_2d(
                Zu_ts, t_k,
                os.path.join(out_dir, f"umap_{file_prefix}_TS_time.png"),
                f"{title_prefix} TS (UMAP; color=time)"
            )
            _scatter_time_2d(
                Zu_txt, t_k,
                os.path.join(out_dir, f"umap_{file_prefix}_TXT_time.png"),
                f"{title_prefix} TXT (UMAP; color=time)"
            )

    elif color_by == "movement":
        _scatter_movement_2d(
            Z_ts, mv_k,
            os.path.join(out_dir, f"tsne_{file_prefix}_TS_movement.png"),
            f"{title_prefix} TS (t-SNE; color=movement)"
        )
        _scatter_movement_2d(
            Z_txt, mv_k,
            os.path.join(out_dir, f"tsne_{file_prefix}_TXT_movement.png"),
            f"{title_prefix} TXT (t-SNE; color=movement)"
        )
        if Zu_ts is not None and Zu_txt is not None:
            _scatter_movement_2d(
                Zu_ts, mv_k,
                os.path.join(out_dir, f"umap_{file_prefix}_TS_movement.png"),
                f"{title_prefix} TS (UMAP; color=movement)"
            )
            _scatter_movement_2d(
                Zu_txt, mv_k,
                os.path.join(out_dir, f"umap_{file_prefix}_TXT_movement.png"),
                f"{title_prefix} TXT (UMAP; color=movement)"
            )
    else:
        raise ValueError(f"Unknown color_by: {color_by}")

    if cfg.save_npz:
        np.savez(
            os.path.join(out_dir, f"tokens_{tag}.npz"),
            ts=ts_np,
            txt=txt_np,
            t=t_k,
            movement=mv_k,
        )

    return retrieval



# =========================
# Main entry
# =========================
def run_retrieval_tsne(
    exp,
    setting_dir: str,
    *,
    cfg: RetrievalTSNEConfig,
) -> Dict[str, Any]:
    """
    Outputs saved to: <setting_dir>/evaluation/retrieval_tsne/
    baseline REMOVED.

    Naming convention:
      - align1 => semantic alignment shared representation (color=time)
      - align2 => movement alignment shared representation (color=movement)

    Must distinguish conditions:
      - Only Semantic Alignment  -> Semantic Alignment Shared Representation
      - Only Movement Alignment  -> Movement Alignment Shared Representation
      - Dual Alignment           -> Semantic Alignment Shared Representation / Movement Alignment Shared Representation
    """
    out_dir = os.path.join(setting_dir, "evaluation", "retrieval_tsne")
    _mkdir(out_dir)

    model = exp.model.module if hasattr(exp.model, "module") else exp.model
    model.eval()
    exp.mlp.eval()

    use1 = bool(getattr(model, "use_alignment1_default", False))
    use2 = bool(getattr(model, "use_alignment2_default", False))

    if use1 and (not use2):
        cond_label = "Only SA"
    elif use2 and (not use1):
        cond_label = "Only MA"
    elif use1 and use2:
        cond_label = "DA"
    else:
        cond_label = "No Alignment"

    test_data, test_loader = exp._get_data(flag="test")

    # ---- TRACE PRINTS ----
    print(f"[retrieval_tsne] setting_dir = {setting_dir}")
    print(f"[retrieval_tsne] test_data len = {len(test_data) if hasattr(test_data, '__len__') else 'NA'}")
    try:
        print(f"[retrieval_tsne] test_loader len(batches) = {len(test_loader)}")
    except Exception:
        print("[retrieval_tsne] test_loader len(batches) = NA")
    print(f"[retrieval_tsne] cfg.align1_batches = {cfg.align1_batches}, cfg.align2_batches = {cfg.align2_batches}")
    print(f"[retrieval_tsne] mode_detected use_alignment_1={use1}, use_alignment_2={use2}")
    print(f"[retrieval_tsne] cfg.enable_dedup = {bool(cfg.enable_dedup)}")

    summary: Dict[str, Any] = {
        "setting_dir": setting_dir,
        "mode_detected": {"use_alignment_1": use1, "use_alignment_2": use2},
        "note": "baseline removed. align1: color=time; align2: color=movement{-1,+1}.",
        "diag": {
            "eval_retrieval_tsne_loaded_from": __file__,
            "tsne_perplexity_requested": float(cfg.tsne_perplexity),
            "align1_batches": cfg.align1_batches,
            "align2_batches": cfg.align2_batches,
            "cond_label": cond_label,
            "enable_dedup": bool(cfg.enable_dedup),
        }
    }

    # -------------------------
    # align1 => Semantic Alignment Shared Representation
    # -------------------------
    if use1:
        diag_a1 = {}
        title_prefix_a1 = f"{cond_label} - Semantic Shared Representation"

        if cfg.align1_batches == 1 or cfg.align1_batches == "1":
            batch = next(iter(test_loader))
            batch_x, batch_y, batch_x_mark, batch_y_mark, index = batch

            print(f"[retrieval_tsne][align1] SINGLE batch mode: batch_x={tuple(batch_x.shape)} index_type={type(index)}")

            ts_btd, txt_btd, mv_bt, diag_a1 = _extract_align1_tokens(
                exp, model, batch_x, batch_x_mark, index, test_data
            )

            ts_kd = ts_btd[0]
            txt_kd = txt_btd[0]
            t_k = np.arange(1, ts_kd.shape[0] + 1, dtype=np.float32)
            mv_k = mv_bt[0].detach().cpu().numpy()

        else:
            ts_all, txt_all, mv_all = [], [], []
            idx_all = []  # NEW: keep indices per batch for dedup
            total_samples = 0
            total_tokens = 0

            for bi, batch in enumerate(_iter_n_batches(test_loader, cfg.align1_batches)):
                batch_x, batch_y, batch_x_mark, batch_y_mark, index = batch

                ts_btd, txt_btd, mv_bt, diag_a1 = _extract_align1_tokens(
                    exp, model, batch_x, batch_x_mark, index, test_data
                )

                B, T, D = ts_btd.shape
                tokens_here = int(B * T)
                total_samples += int(B)
                total_tokens += tokens_here

                print(
                    f"[retrieval_tsne][align1] batch#{bi}: B={B}, T={T}, D={D}, tokens(batch)=B*T={tokens_here}, "
                    f"cum_samples={total_samples}, cum_tokens={total_tokens}"
                )

                ts_all.append(ts_btd)
                txt_all.append(txt_btd)
                mv_all.append(mv_bt)
                idx_all.append(index)

            ts_btd = torch.cat(ts_all, dim=0)
            txt_btd = torch.cat(txt_all, dim=0)
            mv_bt = torch.cat(mv_all, dim=0)

            # concat indices (for dedup)
            if len(idx_all) > 0:
                if torch.is_tensor(idx_all[0]):
                    index_cat = torch.cat([x.detach().cpu() for x in idx_all], dim=0)
                else:
                    index_cat = sum([list(x) for x in idx_all], [])
            else:
                index_cat = None

            print(f"[retrieval_tsne][align1] concat shapes: ts={tuple(ts_btd.shape)} txt={tuple(txt_btd.shape)} mv={tuple(mv_bt.shape)}")

            if bool(cfg.enable_dedup) and index_cat is not None:
                ts_kd, txt_kd, t_k, mv_k = _collect_tokens_with_dedup(ts_btd, txt_btd, mv_bt, index_cat)
                print(f"[retrieval_tsne][align1] DEDUP enabled -> K={int(ts_kd.shape[0])}")
            else:
                ts_kd, txt_kd, t_k, mv_k = _collect_tokens_no_dedup(ts_btd, txt_btd, mv_bt)
                print(f"[retrieval_tsne][align1] DEDUP disabled -> K={int(ts_kd.shape[0])}")

        print(f"[retrieval_tsne][align1] final shapes: ts={tuple(ts_kd.shape)} txt={tuple(txt_kd.shape)} t={int(t_k.shape[0])} mv={int(mv_k.shape[0])}")

        summary["align1"] = {}
        summary["align1"]["display_name"] = title_prefix_a1
        summary["align1"]["plots"] = {"TS": "time", "TXT": "time"}
        summary["align1"]["diag"] = diag_a1
        summary["align1"]["retrieval"] = _save_space(
            out_dir,
            "align1",
            ts_kd,
            txt_kd,
            t_k,
            mv_k,
            cfg,
            color_by="time",
            title_prefix=title_prefix_a1,
        )

    # -------------------------
    # align2 => Movement Alignment Shared Representation
    # -------------------------
    if use2:
        title_prefix_a2 = f"{cond_label} - Movement Contrastive Representation"

        if cfg.align2_batches == 1 or cfg.align2_batches == "1":
            batch = next(iter(test_loader))
            batch_x, batch_y, batch_x_mark, batch_y_mark, index = batch

            print(f"[retrieval_tsne][align2] SINGLE batch mode: batch_x={tuple(batch_x.shape)} index_type={type(index)}")

            ts_btd, txt_btd, mv_bt = _extract_align2_tokens(exp, model, batch_x, index, test_data)

            ts_kd = ts_btd[0]
            txt_kd = txt_btd[0]
            t_k = np.arange(1, ts_kd.shape[0] + 1, dtype=np.float32)
            mv_k = mv_bt[0].detach().cpu().numpy()

        else:
            ts_all, txt_all, mv_all = [], [], []
            idx_all = []  # NEW
            total_samples = 0
            total_tokens = 0

            for bi, batch in enumerate(_iter_n_batches(test_loader, cfg.align2_batches)):
                batch_x, batch_y, batch_x_mark, batch_y_mark, index = batch
                ts_btd, txt_btd, mv_bt = _extract_align2_tokens(exp, model, batch_x, index, test_data)

                B, T, D = ts_btd.shape
                tokens_here = int(B * T)
                total_samples += int(B)
                total_tokens += tokens_here

                print(
                    f"[retrieval_tsne][align2] batch#{bi}: B={B}, T={T}, D={D}, tokens(batch)=B*T={tokens_here}, "
                    f"cum_samples={total_samples}, cum_tokens={total_tokens}"
                )

                ts_all.append(ts_btd)
                txt_all.append(txt_btd)
                mv_all.append(mv_bt)
                idx_all.append(index)

            ts_btd = torch.cat(ts_all, dim=0)
            txt_btd = torch.cat(txt_all, dim=0)
            mv_bt = torch.cat(mv_all, dim=0)

            if len(idx_all) > 0:
                if torch.is_tensor(idx_all[0]):
                    index_cat = torch.cat([x.detach().cpu() for x in idx_all], dim=0)
                else:
                    index_cat = sum([list(x) for x in idx_all], [])
            else:
                index_cat = None

            print(f"[retrieval_tsne][align2] concat shapes: ts={tuple(ts_btd.shape)} txt={tuple(txt_btd.shape)} mv={tuple(mv_bt.shape)}")

            if bool(cfg.enable_dedup) and index_cat is not None:
                ts_kd, txt_kd, t_k, mv_k = _collect_tokens_with_dedup(ts_btd, txt_btd, mv_bt, index_cat)
                print(f"[retrieval_tsne][align2] DEDUP enabled -> K={int(ts_kd.shape[0])}")
            else:
                ts_kd, txt_kd, t_k, mv_k = _collect_tokens_no_dedup(ts_btd, txt_btd, mv_bt)
                print(f"[retrieval_tsne][align2] DEDUP disabled -> K={int(ts_kd.shape[0])}")

        print(f"[retrieval_tsne][align2] final shapes: ts={tuple(ts_kd.shape)} txt={tuple(txt_kd.shape)} t={int(t_k.shape[0])} mv={int(mv_k.shape[0])}")

        summary["align2"] = {}
        summary["align2"]["display_name"] = title_prefix_a2
        summary["align2"]["plots"] = {"TS": "movement", "TXT": "movement"}
        summary["align2"]["retrieval"] = _save_space(
            out_dir,
            "align2",
            ts_kd,
            txt_kd,
            t_k,
            mv_k,
            cfg,
            color_by="movement",
            title_prefix=title_prefix_a2,
        )

    if cfg.run_cross_inputs:
        summary["cross_inputs_note"] = "cross_inputs kept as TODO (not requested in this patch)."

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
