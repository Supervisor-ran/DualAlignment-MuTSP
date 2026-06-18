# eval_grad_consistency.py
# ------------------------------------------------------------
# Gradient consistency:
# cosine distribution between gradients of:
#   - MSE loss (forecast)
#   - alignment loss (returned by model forward)
#
# Updates in this version:
# - Pass the SAME forward kwargs as your training/vali pipeline:
#     use_alignment_1/use_alignment_2, alignment_mode/delta/tau, y_true,
#     use_cross, cross_mode
# - Works with MultiVarIndepWrapper (DATransformer multivar_indep):
#     model.named_parameters() includes models.{v}.* so per-submodel grads
#     are naturally included.
# - Optional: include exp.mlp params in grad stats (include_text_mlp=True)
# - Adds light diagnostics to summary (counts of None grads, etc.)
#
# NOTE:
# - This script does NOT call .backward(); uses torch.autograd.grad
# - It expects model forward signature supports:
#     model(batch_x, batch_x_mark, dec_inp, batch_y_mark, z_text=..., ...)
# ------------------------------------------------------------

from __future__ import annotations
import os
import json
import csv
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


@dataclass
class GradConsistencyConfig:
    max_batches: int = 50
    max_params: int = 200000  # cap rows written to CSV (avoid huge files)
    bins_param: int = 60
    bins_batch: int = 40
    save_csv: bool = True
    save_plots: bool = True

    # NEW
    include_text_mlp: bool = False  # include exp.mlp params in grad cosine stats
    strict_align: bool = False      # if True, raise when align_loss is None (otherwise skip)


def _mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def _iter_named_params(module) -> List[Tuple[str, torch.nn.Parameter]]:
    return [(n, p) for (n, p) in module.named_parameters() if p.requires_grad]


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a = a.flatten()
    b = b.flatten()
    na = a.norm() + eps
    nb = b.norm() + eps
    return float((a @ b).item() / (na * nb))


def _write_csv(path: str, rows: List[Dict[str, Any]]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


@torch.no_grad()
def _prepare_text_emb(exp, dataset, index, T_enc: int) -> torch.Tensor:
    """
    Use executor-side text encoding (then MLP), and time-align to encoder length.
    Matches your train/vali pipeline.
    """
    # Try dataset.get_text_pair first
    hist_text = None
    if hasattr(dataset, "get_text_pair"):
        try:
            hist_text, _ = dataset.get_text_pair(index)
        except Exception:
            hist_text = None

    # Fallback to dataset.get_text
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

    # text MLP (may be frozen; if frozen, requires_grad=False and won't be counted unless include_text_mlp=False anyway)
    if getattr(exp, "mlp", None) is not None:
        txt = exp.mlp(txt)

    txt = exp._time_align_text_emb(txt, target_len=T_enc)
    return txt


def _get_pred_len_label_len(exp, model) -> Tuple[int, int]:
    pred_len = getattr(getattr(exp, "args", None), "pred_len", None)
    label_len = getattr(getattr(exp, "args", None), "label_len", None)

    if pred_len is None:
        pred_len = int(getattr(model, "pred_len"))
    if label_len is None:
        label_len = int(getattr(getattr(exp, "args", None), "label_len", pred_len))
    return int(pred_len), int(label_len)


def _forward_kwargs_from_exp(exp, batch_y: torch.Tensor) -> Dict[str, Any]:
    """
    Mirror the kwargs used in Exp_Long_Term_Forecast for (plus/DATransformer) branches.
    """
    args = getattr(exp, "args", None)

    def _get(name: str, default: Any):
        return getattr(args, name, default) if args is not None else default

    return dict(
        use_alignment_1=_get("use_alignment_1", True),
        alignment_mode=_get("alignment_mode", "cosine"),
        alignment_delta=_get("alignment_delta", 0),
        alignment_tau=_get("alignment_tau", 0.07),
        use_alignment_2=_get("use_alignment_2", False),
        y_true=batch_y,  # IMPORTANT: many align losses need ground-truth
        use_cross=_get("use_cross_modality", True),
        cross_mode=_get("cross_modality_mode", "seperate_cross"),
    )


def run_grad_consistency(
    exp,
    setting_dir: str,
    *,
    cfg: GradConsistencyConfig,
) -> Dict[str, Any]:
    """
    Outputs saved to: <setting_dir>/evaluation/grad_consistency/
    """
    out_dir = os.path.join(setting_dir, "evaluation", "grad_consistency")
    _mkdir(out_dir)

    model = exp.model.module if hasattr(exp.model, "module") else exp.model
    model.eval()

    if getattr(exp, "mlp", None) is not None:
        exp.mlp.eval()

    test_data, test_loader = exp._get_data(flag="test")
    device = exp.device

    # collect params
    named_params_model = _iter_named_params(model)

    named_params_mlp: List[Tuple[str, torch.nn.Parameter]] = []
    if cfg.include_text_mlp and getattr(exp, "mlp", None) is not None:
        named_params_mlp = _iter_named_params(exp.mlp)

    named_params = []
    named_params.extend([(f"model::{n}", p) for (n, p) in named_params_model])
    named_params.extend([(f"mlp::{n}", p) for (n, p) in named_params_mlp])

    if not named_params:
        raise RuntimeError("No trainable parameters found (requires_grad=True) in model/mlp.")

    names = [n for (n, _) in named_params]
    params = [p for (_, p) in named_params]

    pred_len, label_len = _get_pred_len_label_len(exp, model)

    # outputs
    cos_per_param: List[float] = []
    cos_overall_per_batch: List[float] = []
    rows_param: List[Dict[str, Any]] = []
    rows_batch: List[Dict[str, Any]] = []

    # counters
    total_seen = 0
    used = 0
    skipped_no_align = 0
    skipped_nonfinite_align = 0
    skipped_no_valid_pairs = 0

    # diagnostics
    diag_align_none_batches: List[int] = []
    diag_valid_pairs_per_batch: List[int] = []
    diag_none_grad_mse = 0
    diag_none_grad_align = 0

    for b, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(test_loader):
        if b >= cfg.max_batches:
            break

        total_seen += 1

        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        # decoder inp like exp.vali/test
        dec_zeros = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :label_len, :], dec_zeros], dim=1).to(device)

        # executor text embeddings
        z_text = _prepare_text_emb(exp, test_data, index, T_enc=batch_x.shape[1]).to(device)

        fw_kw = _forward_kwargs_from_exp(exp, batch_y=batch_y)

        # ---- single forward ----
        out = model(
            batch_x, batch_x_mark, dec_inp, batch_y_mark,
            z_text=z_text,
            **fw_kw
        )

        if isinstance(out, tuple):
            pred, align_loss = out
        else:
            pred, align_loss = out, None

        # target
        y_true = batch_y[:, -pred_len:, :]
        mse_loss = F.mse_loss(pred, y_true)

        # no align loss => skip (expected for pure-MSE checkpoints)
        if align_loss is None or (not torch.is_tensor(align_loss)):
            skipped_no_align += 1
            diag_align_none_batches.append(b)
            rows_batch.append({
                "batch": b,
                "status": "skipped_no_align",
                "overall_cosine": None,
                "valid_param_pairs": 0,
                "mse": float(mse_loss.detach().item()),
                "align": None,
            })
            if cfg.strict_align:
                raise RuntimeError(f"align_loss is None at batch={b} (strict_align=True).")
            continue

        # non-finite align loss => skip
        if (not torch.isfinite(align_loss).all().item()):
            skipped_nonfinite_align += 1
            rows_batch.append({
                "batch": b,
                "status": "skipped_nonfinite_align",
                "overall_cosine": None,
                "valid_param_pairs": 0,
                "mse": float(mse_loss.detach().item()),
                "align": float(align_loss.detach().float().item()) if align_loss.numel() == 1 else None,
            })
            continue

        # ---- grads via autograd.grad (no .backward) ----
        grads_mse = torch.autograd.grad(
            mse_loss, params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )

        grads_align = torch.autograd.grad(
            align_loss, params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )

        v_mse_list = []
        v_align_list = []
        valid_pairs = 0

        for name, g1, g2 in zip(names, grads_mse, grads_align):
            if g1 is None:
                diag_none_grad_mse += 1
            if g2 is None:
                diag_none_grad_align += 1

            if g1 is None or g2 is None:
                continue

            g1d = g1.detach()
            g2d = g2.detach()

            if (not torch.isfinite(g1d).all().item()) or (not torch.isfinite(g2d).all().item()):
                continue

            c = _cosine(g1d, g2d)
            cos_per_param.append(c)
            valid_pairs += 1

            v_mse_list.append(g1d.flatten())
            v_align_list.append(g2d.flatten())

            if cfg.save_csv and len(rows_param) < cfg.max_params:
                rows_param.append({
                    "batch": b,
                    "param": name,
                    "cosine": c,
                    "norm_mse": float(g1d.norm().item()),
                    "norm_align": float(g2d.norm().item()),
                })

        diag_valid_pairs_per_batch.append(int(valid_pairs))

        if valid_pairs == 0:
            skipped_no_valid_pairs += 1
            rows_batch.append({
                "batch": b,
                "status": "skipped_no_valid_grad_pairs",
                "overall_cosine": None,
                "valid_param_pairs": 0,
                "mse": float(mse_loss.detach().item()),
                "align": float(align_loss.detach().float().item()) if align_loss.numel() == 1 else None,
            })
            continue

        v1 = torch.cat(v_mse_list, dim=0)
        v2 = torch.cat(v_align_list, dim=0)
        overall_c = _cosine(v1, v2)
        cos_overall_per_batch.append(overall_c)

        used += 1
        rows_batch.append({
            "batch": b,
            "status": "ok",
            "overall_cosine": overall_c,
            "valid_param_pairs": int(valid_pairs),
            "mse": float(mse_loss.detach().item()),
            "align": float(align_loss.detach().float().item()) if align_loss.numel() == 1 else None,
        })

    # ---- outputs ----
    if cfg.save_csv:
        _write_csv(os.path.join(out_dir, "grad_cosine_per_param.csv"), rows_param)
        _write_csv(os.path.join(out_dir, "grad_cosine_per_batch.csv"), rows_batch)

    if cfg.save_plots:
        if len(cos_per_param) > 0:
            plt.figure()
            plt.hist(np.array(cos_per_param, dtype=np.float32), bins=int(cfg.bins_param))
            plt.title("Grad cosine (MSE vs AlignLoss) per-param")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "grad_cosine_hist.png"), dpi=200)
            plt.close()

        if len(cos_overall_per_batch) > 0:
            plt.figure()
            plt.hist(np.array(cos_overall_per_batch, dtype=np.float32), bins=int(cfg.bins_batch))
            plt.title("Overall grad cosine (MSE vs AlignLoss) per-batch")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "grad_cosine_overall_hist.png"), dpi=200)
            plt.close()

    def _safe_mean(xs: List[float]) -> Optional[float]:
        return float(np.mean(xs)) if len(xs) else None

    def _safe_std(xs: List[float]) -> Optional[float]:
        return float(np.std(xs)) if len(xs) else None

    summary = {
        "batches_total_seen": int(total_seen),
        "batches_used": int(used),
        "batches_skipped_no_align": int(skipped_no_align),
        "batches_skipped_nonfinite_align": int(skipped_nonfinite_align),
        "batches_skipped_no_valid_grad_pairs": int(skipped_no_valid_pairs),
        "num_param_cosines": int(len(cos_per_param)),
        "cosine_mean": _safe_mean(cos_per_param),
        "cosine_std": _safe_std(cos_per_param),
        "overall_cosine_mean": _safe_mean(cos_overall_per_batch),
        "overall_cosine_std": _safe_std(cos_overall_per_batch),
        "pred_len": int(pred_len),
        "label_len": int(label_len),
        "config": cfg.__dict__,
        "diag": {
            "include_text_mlp": bool(cfg.include_text_mlp),
            "total_trainable_params_counted": int(len(params)),
            "align_none_batches_first10": diag_align_none_batches[:10],
            "valid_pairs_per_batch_first10": diag_valid_pairs_per_batch[:10],
            "none_grad_mse_count": int(diag_none_grad_mse),
            "none_grad_align_count": int(diag_none_grad_align),
        },
    }

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
