# eval_noise_injection.py
# ------------------------------------------------------------
# Noise injection robustness (STRONG CONSISTENCY / AUTO-FOLLOW CKPT):
# - Add Gaussian noise to x_enc (time series) with sigma = level * std(x_enc)
# - Evaluate forecast MSE over test loader
# - Optionally drop text modality (z_text=None) to simulate missing text
#
# STRONG CONSISTENCY PATCH:
# - If cfg.use_alignment_1/2/use_cross/... are None:
#     we explicitly pull defaults from exp.args (loaded from checkpoint)
#     and pass them into model.forward, so behavior matches checkpoint args
#     deterministically.
# - Always pass y_true=batch_y (safe for models that ignore it).
#
# Reproducibility patch:
# - cfg.seed + cfg.deterministic controls global seed & deterministic backends
# - each noise_level uses its own torch.Generator seeded by (seed + 1000*level_index)
#   to avoid RNG consumption coupling across levels/batches.
# ------------------------------------------------------------

from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


@dataclass
class NoiseInjectionConfig:
    noise_levels: Optional[List[float]] = None  # e.g. [0.0, 0.05, 0.1, 0.2]
    max_batches: int = 100
    drop_text: bool = False

    # --- Reproducibility ---
    seed: int = 42
    deterministic: bool = True

    # --- Overrides (None => follow checkpoint args explicitly) ---
    use_alignment_1: Optional[bool] = None
    use_alignment_2: Optional[bool] = None
    use_cross: Optional[bool] = None
    cross_mode: Optional[str] = None

    alignment_mode: Optional[str] = None
    alignment_delta: Optional[int] = None
    alignment_tau: Optional[float] = None

    def __post_init__(self):
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.05, 0.1, 0.2]


def _mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def _seed_all(seed: int, deterministic: bool = True):
    """
    Best-effort deterministic setup.
    Note: full determinism may still depend on model ops & CUDA kernels.
    """
    import random

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Optional strict mode (may raise on nondeterministic ops):
        # torch.use_deterministic_algorithms(True)


@torch.no_grad()
def _prepare_text_emb_or_none(exp, dataset, index, T_enc: int, drop_text: bool) -> Optional[torch.Tensor]:
    if drop_text:
        return None

    hist_text = None
    if hasattr(dataset, "get_text_pair"):
        try:
            hist_text, _ = dataset.get_text_pair(index)
        except Exception:
            hist_text = None

    if hist_text is None and hasattr(dataset, "get_text"):
        tmp = dataset.get_text(index)
        try:
            hist_text = tmp[:, 1]
        except Exception:
            hist_text = tmp

    if hist_text is None:
        return None

    if getattr(exp, "use_bertopic_finbert", False):
        txt = exp._encode_with_bertopic_finbert(hist_text)
    else:
        txt = exp._encode_text_list(hist_text)

    # exp.mlp may be None or frozen; call if exists
    if getattr(exp, "mlp", None) is not None:
        txt = exp.mlp(txt)

    txt = exp._time_align_text_emb(txt, target_len=T_enc)
    return txt


def _add_noise_like(
    x: torch.Tensor,
    level: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if level <= 0:
        return x
    # per-sample std over time dim (T) (keeps channel dim)
    std = x.std(dim=1, keepdim=True).clamp_min(1e-6)
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * (std * level)
    return x + noise


def _resolve_fw_kwargs_from_ckpt(exp, cfg: NoiseInjectionConfig) -> Dict[str, Any]:
    """
    STRONG CONSISTENCY:
    - If cfg field is None => pull from exp.args (checkpoint-loaded)
    - Else => use cfg override
    """
    args = getattr(exp, "args", None)

    def _get_from_args(name: str, default: Any) -> Any:
        if args is None:
            return default
        return getattr(args, name, default)

    fw: Dict[str, Any] = {}

    # ---- alignment flags ----
    # train code uses:
    #   use_alignment_1, use_alignment_2, use_cross_modality (sometimes passed as use_cross), cross_modality_mode
    # we map to model.forward expected keys:
    #   use_alignment_1, use_alignment_2, use_cross, cross_mode
    if cfg.use_alignment_1 is None:
        fw["use_alignment_1"] = bool(_get_from_args("use_alignment_1", True))
    else:
        fw["use_alignment_1"] = bool(cfg.use_alignment_1)

    if cfg.use_alignment_2 is None:
        fw["use_alignment_2"] = bool(_get_from_args("use_alignment_2", False))
    else:
        fw["use_alignment_2"] = bool(cfg.use_alignment_2)

    if cfg.use_cross is None:
        fw["use_cross"] = bool(_get_from_args("use_cross_modality", True))
    else:
        fw["use_cross"] = bool(cfg.use_cross)

    if cfg.cross_mode is None:
        fw["cross_mode"] = str(_get_from_args("cross_modality_mode", "seperate_cross"))
    else:
        fw["cross_mode"] = str(cfg.cross_mode)

    # ---- alignment hyperparams ----
    if cfg.alignment_mode is None:
        fw["alignment_mode"] = str(_get_from_args("alignment_mode", "cosine"))
    else:
        fw["alignment_mode"] = str(cfg.alignment_mode)

    if cfg.alignment_delta is None:
        fw["alignment_delta"] = int(_get_from_args("alignment_delta", 0))
    else:
        fw["alignment_delta"] = int(cfg.alignment_delta)

    if cfg.alignment_tau is None:
        fw["alignment_tau"] = float(_get_from_args("alignment_tau", 0.07))
    else:
        fw["alignment_tau"] = float(cfg.alignment_tau)

    return fw


@torch.no_grad()
def run_noise_injection(
    exp,
    setting_dir: str,
    *,
    cfg: NoiseInjectionConfig,
) -> Dict[str, Any]:
    """
    Outputs saved to: <setting_dir>/evaluation/noise_injection/
    """
    out_dir = os.path.join(
        setting_dir, "evaluation", "noise_injection", f"drop_text_{bool(cfg.drop_text)}"
    )
    _mkdir(out_dir)

    model = exp.model.module if hasattr(exp.model, "module") else exp.model
    model.eval()
    if getattr(exp, "mlp", None) is not None:
        exp.mlp.eval()

    test_data, test_loader = exp._get_data(flag="test")
    device = exp.device

    # --- reproducibility ---
    _seed_all(int(cfg.seed), deterministic=bool(cfg.deterministic))

    # --- STRONG CONSISTENCY: explicitly follow checkpoint args unless overridden ---
    base_fw = _resolve_fw_kwargs_from_ckpt(exp, cfg)

    # helpful diag dump
    diag = {
        "resolved_forward_kwargs": dict(base_fw),
        "cfg": cfg.__dict__,
        "exp_args_excerpt": {
            "use_alignment_1": getattr(exp.args, "use_alignment_1", None) if hasattr(exp, "args") else None,
            "use_alignment_2": getattr(exp.args, "use_alignment_2", None) if hasattr(exp, "args") else None,
            "use_cross_modality": getattr(exp.args, "use_cross_modality", None) if hasattr(exp, "args") else None,
            "cross_modality_mode": getattr(exp.args, "cross_modality_mode", None) if hasattr(exp, "args") else None,
            "alignment_mode": getattr(exp.args, "alignment_mode", None) if hasattr(exp, "args") else None,
            "alignment_delta": getattr(exp.args, "alignment_delta", None) if hasattr(exp, "args") else None,
            "alignment_tau": getattr(exp.args, "alignment_tau", None) if hasattr(exp, "args") else None,
        },
    }
    with open(os.path.join(out_dir, "forward_kwargs_diag.json"), "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2, ensure_ascii=False)

    results = []
    levels = cfg.noise_levels or []

    pred_len = int(getattr(exp.args, "pred_len"))
    label_len = int(getattr(exp.args, "label_len"))

    for li, level in enumerate(levels):
        mse_list: List[float] = []

        # independent RNG stream per noise level
        g = torch.Generator(device=device)
        g.manual_seed(int(cfg.seed) + 1000 * int(li))

        for b, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(test_loader):
            if b >= int(cfg.max_batches):
                break

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            # inject noise on x
            x_noisy = _add_noise_like(batch_x, float(level), generator=g)

            # build dec_inp (same as your DATransformer path)
            dec_zeros = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_zeros], dim=1).to(device)

            z_text = _prepare_text_emb_or_none(
                exp, test_data, index, T_enc=x_noisy.shape[1], drop_text=bool(cfg.drop_text)
            )
            if z_text is not None:
                z_text = z_text.to(device)

            # forward (STRONG CONSISTENCY)
            fw = dict(base_fw)

            # ✅ always provide y_true to match training/vali + grad_consistency
            fw["y_true"] = batch_y

            out = model(
                x_noisy, batch_x_mark, dec_inp, batch_y_mark,
                z_text=z_text,
                **fw
            )

            if isinstance(out, tuple):
                pred, _align_loss = out
            else:
                pred = out

            y_true = batch_y[:, -pred_len:, :]
            mse = F.mse_loss(pred, y_true).item()
            mse_list.append(float(mse))

        results.append({
            "noise_level": float(level),
            "mse_mean": float(np.mean(mse_list)) if mse_list else float("nan"),
            "mse_std": float(np.std(mse_list)) if mse_list else float("nan"),
            "batches_used": int(len(mse_list)),
            "drop_text": bool(cfg.drop_text),
        })

    # save json (full)
    with open(os.path.join(out_dir, "noise_results.json"), "w", encoding="utf-8") as f:
        json.dump({"config": cfg.__dict__, "resolved_forward_kwargs": base_fw, "results": results},
                  f, indent=2, ensure_ascii=False)

    # plot
    xs = [r["noise_level"] for r in results]
    ys = [r["mse_mean"] for r in results]
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("noise_level (sigma = level * std(x))")
    plt.ylabel("forecast MSE (mean)")
    plt.title("Noise Injection Robustness")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "mse_vs_noise.png"), dpi=200)
    plt.close()

    summary = {
        "config": {
            "seed": int(cfg.seed),
            "deterministic": bool(cfg.deterministic),
            "max_batches": int(cfg.max_batches),
            "drop_text": bool(cfg.drop_text),
            "noise_levels": list(levels),
        },
        "resolved_forward_kwargs": dict(base_fw),
        "num_levels": int(len(results)),
        "best_level": float(xs[int(np.nanargmin(ys))]) if len(ys) else None,
        "best_mse": float(np.nanmin(ys)) if len(ys) else None,
        "results": results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary
