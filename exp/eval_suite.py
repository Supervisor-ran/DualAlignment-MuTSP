# eval_suite.py
# ------------------------------------------------------------
# Suite runner (NO CLI):
# - retrieval+tsne
# - grad consistency
# - noise injection (ONLY drop_text=False)
#
# PLUS: group-merge across 4 ablation checkpoints that share the same
# prefix before the 5th underscore, e.g.:
# TTC_Climate_2021_12_2512242314_...
# group key => TTC_Climate_2021_12_2512242314
#
# We create:
# abl_test/merged/<group_key>/
# - retrieval_table.csv
# - grad_consistency_table.csv
# - noise_injection_table.csv
# - tsne_gallery.png
# - grad_hist_row.png
# - noise_gallery.png (1 plot, 4 lines: baseline/align1/align2/both; with_text only)
#
# Conservative strategy updates:
# - Baseline checkpoint may still exist (for noise_gallery, classify, etc.)
# - But retrieval/tsne no longer expects or consumes baseline plots/json.
# - "trend" is renamed to "movement" in retrieval/tsne outputs and gallery lookup.
# - Gallery rows updated:
#     align1 : only time color
#     align2 : only movement color
#     cross_inputs: keep (time + movement) if present
# ------------------------------------------------------------

from __future__ import annotations
import os
import sys
import json
import csv
import traceback
from typing import Any, Dict, Optional, List, Tuple
from types import SimpleNamespace

import torch
import torch.nn as nn  # for materialize


# ---- make same-folder import robust ----
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
os.chdir(PROJECT_ROOT)

if HERE not in sys.path:
    sys.path.insert(0, HERE)

from exp_long_term_forecasting import Exp_Long_Term_Forecast

from eval_retrieval_tsne import run_retrieval_tsne, RetrievalTSNEConfig
from eval_grad_consistency import run_grad_consistency, GradConsistencyConfig
from eval_noise_injection import run_noise_injection, NoiseInjectionConfig

# ============================================================
# Suite switches (NO CLI)
# ============================================================
class SuiteSwitches:
    # master switches
    RUN_RETRIEVAL_TSNE = False
    RUN_GRAD_CONSISTENCY = True
    RUN_NOISE_INJECTION = True

    # retrieval tsne config
    ALIGN1_BATCHES = 1   # int or "all"
    ALIGN2_BATCHES = 1   # int or "all"
    RUN_CROSS_INPUTS = False # 你现在不需要 cross_inputs 就关掉

    # noise injection config (only used if RUN_NOISE_INJECTION=True)
    NOISE_LEVELS = [0.0, 0.05, 0.1, 0.2]
    NOISE_MAX_BATCHES = 100
    NOISE_DROP_TEXT = False  # 你之前就是 ONLY drop_text=False



# ============================================================
# Helpers
# ============================================================
def _mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def _is_setting_dir_has_ckpt(setting_dir: str) -> bool:
    return (
        os.path.isfile(os.path.join(setting_dir, "checkpoint_all.pth"))
        or os.path.isfile(os.path.join(setting_dir, "checkpoint.pth"))
    )



def _find_setting_dirs(checkpoints_root: str) -> List[str]:
    """
    Find subfolders under checkpoints_root that contain checkpoint_all.pth OR checkpoint.pth.
    We scan depth=2 to be tolerant.
    """
    out: List[str] = []
    if not os.path.isdir(checkpoints_root):
        return out

    # depth=1
    for name in sorted(os.listdir(checkpoints_root)):
        p = os.path.join(checkpoints_root, name)
        if os.path.isdir(p) and _is_setting_dir_has_ckpt(p):
            out.append(p)

    # depth=2 (optional)
    if not out:
        for root, _dirs, files in os.walk(checkpoints_root):
            if ("checkpoint_all.pth" in files) or ("checkpoint.pth" in files):
                out.append(root)
        out = sorted(list(set(out)))

    return out



def _load_checkpoint(ckpt_path: str, device: str = "cpu") -> Dict[str, Any]:
    return torch.load(ckpt_path, map_location=device)


def _to_namespace(args_obj: Any):
    """
    Exp_Long_Term_Forecast expects Namespace-like.
    Accept dict / Namespace-like object.
    Must be picklable (Windows DataLoader num_workers>0).
    """
    if hasattr(args_obj, "__dict__") and not isinstance(args_obj, dict):
        return args_obj

    if isinstance(args_obj, dict):
        return SimpleNamespace(**args_obj)

    raise TypeError(f"Unsupported args type in checkpoint: {type(args_obj)}")


def _extract_state_dict(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    for k in ["model", "state_dict", "model_state_dict", "net", "network"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        if "state_dict" in ckpt["model"] and isinstance(ckpt["model"]["state_dict"], dict):
            return ckpt["model"]["state_dict"]
    raise KeyError("Cannot find model state_dict in checkpoint.")


def _extract_mlp_state_dict(ckpt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for k in ["mlp", "mlp_state_dict", "text_mlp", "prompt_mlp"]:
        if k in ckpt and isinstance(ckpt[k], dict):
            return ckpt[k]
    return None


def _setting_name(setting_dir: str) -> str:
    return os.path.basename(os.path.abspath(setting_dir))


def _group_key_from_setting_name(name: str) -> str:
    """
    group key = before the 4th underscore => first 4 tokens split by '_'
    e.g. TTC_Climate_2021_12_251224... -> TTC_Climate_2021_12
    """
    parts = name.split("_")
    if len(parts) < 5:
        return name
    return "_".join(parts[:5])


# ============================================================
# Materialize lazy layers (NO align2model_ts fallback)
# ============================================================
def _has_any_key_prefix(sd: Dict[str, Any], prefix: str) -> bool:
    p = prefix + "."
    for k in sd.keys():
        if k == prefix or k.startswith(p):
            return True
    return False


def _pick_weight_bias(sd: Dict[str, Any], base: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
    """
    Try:
      base.weight / base.bias
      base.0.weight / base.0.bias  (nn.Sequential)
    """
    w = sd.get(f"{base}.weight", None)
    b = sd.get(f"{base}.bias", None)
    if w is not None:
        return w, b, "plain"

    w = sd.get(f"{base}.0.weight", None)
    b = sd.get(f"{base}.0.bias", None)
    if w is not None:
        return w, b, "seq0"

    return None, None, "missing"


def _make_layer_from_weight_bias(w: torch.Tensor, b: Optional[torch.Tensor]) -> nn.Module:
    """
    Infer module type from weight shape:
      - 2D => Linear
      - 3D => Conv1d
    """
    if w.ndim == 2:
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
        return nn.Linear(in_f, out_f, bias=(b is not None))
    if w.ndim == 3:
        out_c, in_c, k = int(w.shape[0]), int(w.shape[1]), int(w.shape[2])
        return nn.Conv1d(in_c, out_c, kernel_size=k, bias=(b is not None))
    return nn.Identity()


def _materialize_module_if_missing(model: nn.Module, sd: Dict[str, Any], name: str) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "name": name,
        "created": False,
        "reason": None,
        "ckpt_has_prefix": _has_any_key_prefix(sd, name),
        "ckpt_variant": None,
        "module_type": None,
        "weight_shape": None,
        "bias_shape": None,
    }

    if hasattr(model, name):
        report["reason"] = "model_already_has_attr"
        return report

    if not report["ckpt_has_prefix"]:
        report["reason"] = "checkpoint_missing_prefix"
        return report

    w, b, variant = _pick_weight_bias(sd, name)
    report["ckpt_variant"] = variant
    if w is None:
        report["reason"] = "cannot_find_weight_tensor"
        return report

    core = _make_layer_from_weight_bias(w, b)
    report["module_type"] = core.__class__.__name__
    report["weight_shape"] = list(w.shape)
    if b is not None:
        report["bias_shape"] = list(b.shape)

    if variant == "plain":
        mod = core
    elif variant == "seq0":
        mod = nn.Sequential(core)
    else:
        report["reason"] = "unknown_variant"
        return report

    setattr(model, name, mod)
    report["created"] = True
    report["reason"] = "created_from_state_dict"
    return report


def _materialize_for_eval(model: nn.Module, sd: Dict[str, Any]) -> Dict[str, Any]:
    """
    Materialize known lazy modules that eval may touch:
      - align1 projection: _proj_ts1/_proj_ts2 OR proj_ts1/proj_ts2
      - retrieval cross inputs: _txt_for_cross1/_txt_for_cross2, optional _pre_cross_txt_to_align
    NOTE: NO align2model_ts fallback here.
    """
    created_reports: List[Dict[str, Any]] = []

    # align1 variants
    for nm in ["_proj_ts1", "_proj_ts2", "proj_ts1", "proj_ts2"]:
        created_reports.append(_materialize_module_if_missing(model, sd, nm))

    # retrieval cross inputs
    for nm in ["_txt_for_cross1", "_txt_for_cross2", "_pre_cross_txt_to_align"]:
        created_reports.append(_materialize_module_if_missing(model, sd, nm))

    out = {
        "created_count": int(sum(1 for r in created_reports if r.get("created"))),
        "reports": created_reports,
        "has_after": {
            "_proj_ts1": hasattr(model, "_proj_ts1"),
            "_proj_ts2": hasattr(model, "_proj_ts2"),
            "proj_ts1": hasattr(model, "proj_ts1"),
            "proj_ts2": hasattr(model, "proj_ts2"),
            "_txt_for_cross1": hasattr(model, "_txt_for_cross1"),
            "_txt_for_cross2": hasattr(model, "_txt_for_cross2"),
            "_pre_cross_txt_to_align": hasattr(model, "_pre_cross_txt_to_align"),
            "align2model_ts": hasattr(model, "align2model_ts"),
        },
    }
    return out


# ============================================================
# Build exp (patched)
# ============================================================
def _build_exp_from_checkpoint(setting_dir: str) -> Tuple[Any, Dict[str, Any]]:
    ckpt_path = _find_first_existing([
        os.path.join(setting_dir, "checkpoint_all.pth"),
        os.path.join(setting_dir, "checkpoint.pth"),
    ])
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found in: {setting_dir}")

    ckpt = _load_checkpoint(ckpt_path, device="cpu")

    if "args" not in ckpt:
        raise KeyError(f"checkpoint missing 'args': {ckpt_path}")

    args = _to_namespace(ckpt["args"])
    exp = Exp_Long_Term_Forecast(args)

    model = exp.model.module if hasattr(exp.model, "module") else exp.model
    sd = _extract_state_dict(ckpt)

    # materialize lazy layers BEFORE load_state_dict
    materialize_report = _materialize_for_eval(model, sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)

    mlp_sd = _extract_mlp_state_dict(ckpt)
    if mlp_sd is not None and hasattr(exp, "mlp") and exp.mlp is not None:
        exp.mlp.load_state_dict(mlp_sd, strict=False)

    _mkdir(os.path.join(setting_dir, "evaluation"))
    load_report = {
        "ckpt_path": ckpt_path,
        "missing_keys": list(missing) if isinstance(missing, (list, tuple)) else [],
        "unexpected_keys": list(unexpected) if isinstance(unexpected, (list, tuple)) else [],
        "materialize_report": materialize_report,
        "diag": {
            "has_model__proj_ts1": hasattr(model, "_proj_ts1"),
            "has_model__proj_ts2": hasattr(model, "_proj_ts2"),
            "has_model_proj_ts1": hasattr(model, "proj_ts1"),
            "has_model_proj_ts2": hasattr(model, "proj_ts2"),
            "has_model_align2model_ts": hasattr(model, "align2model_ts"),
            "has_model__txt_for_cross1": hasattr(model, "_txt_for_cross1"),
            "has_model__txt_for_cross2": hasattr(model, "_txt_for_cross2"),
            "has_model__pre_cross_txt_to_align": hasattr(model, "_pre_cross_txt_to_align"),
        }
    }
    with open(os.path.join(setting_dir, "evaluation", "load_report.json"), "w", encoding="utf-8") as f:
        json.dump(load_report, f, indent=2, ensure_ascii=False)

    return exp, ckpt


def _safe_read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_csv(path: str, rows: List[Dict[str, Any]]):
    _mkdir(os.path.dirname(path))
    if not rows:
        with open(path, "w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================================================
# Per-setting runner
# ============================================================
def _run_one_setting(setting_dir: str) -> Dict[str, Any]:
    print(f"[eval_suite] Running: {setting_dir}")
    _mkdir(os.path.join(setting_dir, "evaluation"))

    exp, ckpt = _build_exp_from_checkpoint(setting_dir)

    retrieval_summary = None
    grad_summary = None
    noise_summary_text = None

    # 1) Retrieval + t-SNE
    if SuiteSwitches.RUN_RETRIEVAL_TSNE:
        rt_cfg = RetrievalTSNEConfig(
            tsne_perplexity=30.0,
            tsne_random_state=42,
            umap_random_state=42,
            save_npz=True,
            align1_batches=SuiteSwitches.ALIGN1_BATCHES,
            align2_batches=SuiteSwitches.ALIGN2_BATCHES,
            run_cross_inputs=SuiteSwitches.RUN_CROSS_INPUTS,
        )
        retrieval_summary = run_retrieval_tsne(exp, setting_dir, cfg=rt_cfg)
    else:
        print("[eval_suite] SKIP run_retrieval_tsne (SuiteSwitches.RUN_RETRIEVAL_TSNE=False)")

    # 2) Grad consistency
    if SuiteSwitches.RUN_GRAD_CONSISTENCY:
        gc_cfg = GradConsistencyConfig(
            max_batches=50,
            max_params=999999,
        )
        for k in ["use_alignment_1", "use_alignment_2", "use_cross", "cross_mode",
                  "alignment_mode", "alignment_delta", "alignment_tau"]:
            if hasattr(gc_cfg, k):
                setattr(gc_cfg, k, None)
        grad_summary = run_grad_consistency(exp, setting_dir, cfg=gc_cfg)
    else:
        print("[eval_suite] SKIP run_grad_consistency (SuiteSwitches.RUN_GRAD_CONSISTENCY=False)")

    # 3) Noise injection (ONLY drop_text=False)
    if SuiteSwitches.RUN_NOISE_INJECTION:
        ni_cfg_text = NoiseInjectionConfig(
            noise_levels=list(SuiteSwitches.NOISE_LEVELS),
            max_batches=int(SuiteSwitches.NOISE_MAX_BATCHES),
            drop_text=bool(SuiteSwitches.NOISE_DROP_TEXT),
        )
        noise_summary_text = run_noise_injection(exp, setting_dir, cfg=ni_cfg_text)
    else:
        print("[eval_suite] SKIP run_noise_injection (SuiteSwitches.RUN_NOISE_INJECTION=False)")

    summary = {
        "setting_dir": setting_dir,
        "setting_name": _setting_name(setting_dir),
        "group_key": _group_key_from_setting_name(_setting_name(setting_dir)),
        "ckpt": {
            "has_args": True,
            "args_keys": sorted(list(ckpt["args"].__dict__.keys())) if hasattr(ckpt["args"], "__dict__")
            else sorted(list(ckpt["args"].keys())) if isinstance(ckpt["args"], dict) else [],
        },
        "switches": {
            "RUN_RETRIEVAL_TSNE": SuiteSwitches.RUN_RETRIEVAL_TSNE,
            "RUN_GRAD_CONSISTENCY": SuiteSwitches.RUN_GRAD_CONSISTENCY,
            "RUN_NOISE_INJECTION": SuiteSwitches.RUN_NOISE_INJECTION,
            "ALIGN1_BATCHES": SuiteSwitches.ALIGN1_BATCHES,
            "ALIGN2_BATCHES": SuiteSwitches.ALIGN2_BATCHES,
            "RUN_CROSS_INPUTS": SuiteSwitches.RUN_CROSS_INPUTS,
        },
        "retrieval_tsne": retrieval_summary,
        "grad_consistency": grad_summary,
        "noise_injection": {
            "with_text": noise_summary_text,
        },
    }

    with open(os.path.join(setting_dir, "evaluation", "suite_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary



# ============================================================
# Merge helpers: classify ablations by (use_alignment_1, use_alignment_2)
# ============================================================
def _detect_align_flags(setting_dir: str) -> Tuple[Optional[bool], Optional[bool]]:
    """
    Scheme B:
    Read (use_alignment_1, use_alignment_2) directly from checkpoint args,
    no dependency on retrieval_tsne outputs.
    """
    ckpt_path = _find_first_existing([
        os.path.join(setting_dir, "checkpoint_all.pth"),
        os.path.join(setting_dir, "checkpoint.pth"),
    ])
    if ckpt_path is None:
        return None, None

    try:
        ckpt = _load_checkpoint(ckpt_path, device="cpu")
    except Exception:
        return None, None

    args = ckpt.get("args", None)
    if args is None:
        return None, None

    # args can be Namespace-like or dict
    def _get(k: str, default=None):
        if isinstance(args, dict):
            return args.get(k, default)
        return getattr(args, k, default)

    # main expected keys
    a1 = _get("use_alignment_1", None)
    a2 = _get("use_alignment_2", None)

    # optional fallbacks (in case you used other naming in older runs)
    if a1 is None:
        a1 = _get("alignment_1", _get("align1", None))
    if a2 is None:
        a2 = _get("alignment_2", _get("align2", None))

    # normalize to bool/None
    a1 = bool(a1) if a1 is not None else None
    a2 = bool(a2) if a2 is not None else None
    return a1, a2



def _classify_member(setting_dir: str) -> str:
    """
    Returns one of:
    baseline / align1 / align2 / both / unknown
    """
    a1, a2 = _detect_align_flags(setting_dir)
    if a1 is False and a2 is False:
        return "baseline"
    if a1 is True and a2 is False:
        return "align1"
    if a1 is False and a2 is True:
        return "align2"
    if a1 is True and a2 is True:
        return "both"
    return "unknown"


# ============================================================
# Collect rows
# ============================================================
def _collect_retrieval_rows(setting_dir: str, setting_name: str, group_key: str) -> List[Dict[str, Any]]:
    """
    Conservative strategy:
    - Baseline retrieval is no longer required, so we do NOT read retrieval_baseline.json.
    - We still collect align1/align2/cross_inputs if present.
    - mode_detected row is kept for ablation classification.
    """
    out: List[Dict[str, Any]] = []
    base = os.path.join(setting_dir, "evaluation", "retrieval_tsne")
    summ = _safe_read_json(os.path.join(base, "summary.json"))

    def _push(space: str, j: Optional[Dict[str, Any]]):
        if not j:
            return
        row = {
            "group_key": group_key,
            "setting_name": setting_name,
            "space": space,
            "T": j.get("T", None),
            "dim_ts": j.get("dim_ts", None),
            "dim_txt": j.get("dim_txt", None),
            "dim_used_for_sim": j.get("dim_used_for_sim", None),
            "TS2TXT_R@1": (j.get("TS->Text", {}) or {}).get("R@1", None),
            "TS2TXT_MRR": (j.get("TS->Text", {}) or {}).get("MRR", None),
            "TXT2TS_R@1": (j.get("Text->TS", {}) or {}).get("R@1", None),
            "TXT2TS_MRR": (j.get("Text->TS", {}) or {}).get("MRR", None),
        }
        out.append(row)

    # baseline removed
    _push("align1", _safe_read_json(os.path.join(base, "retrieval_align1.json")))
    _push("align2", _safe_read_json(os.path.join(base, "retrieval_align2.json")))
    _push("cross_inputs", _safe_read_json(os.path.join(base, "retrieval_cross_inputs.json")))

    if summ and "mode_detected" in summ:
        out.append({
            "group_key": group_key,
            "setting_name": setting_name,
            "space": "mode_detected",
            "use_alignment_1": (summ.get("mode_detected") or {}).get("use_alignment_1", None),
            "use_alignment_2": (summ.get("mode_detected") or {}).get("use_alignment_2", None),
        })

    return out


def _collect_grad_rows(setting_dir: str, setting_name: str, group_key: str) -> List[Dict[str, Any]]:
    base = os.path.join(setting_dir, "evaluation", "grad_consistency")
    summ = _safe_read_json(os.path.join(base, "summary.json"))
    if not summ:
        return []
    return [{
        "group_key": group_key,
        "setting_name": setting_name,
        "batches_used": summ.get("batches_used", None),
        "num_param_cosines": summ.get("num_param_cosines", None),
        "cosine_mean": summ.get("cosine_mean", None),
        "cosine_std": summ.get("cosine_std", None),
        "overall_cosine_mean": summ.get("overall_cosine_mean", None),
        "overall_cosine_std": summ.get("overall_cosine_std", None),
    }]


def _collect_noise_rows(setting_dir: str, setting_name: str, group_key: str) -> List[Dict[str, Any]]:
    """
    ONLY with_text (drop_text=False)
    expects:
      evaluation/noise_injection/drop_text_False/summary.json
    """
    out: List[Dict[str, Any]] = []
    base = os.path.join(setting_dir, "evaluation", "noise_injection")

    p = os.path.join(base, "drop_text_False", "summary.json")
    summ = _safe_read_json(p)
    if not summ or "results" not in summ:
        return out

    for r in summ["results"]:
        out.append({
            "group_key": group_key,
            "setting_name": setting_name,
            "variant": "with_text",
            "noise_level": r.get("noise_level", None),
            "mse_mean": r.get("mse_mean", None),
            "mse_std": r.get("mse_std", None),
            "batches_used": r.get("batches_used", None),
            "drop_text": r.get("drop_text", None),
        })
    return out


# ============================================================
# Gallery making
# ============================================================
def _try_import_pil():
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
        return Image, ImageDraw, ImageFont
    except Exception:
        return None, None, None


def _find_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.isfile(p):
            return p
    return None


def _make_grad_hist_row_for_group(group_out_dir: str, ablation_map: Dict[str, Dict[str, Any]]):
    """
    Create 3-row, 1-col histogram figure:
      top:    align1
      middle: align2
      bottom: both

    - Read per-param cosine values from:
        evaluation/grad_consistency/grad_cosine_per_param.csv
      (so we can enforce same x-axis scale/ticks/bins)
    - Same x scale and tick interval across all 3 plots
    - Title AlignLoss wording:
        align1 -> Semantic AlignLoss
        align2 -> Movement AlignLoss
        both   -> Dual AlignLoss
    - Mark mean/std on x-axis (xlabel) and draw vlines at mean and mean±std
    """
    import csv
    import numpy as np
    import matplotlib.pyplot as plt

    order = ["align1", "align2", "both"]
    title_map = {
        "align1": "Grad cosine (MSE vs Semantic AlignLoss) per-param",
        "align2": "Grad cosine (MSE vs Movement AlignLoss) per-param",
        "both":   "Grad cosine (MSE vs Dual AlignLoss) per-param",
    }

    # ----- fixed x-axis scale & interval for easy comparison -----
    x_min, x_max = -1.0, 1.0
    bins = 60
    xticks = np.linspace(-1.0, 1.0, 9)  # -1.00, -0.75, ... , 1.00

    def _read_cosines(csv_path: str) -> np.ndarray:
        vals = []
        with open(csv_path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                v = row.get("cosine", None)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if np.isfinite(fv):
                    vals.append(fv)
        return np.asarray(vals, dtype=np.float32)

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8.2, 10.0), dpi=200)

    missing_msgs = []
    any_ok = False

    for ax, key in zip(axes, order):
        m = ablation_map.get(key, None)
        if not m:
            ax.text(0.5, 0.5, "MISSING (no member)", ha="center", va="center")
            ax.set_title(title_map.get(key, key))
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(xticks)
            missing_msgs.append(f"{key}: not found in ablation_map")
            continue

        sd = m["setting_dir"]
        p = os.path.join(sd, "evaluation", "grad_consistency", "grad_cosine_per_param.csv")
        if not os.path.isfile(p):
            ax.text(0.5, 0.5, "MISSING (no CSV)", ha="center", va="center")
            ax.set_title(title_map.get(key, key))
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(xticks)
            missing_msgs.append(f"{key}: missing grad_cosine_per_param.csv")
            continue

        cos = _read_cosines(p)
        if cos.size == 0:
            ax.text(0.5, 0.5, "EMPTY (no cosine rows)", ha="center", va="center")
            ax.set_title(title_map.get(key, key))
            ax.set_xlim(x_min, x_max)
            ax.set_xticks(xticks)
            missing_msgs.append(f"{key}: empty cosine data")
            continue

        mu = float(np.mean(cos))
        sdv = float(np.std(cos))

        ax.hist(cos, bins=bins, range=(x_min, x_max))
        ax.axvline(mu, linestyle="--")
        ax.axvline(mu - sdv, linestyle=":")
        ax.axvline(mu + sdv, linestyle=":")

        ax.set_title(title_map.get(key, key))
        ax.set_xlim(x_min, x_max)
        ax.set_xticks(xticks)
        ax.set_xlabel(f"cosine   mean={mu:.4f}   std={sdv:.4f}")

        any_ok = True

    plt.tight_layout()
    if any_ok:
        out_path = os.path.join(group_out_dir, "grad_hist_row.png")
        plt.savefig(out_path)
    else:
        with open(os.path.join(group_out_dir, "grad_hist_row_skipped.txt"), "w", encoding="utf-8") as f:
            f.write("No valid grad_consistency CSV found. Nothing plotted.\n")
    plt.close()

    if missing_msgs:
        with open(os.path.join(group_out_dir, "grad_hist_row_missing.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(missing_msgs) + "\n")


def _make_noise_gallery_for_group(group_out_dir: str, ablation_map: Dict[str, Dict[str, Any]]):
    """
    Single plot with 4 lines (drop_text=False only):
      baseline, align1, align2, both
    Legend labels:
      baseline -> baseline
      align1   -> only semantic alignment
      align2   -> only movement alignment
      both     -> dual alignment
    """
    import json
    import matplotlib.pyplot as plt

    order = ["baseline", "align1", "align2", "both"]
    label_map = {
        "baseline": "baseline",
        "align1": "only SA",
        "align2": "only MA",
        "both": "DA",
    }

    missing: List[str] = []

    plt.figure(figsize=(7.5, 5.2), dpi=200)

    any_plotted = False
    for key in order:
        m = ablation_map.get(key, None)
        if not m:
            missing.append(f"{key}: not found in ablation_map")
            continue

        sd = m["setting_dir"]
        summ_path = os.path.join(sd, "evaluation", "noise_injection", "drop_text_False", "summary.json")
        if not os.path.isfile(summ_path):
            missing.append(f"{key}: missing summary.json")
            continue

        try:
            with open(summ_path, "r", encoding="utf-8") as f:
                summ = json.load(f)
            results = summ.get("results", [])
            xs = [r.get("noise_level", None) for r in results]
            ys = [r.get("mse_mean", None) for r in results]

            xy = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
            if not xy:
                missing.append(f"{key}: empty results")
                continue

            xs2, ys2 = zip(*xy)
            plt.plot(xs2, ys2, marker="o", label=label_map.get(key, key))
            any_plotted = True
        except Exception as e:
            missing.append(f"{key}: load fail ({repr(e)})")

    plt.xlabel("noise_level (sigma = level * std(x))")
    plt.ylabel("forecast MSE (mean)")
    plt.title("Noise Injection Robustness")
    plt.tight_layout()

    if any_plotted:
        plt.legend()
        out_path = os.path.join(group_out_dir, "noise_gallery.png")
        plt.savefig(out_path)
    else:
        with open(os.path.join(group_out_dir, "noise_gallery_skipped.txt"), "w", encoding="utf-8") as f:
            f.write("No valid noise_injection summaries found. Nothing plotted.\n")

    plt.close()

    if missing:
        with open(os.path.join(group_out_dir, "noise_gallery_missing.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(missing) + "\n")

def _make_umap_gallery_for_group(group_out_dir: str, ablation_map: Dict[str, Dict[str, Any]]):
    """
    UMAP gallery layout (8 images total, 2 per row), same as tsne_gallery:
      Row 1: only semantic alignment  -> member 'align1' : (align1 TS time, align1 TXT time)
      Row 2: only movement alignment  -> member 'align2' : (align2 TS movement, align2 TXT movement)
      Row 3: dual alignment           -> member 'both'   : (align1 TS time, align1 TXT time)
      Row 4: dual alignment           -> member 'both'   : (align2 TS movement, align2 TXT movement)

    Notes:
    - baseline not used
    - ONLY use umap_*.png (no fallback to tsne)
    - Output: <group_out_dir>/umap_gallery.png
    """
    Image, ImageDraw, ImageFont = _try_import_pil()
    if Image is None:
        with open(os.path.join(group_out_dir, "umap_gallery_skipped.txt"), "w", encoding="utf-8") as f:
            f.write("PIL not available; skipped umap gallery.\n")
        return

    def _umap_path(setting_dir: str, space: str, mod: str, color: str) -> Optional[str]:
        base = os.path.join(setting_dir, "evaluation", "retrieval_tsne")
        umap_name = f"umap_{space}_{mod}_{color}.png"
        p = os.path.join(base, umap_name)
        return p if os.path.isfile(p) else None

    rows = []

    # Row 1: only semantic alignment (align1 member)
    m_align1 = ablation_map.get("align1", None)
    if m_align1 is not None:
        sd = m_align1["setting_dir"]
        rows.append((
            "only semantic alignment",
            [
                _umap_path(sd, "align1", "TS", "time"),
                _umap_path(sd, "align1", "TXT", "time"),
            ],
        ))
    else:
        rows.append(("only semantic alignment", [None, None]))

    # Row 2: only movement alignment (align2 member)
    m_align2 = ablation_map.get("align2", None)
    if m_align2 is not None:
        sd = m_align2["setting_dir"]
        rows.append((
            "only movement alignment",
            [
                _umap_path(sd, "align2", "TS", "movement"),
                _umap_path(sd, "align2", "TXT", "movement"),
            ],
        ))
    else:
        rows.append(("only movement alignment", [None, None]))

    # Rows 3 & 4: dual alignment (both member)
    m_both = ablation_map.get("both", None)
    if m_both is not None:
        sd = m_both["setting_dir"]
        rows.append((
            "dual alignment",
            [
                _umap_path(sd, "align1", "TS", "time"),
                _umap_path(sd, "align1", "TXT", "time"),
            ],
        ))
        rows.append((
            "dual alignment",
            [
                _umap_path(sd, "align2", "TS", "movement"),
                _umap_path(sd, "align2", "TXT", "movement"),
            ],
        ))
    else:
        rows.append(("dual alignment", [None, None]))
        rows.append(("dual alignment", [None, None]))

    # ---- Layout params (same as tsne_gallery) ----
    n_rows = len(rows)  # 4
    n_cols = 2

    cell_w = 620
    cell_h = 480
    pad = 18
    title_h = 60
    row_label_w = 320

    W = row_label_w + pad + n_cols * (cell_w + pad) + pad
    H = title_h + pad + n_rows * (cell_h + pad) + pad

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Column headers
    draw.text((row_label_w + pad + 0 * (cell_w + pad), pad), "TS", fill=(0, 0, 0), font=font)
    draw.text((row_label_w + pad + 1 * (cell_w + pad), pad), "TXT", fill=(0, 0, 0), font=font)

    # Rows content
    for r, (row_title, img_paths) in enumerate(rows):
        y = title_h + pad + r * (cell_h + pad)

        # Row label
        draw.text((pad, y + 8), row_title, fill=(0, 0, 0), font=font)

        # Two images per row
        for c in range(n_cols):
            x = row_label_w + pad + c * (cell_w + pad)
            p = img_paths[c] if c < len(img_paths) else None

            if p is None:
                draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(0, 0, 0), width=2)
                draw.text((x + 12, y + 12), "MISSING", fill=(200, 0, 0), font=font)
                continue

            try:
                im = Image.open(p).convert("RGB")
                im = im.resize((cell_w, cell_h))
                canvas.paste(im, (x, y))
            except Exception:
                draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(0, 0, 0), width=2)
                draw.text((x + 12, y + 12), "LOAD FAIL", fill=(200, 0, 0), font=font)

    canvas.save(os.path.join(group_out_dir, "umap_gallery.png"))


def _make_tsne_gallery_for_group(group_out_dir: str, ablation_map: Dict[str, Dict[str, Any]]):
    """
    New gallery layout (8 images total, 2 per row):
      Row 1: only semantic alignment  -> from member 'align1' : (align1 TS time, align1 TXT time)
      Row 2: only movement alignment  -> from member 'align2' : (align2 TS movement, align2 TXT movement)
      Row 3: dual alignment           -> from member 'both'   : (align1 TS time, align1 TXT time)
      Row 4: dual alignment           -> from member 'both'   : (align2 TS movement, align2 TXT movement)

    Notes:
    - baseline not used here
    - use tsne_*.png if exists, else fallback to umap_*.png
    """
    Image, ImageDraw, ImageFont = _try_import_pil()
    if Image is None:
        with open(os.path.join(group_out_dir, "tsne_gallery_skipped.txt"), "w", encoding="utf-8") as f:
            f.write("PIL not available; skipped gallery.\n")
        return

    def _img_path(setting_dir: str, space: str, mod: str, color: str) -> Optional[str]:
        base = os.path.join(setting_dir, "evaluation", "retrieval_tsne")
        tsne_name = f"tsne_{space}_{mod}_{color}.png"
        umap_name = f"umap_{space}_{mod}_{color}.png"
        return _find_first_existing([os.path.join(base, tsne_name), os.path.join(base, umap_name)])

    # ---- Build 4 rows, each row contains exactly 2 images ----
    rows = []

    # Row 1: only semantic alignment (align1 member)
    m_align1 = ablation_map.get("align1", None)
    if m_align1 is not None:
        sd = m_align1["setting_dir"]
        rows.append((
            "only semantic alignment",
            [
                _img_path(sd, "align1", "TS", "time"),
                _img_path(sd, "align1", "TXT", "time"),
            ],
        ))
    else:
        rows.append(("only semantic alignment", [None, None]))

    # Row 2: only movement alignment (align2 member)
    m_align2 = ablation_map.get("align2", None)
    if m_align2 is not None:
        sd = m_align2["setting_dir"]
        rows.append((
            "only movement alignment",
            [
                _img_path(sd, "align2", "TS", "movement"),
                _img_path(sd, "align2", "TXT", "movement"),
            ],
        ))
    else:
        rows.append(("only movement alignment", [None, None]))

    # Rows 3 & 4: dual alignment (both member)
    m_both = ablation_map.get("both", None)
    if m_both is not None:
        sd = m_both["setting_dir"]
        rows.append((
            "dual alignment",
            [
                _img_path(sd, "align1", "TS", "time"),
                _img_path(sd, "align1", "TXT", "time"),
            ],
        ))
        rows.append((
            "dual alignment",
            [
                _img_path(sd, "align2", "TS", "movement"),
                _img_path(sd, "align2", "TXT", "movement"),
            ],
        ))
    else:
        rows.append(("dual alignment", [None, None]))
        rows.append(("dual alignment", [None, None]))

    # ---- Layout params ----
    n_rows = len(rows)  # should be 4
    n_cols = 2

    cell_w = 620
    cell_h = 480
    pad = 18
    title_h = 60
    row_label_w = 320

    W = row_label_w + pad + n_cols * (cell_w + pad) + pad
    H = title_h + pad + n_rows * (cell_h + pad) + pad

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # Column headers
    draw.text((row_label_w + pad + 0 * (cell_w + pad), pad), "TS", fill=(0, 0, 0), font=font)
    draw.text((row_label_w + pad + 1 * (cell_w + pad), pad), "TXT", fill=(0, 0, 0), font=font)

    # Rows content
    for r, (row_title, img_paths) in enumerate(rows):
        y = title_h + pad + r * (cell_h + pad)

        # Row label
        draw.text((pad, y + 8), row_title, fill=(0, 0, 0), font=font)

        # Two images per row
        for c in range(n_cols):
            x = row_label_w + pad + c * (cell_w + pad)
            p = img_paths[c] if c < len(img_paths) else None

            if p is None:
                draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(0, 0, 0), width=2)
                draw.text((x + 12, y + 12), "MISSING", fill=(200, 0, 0), font=font)
                continue

            try:
                im = Image.open(p).convert("RGB")
                im = im.resize((cell_w, cell_h))
                canvas.paste(im, (x, y))
            except Exception:
                draw.rectangle([x, y, x + cell_w, y + cell_h], outline=(0, 0, 0), width=2)
                draw.text((x + 12, y + 12), "LOAD FAIL", fill=(200, 0, 0), font=font)

    canvas.save(os.path.join(group_out_dir, "tsne_gallery.png"))



# ============================================================
# Group merge
# ============================================================
def _merge_groups(checkpoints_root: str, per_setting_summaries: List[Dict[str, Any]]):
    merged_root = os.path.join(checkpoints_root, "merged")
    _mkdir(merged_root)

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for s in per_setting_summaries:
        gk = s.get("group_key", None) or _group_key_from_setting_name(s.get("setting_name", ""))
        groups.setdefault(gk, []).append(s)

    for gk, members in groups.items():
        members = sorted(members, key=lambda x: x.get("setting_name", ""))

        group_out = os.path.join(merged_root, gk)
        _mkdir(group_out)

        retrieval_rows: List[Dict[str, Any]] = []
        grad_rows: List[Dict[str, Any]] = []
        noise_rows: List[Dict[str, Any]] = []

        ablation_map: Dict[str, Dict[str, Any]] = {}
        for m in members:
            sd = m["setting_dir"]
            cls = _classify_member(sd)
            if cls not in ablation_map:
                ablation_map[cls] = m

        for m in members:
            sd = m["setting_dir"]
            name = m.get("setting_name", _setting_name(sd))
            retrieval_rows.extend(_collect_retrieval_rows(sd, name, gk))
            grad_rows.extend(_collect_grad_rows(sd, name, gk))
            noise_rows.extend(_collect_noise_rows(sd, name, gk))

        _write_csv(os.path.join(group_out, "retrieval_table.csv"), retrieval_rows)
        _write_csv(os.path.join(group_out, "grad_consistency_table.csv"), grad_rows)
        _write_csv(os.path.join(group_out, "noise_injection_table.csv"), noise_rows)

        _make_tsne_gallery_for_group(group_out, ablation_map)
        _make_umap_gallery_for_group(group_out, ablation_map)
        _make_grad_hist_row_for_group(group_out, ablation_map)
        _make_noise_gallery_for_group(group_out, ablation_map)

        with open(os.path.join(group_out, "merged_summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "group_key": gk,
                    "num_members": len(members),
                    "members": [m.get("setting_name", "") for m in members],
                    "ablation_map": {k: v.get("setting_name", "") for k, v in ablation_map.items()},
                    "paths": {
                        "retrieval_table": os.path.join(group_out, "retrieval_table.csv"),
                        "grad_table": os.path.join(group_out, "grad_consistency_table.csv"),
                        "noise_table": os.path.join(group_out, "noise_injection_table.csv"),
                        "tsne_gallery": os.path.join(group_out, "tsne_gallery.png"),
                        "umap_gallery": os.path.join(group_out, "umap_gallery.png"),
                        "grad_hist_row": os.path.join(group_out, "grad_hist_row.png"),
                        "noise_gallery": os.path.join(group_out, "noise_gallery.png"),
                    }
                },
                f,
                indent=2,
                ensure_ascii=False,
            )


# ============================================================
# Main
# ============================================================
def main(single_setting_dir: Optional[str] = None):
    checkpoints_root = os.path.abspath(os.path.join(HERE, "..", "ablation"))

    if single_setting_dir is not None and str(single_setting_dir).strip():
        setting_dirs = [os.path.abspath(single_setting_dir)]
    else:
        setting_dirs = _find_setting_dirs(checkpoints_root)

    if not setting_dirs:
        raise RuntimeError(f"No setting dirs found. Expect checkpoint_all.pth under: {checkpoints_root}")

    all_summaries: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []

    for sd in setting_dirs:
        try:
            all_summaries.append(_run_one_setting(sd))
        except Exception as e:
            tb = traceback.format_exc()
            failed.append({"setting_dir": sd, "error": repr(e)})

            print(f"[eval_suite] FAILED: {sd} -> {repr(e)}")
            print(tb)

            try:
                _mkdir(os.path.join(sd, "evaluation"))
                with open(os.path.join(sd, "evaluation", "suite_failed.json"), "w", encoding="utf-8") as f:
                    json.dump({"error": repr(e), "traceback": tb}, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

    out_path = os.path.join(checkpoints_root, "evaluation_all_summary.json")
    _mkdir(checkpoints_root)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_settings": len(setting_dirs),
                "num_succeeded": len(all_summaries),
                "num_failed": len(failed),
                "failed": failed,
                "summaries": all_summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    _merge_groups(checkpoints_root, all_summaries)

    print(f"[eval_suite] Done. Wrote: {out_path}")
    print(f"[eval_suite] Merged outputs in: {os.path.join(checkpoints_root, 'merged')}")


if __name__ == "__main__":
    SINGLE_SETTING_DIR = ""  # or r"./abl_test/your_setting_dir"
    main(SINGLE_SETTING_DIR)
