# exp/exp_tools/dataset_ops.py

from __future__ import annotations

import torch


def get_prior_y_safe(exp, dataset, index):
    """
    从 dataset 安全地拿 prior_y：
    - 若数据集没有 prior（比如 features='M'），返回 None
    - 否则返回已经放到正确 device 的 torch.Tensor
    期望形状: [B, pred_len, 1]
    """
    prior_np = None
    try:
        prior_np = dataset.get_prior_y(index)
    except Exception:
        prior_np = None

    if prior_np is None:
        return None

    return torch.from_numpy(prior_np).float().to(exp.device)


def get_text_inputs(dataset, index):
    """
    统一你 train/vali/test 的“拿文本”逻辑，避免重复三遍。
    返回: (hist_text, fut_text)
    - 如果 dataset 有 get_text_pair(index)，优先用它
    - 否则用 dataset.get_text(index)，并尽量取 tmp[:,1] 作为 hist_text（兼容你的实现）
    """
    hist_text, fut_text = None, None

    if hasattr(dataset, "get_text_pair"):
        try:
            hist_text, fut_text = dataset.get_text_pair(index)
        except Exception:
            hist_text, fut_text = None, None

    if hist_text is None:
        tmp = dataset.get_text(index)
        try:
            hist_text = tmp[:, 1]
        except Exception:
            hist_text = tmp

    return hist_text, fut_text
