# exp/exp_tools/tensor_ops.py

from __future__ import annotations

import torch
import torch.nn.functional as F


def norm(input_emb: torch.Tensor) -> torch.Tensor:
    """
    和你原来完全一致：
    input_emb: [..., D]（你常用的是 [B, T, 1] 或 [B, D, 1]）
    """
    input_emb = input_emb - input_emb.mean(1, keepdim=True).detach()
    input_emb = input_emb / torch.sqrt(
        torch.var(input_emb, dim=1, keepdim=True, unbiased=False) + 1e-5
    )
    return input_emb


def info_nce_loss(z_time: torch.Tensor, z_text: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """用于对比损失；若模型不返回对应向量则不会被调用"""
    z_time = F.normalize(z_time, dim=-1)
    z_text = F.normalize(z_text, dim=-1)
    logits = z_time @ z_text.t() / tau
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def apply_time_mask(x: torch.Tensor, ratio: float) -> torch.Tensor:
    """对输入 batch_x 在时间维随机掩码（不改形状）；x:[B, L, C]"""
    if ratio <= 0.0:
        return x
    B, L, C = x.shape
    mask = (torch.rand(B, L, device=x.device) > ratio).float().unsqueeze(-1)  # [B, L, 1]
    return x * mask


def align_dec_mark(args, dec_mark: torch.Tensor, target_pred: int) -> torch.Tensor:
    """把解码端时间标记对齐到 label_len + target_pred。"""
    need = args.label_len + target_pred
    cur = dec_mark.shape[1]
    if cur == need:
        return dec_mark
    elif cur > need:
        return dec_mark[:, :need, :]
    else:
        pad = dec_mark[:, -1:, :].repeat(1, need - cur, 1)
        return torch.cat([dec_mark, pad], dim=1)


def time_align_text_emb(txt_emb: torch.Tensor | None, target_len: int) -> torch.Tensor | None:
    """
    将文本嵌入在时间维对齐到 target_len：
    - 若 L==target_len：原样返回
    - 若 L==1：直接 repeat
    - 其他 L：用线性插值到 target_len
    """
    if txt_emb is None:
        return None
    B, L, D = txt_emb.shape
    if L == target_len:
        return txt_emb
    if L == 1:
        return txt_emb.repeat(1, target_len, 1)
    return F.interpolate(
        txt_emb.transpose(1, 2), size=target_len, mode="linear", align_corners=True
    ).transpose(1, 2)


def feat_slice(args):
    """返回特征切片：S -> slice(0,1)；M -> slice(None)"""
    return slice(None) if getattr(args, "features", "S") == "M" else slice(0, 1)
