import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe

def metric_masked(preds: np.ndarray, trues: np.ndarray, eps: float = 1e-8):
    """
    只在有效（finite）位置计算指标；跳过真值为0的点（用于MAPE/MSPE）。
    形状支持 (..., T, C)。返回 mae, mse, rmse, mape, mspe
    """
    # 基本检查
    if preds.shape != trues.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape} vs trues {trues.shape}")

    # 有效性掩码：都为有限数
    valid = np.isfinite(preds) & np.isfinite(trues)
    total_valid = int(valid.sum())

    if total_valid == 0:
        # 完全没有有效点，给出明确错误
        raise ValueError("Both preds and trues have no finite values to evaluate.")

    # 如果有样本是“全无效”，将其剔除，避免 reshape 后污染
    B = preds.shape[0]
    flat_valid_per_sample = valid.reshape(B, -1).sum(axis=1)
    keep = flat_valid_per_sample > 0
    dropped = int((~keep).sum())
    if dropped:
        print(f"[metric_masked] Dropped {dropped} samples with 0 valid points.")

    preds = preds[keep]
    trues = trues[keep]
    valid = valid[keep]

    # ===== MAE / MSE / RMSE（仅在 valid 上）=====
    diff = preds - trues
    mae = np.abs(diff)[valid].mean()

    mse = (diff[valid] ** 2).mean()
    rmse = np.sqrt(mse)

    # ===== MAPE / MSPE（额外要求真值非0）=====
    denom_mask = valid & (np.abs(trues) > eps)

    if denom_mask.any():
        rel_err = (diff[denom_mask] / trues[denom_mask])
        mape = np.abs(rel_err).mean() * 100.0  # 常见定义以百分比表示
        mspe = (rel_err ** 2).mean() * 100.0
    else:
        # 如果没有任何真值非0的有效点，就返回 NaN 并给出提示
        print("[metric_masked] No non-zero ground-truth points for MAPE/MSPE; returning NaN for these.")
        mape, mspe = np.nan, np.nan

    return mae, mse, rmse, mape, mspe
