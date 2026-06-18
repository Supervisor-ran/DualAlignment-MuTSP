# exp/exp_tools/optim_ops.py

from __future__ import annotations

import torch.nn as nn
from torch import optim


def select_optimizer(exp):
    return optim.Adam(exp.model.parameters(), lr=exp.args.learning_rate)


def select_optimizer_mlp(exp):
    lr = getattr(exp.args, "learning_rate2", getattr(exp, "learning_rate2", 1e-2))
    params = [p for p in exp.mlp.parameters() if p.requires_grad]
    return optim.Adam(params, lr=lr) if len(params) > 0 else None


def select_optimizer_proj(exp):
    lr = getattr(exp.args, "learning_rate3", getattr(exp, "learning_rate3", 1e-3))
    return optim.Adam(exp.mlp_proj.parameters(), lr=lr)


def select_optimizer_weight(exp):
    return optim.Adam(
        [{"params": exp.weight1.parameters()}, {"params": exp.weight2.parameters()}],
        lr=exp.args.learning_rate_weight
    )


def select_criterion(exp):
    if exp.args.loss == "MSE":
        return nn.MSELoss()
    raise ValueError("Unsupported loss function")
