import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class MLP(nn.Module):
    def __init__(self, layer_sizes, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_rate)
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)
        return x

class MultiVarIndepWrapper(nn.Module):
    """
    Wrap N single-var models into an independent multi-var model.
    - batch_x/batch_y: [B, T, N]
    - each submodel consumes [B, T, 1]
    - marks/text are shared across variables
    - outputs are concatenated back to [B, T_out, N]
    """
    def __init__(self, base_model_cls, args, n_vars: int):
        super().__init__()
        assert n_vars >= 1
        self.n_vars = int(n_vars)
        self.args = args

        models = []
        for _ in range(self.n_vars):
            a = copy.deepcopy(args)
            # Force single-var IO for submodel
            a.enc_in = 1
            if hasattr(a, "c_out"):
                a.c_out = 1
            if hasattr(a, "dec_in"):
                a.dec_in = 1
            models.append(base_model_cls(a).float())
        self.models = nn.ModuleList(models)

    def forward(
        self,
        batch_x, batch_x_mark, dec_inp, batch_y_mark,
        **kwargs
    ):
        # kwargs contains: z_text, use_alignment_1, alignment_mode, alignment_delta, alignment_tau,
        # use_alignment_2, y_true, use_cross, cross_mode, ...
        y_true_full = kwargs.get("y_true", None)

        outs = []
        align_losses = []

        for v in range(self.n_vars):
            bx = batch_x[..., v:v+1]
            di = dec_inp[..., v:v+1] if dec_inp is not None else None

            # per-var y_true for alignment if provided
            if y_true_full is not None:
                kwargs_v = dict(kwargs)
                kwargs_v["y_true"] = y_true_full[..., v:v+1]
            else:
                kwargs_v = kwargs

            out = self.models[v](bx, batch_x_mark, di, batch_y_mark, **kwargs_v)

            if isinstance(out, tuple):
                yhat, al = out
                outs.append(yhat)
                align_losses.append(al)
            else:
                outs.append(out)

        outputs = torch.cat(outs, dim=-1)

        if len(align_losses) > 0:
            # average alignment losses across variables
            # (keep dtype/device)
            al_stack = []
            for al in align_losses:
                if al is None:
                    continue
                if not torch.is_tensor(al):
                    al = torch.tensor(al, device=outputs.device, dtype=outputs.dtype)
                al_stack.append(al)
            if len(al_stack) > 0:
                align_loss = torch.stack(al_stack).mean()
                return outputs, align_loss
            else:
                return outputs, None

        return outputs