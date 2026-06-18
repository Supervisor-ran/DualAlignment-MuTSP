import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from torch.ao.quantization import fused_wt_fake_quant_range_neg_127_to_127


# ==========================
# Helpers
# ==========================
class _SharedProj(nn.Module):
    def __init__(self, d_ts, d_txt, d_shared=None):
        super().__init__()

        def _to_int(x, name):
            # 展开 list/tuple 这种误传
            if isinstance(x, (list, tuple)):
                if len(x) != 1:
                    raise TypeError(f"{name} must be int, got {type(x)} with length {len(x)}: {x}")
                x = x[0]
            # numpy 标量
            if isinstance(x, np.generic):
                x = np.asarray(x).item()
            # tensor 标量
            if torch.is_tensor(x):
                x = x.item()
            # 字符串数字
            if isinstance(x, str):
                x = x.strip()
                if x.endswith(','):
                    x = x[:-1]
                x = int(x)
            # 最终转 int
            x = int(x)
            if x <= 0:
                raise ValueError(f"{name} must be > 0, got {x}")
            return x

        d_ts = _to_int(d_ts, "d_ts")
        d_txt = _to_int(d_txt, "d_txt")
        d_shared = _to_int(d_shared if d_shared is not None else d_ts, "d_shared")

        self.p_ts = nn.Linear(d_ts, d_shared, bias=True)  # in=d_ts, out=d_shared
        self.p_txt = nn.Linear(d_txt, d_shared, bias=True)  # in=d_txt, out=d_shared
        self.ln_ts = nn.LayerNorm(d_shared)
        self.ln_txt = nn.LayerNorm(d_shared)

    def forward(self, h_ts, h_txt):
        h_ts = F.normalize(self.ln_ts(self.p_ts(h_ts)), dim=-1)
        h_txt = F.normalize(self.ln_txt(self.p_txt(h_txt)), dim=-1)
        return h_ts, h_txt


class _Text2PredHead(nn.Module):
    """ 文本侧跨模态特征 -> [B, pred_len] 的 MLP 头；外部再沿通道维复制到 [B, pred_len, N] """

    def __init__(self, d_model: int, pred_len: int, hidden: int = None, dropout: float = 0.0):
        super().__init__()
        hidden = hidden or max(128, d_model // 2)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, pred_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D]
        return self.net(x)  # [B, pred_len]


class _CrossModality(nn.Module):
    """ 双向跨模态注意力（带残差+LayerNorm）。
    模式：
      - 'ts-to-text'     : 返回 ts<-text 的更新（单张量）
      - 'text-to-ts'     : 返回 text<-ts 的更新（单张量）
      - 'average'        : 双向各算一次，返回 (out_ts + out_txt)/2
      - 'sum'            : 双向各算一次，返回 out_ts + out_txt
      - 'seperate_cross' : 同时返回 (out_ts, out_txt)，供外部分别使用
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.mha_t2x = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mha_x2t = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_ts = nn.LayerNorm(d_model)
        self.ln_txt = nn.LayerNorm(d_model)

    def forward(
            self,
            h_ts: torch.Tensor,
            h_txt: torch.Tensor,
            mode: str = 'ts-to-text',
            attn_mask_ts=None,
            attn_mask_txt=None
    ):
        # ts <- text
        out_ts, _ = self.mha_t2x(h_ts, h_txt, h_txt, attn_mask=attn_mask_ts)
        out_ts = self.ln_ts(h_ts + out_ts)

        # text <- ts
        out_txt, _ = self.mha_x2t(h_txt, h_ts, h_ts, attn_mask=attn_mask_txt)
        out_txt = self.ln_txt(h_txt + out_txt)

        if mode == 'ts-to-text':
            return out_ts
        if mode == 'text-to-ts':
            return out_txt
        if mode == 'average':
            return 0.5 * (out_ts + out_txt)
        if mode == 'sum':
            return out_ts + out_txt
        if mode == 'seperate_cross':
            return out_ts, out_txt
        raise ValueError(f"Unknown cross_modality mode: {mode}")


# ==========================
# Alignment losses
# ==========================
def _cosine_align_loss(h1: torch.Tensor, h2: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """ 最简单余弦对齐损失：
        L = mean_{b,t} [ 1 - <h1,h2> ] ，假设 h1,h2 已归一化。
    """
    loss = 1.0 - (h1 * h2).sum(dim=-1)  # [B, T]
    if mask is not None:
        loss = loss[mask > 0]
    return loss.mean()


def _info_nce_temporal(
        h1: torch.Tensor,
        h2: torch.Tensor,
        delta: int = 0,
        tau: float = 0.07,
        mask: torch.Tensor = None
) -> torch.Tensor:
    """ 时间感知 InfoNCE：h1,h2 为 [B,T,D]（已归一化），同一 batch 内 |t-t'|<=delta 视为正对。 """
    B, T, D = h1.shape
    N = B * T
    H1 = h1.reshape(N, D)
    H2 = h2.reshape(N, D)
    sim = (H1 @ H2.t()) / tau  # [N,N]

    # 构造块对角正样本掩码
    pos = H1.new_zeros((N, N), dtype=torch.bool)
    time_idx = torch.arange(T, device=H1.device)
    time_mask = (time_idx[None, :] - time_idx[:, None]).abs() <= int(delta)
    for b in range(B):
        s = b * T
        pos[s:s + T, s:s + T] = time_mask

    if mask is not None:
        m = (mask.reshape(N, 1) > 0) & (mask.reshape(1, N) > 0)
        pos = pos & m

    log_num = torch.logsumexp(sim.masked_fill(~pos, float('-inf')), dim=1)
    log_den = torch.logsumexp(sim, dim=1)
    return (-(log_num - log_den)).mean()


def _movement_labels_from_y(y_true: torch.Tensor, pred_len: int) -> torch.Tensor:
    """ 基于 label y 生成每个变量的 movement 标签（-1/0/+1）
        y_true: [B, L_y, N]（通常传 batch_y）
        取最后 pred_len 段做一阶差分，按时间平均符号得到 [B, N] 的 {-1,0,1}
    """
    # 只取预测窗口的末尾 pred_len
    y_win = y_true[:, -pred_len:, :]  # [B, pred_len, N]
    if y_win.shape[1] < 2:  # 不足以做差分，全部置 0
        return torch.zeros(y_win.shape[0], y_win.shape[2], dtype=torch.long, device=y_win.device)

    diff = y_win[:, 1:, :] - y_win[:, :-1, :]  # [B, pred_len-1, N]
    signs = torch.sign(diff)  # {-1,0,+1}
    agg = signs.mean(dim=1)  # [B, N]，按时间平均

    labels = torch.where(
        agg > 0,
        torch.ones_like(agg, dtype=torch.long),
        torch.where(agg < 0, -torch.ones_like(agg, dtype=torch.long),
                    torch.zeros_like(agg, dtype=torch.long))
    )
    # [B, N] in {-1,0,1}
    return labels


def _supcon_tokens(h: torch.Tensor, labels: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """ 监督式对比（按 token 维，通常是变量数 N）：
        h: [B, T, D]（已归一化或未归一化，这里会再归一）
        labels: [B, T] in {-1,0,+1}，同 label 且非 0 为正对
        输出标量 loss（批内平均）
    """
    B, T, D = h.shape
    labels = labels.to(h.device)

    loss_sum = 0.0
    valid_batches = 0
    eye = torch.eye(T, dtype=torch.bool, device=h.device)

    for b in range(B):
        lbl = labels[b]  # [T]

        # 正样本：同 label 且 != 0，且去掉对角
        pos_mask = (lbl[:, None] == lbl[None, :]) & (lbl[:, None] != 0)
        pos_mask = pos_mask & (~eye)
        if not pos_mask.any():
            continue

        hb = F.normalize(h[b], dim=-1)  # [T, D]
        sim = (hb @ hb.t()) / float(tau)  # [T, T]

        # 去除自对比
        sim = sim.masked_fill(eye, float('-inf'))

        log_den = torch.logsumexp(sim, dim=1)  # [T]
        sim_pos = sim.masked_fill(~pos_mask, float('-inf'))
        log_num = torch.logsumexp(sim_pos, dim=1)  # [T]

        valid = torch.isfinite(log_num)
        if valid.any():
            loss_b = (-(log_num[valid] - log_den[valid])).mean()
            loss_sum = loss_sum + loss_b
            valid_batches += 1

    if valid_batches == 0:
        return h.new_tensor(0.0)
    return loss_sum / valid_batches


class Model(nn.Module):
    """ iTransformer (Encoder-only backbone) + 可选对齐与跨模态。
    - alignment: 'cosine' / 'infonce'（只影响损失，或在 use_align_repre=True 时也影响表示）
    - use_align_repre:
        * False：对齐仅用于 loss；预测仍用原 enc_out/text_in_proj
        * True ：对齐表示通过“残差加法”注入预测路径（无论是否启用 cross-modality）
    - cross-modality:
        * 'ts-to-text' / 'text-to-ts' / 'average' / 'sum' / 'seperate_cross'
    """

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_detrend_reverse = bool(getattr(configs, 'use_detrend_reverse', False))
        self.configs = configs

        # d_model
        self.d_model = int(configs.d_model)

        # Embedding & Encoder
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout
        )
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False, configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model, configs.d_ff,
                    dropout=configs.dropout, activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # Decoder / Heads
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.seq_len, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model, configs.num_class)

        # ===== 对齐/跨模态配置 =====
        self.alignment_mode_default = getattr(configs, 'alignment_mode', 'cosine')
        self.alignment_delta_default = getattr(configs, 'alignment_delta', 0)
        self.alignment_tau_default = getattr(configs, 'alignment_tau', 0.07)
        # 对齐特征维度，同时也作为 TS self-attention 的特征维
        self.align_dim = int(getattr(configs, 'align_dim', configs.d_model))
        self.use_align_repre = bool(getattr(configs, 'use_align_repre', False))

        self.use_cross_default = getattr(configs, 'use_cross_modality', False)
        self.cross_mode_default = getattr(configs, 'cross_modality_mode', 'ts-to-text')
        self.prompt_weight = float(getattr(configs, 'prompt_weight', 0.5))

        # 双对齐开关与超参
        self.use_alignment1_default = bool(getattr(configs, 'use_alignment_1', False))
        self.use_alignment2_default = bool(getattr(configs, 'use_alignment_2', False))
        self.alignment2_tau_default = float(getattr(configs, 'alignment2_tau', 0.07))
        self.alignment1_weight = float(getattr(configs, 'alignment1_weight', 0.2))
        self.alignment2_weight = float(getattr(configs, 'alignment2_weight', 0.2))

        # 文本输入维度
        self.text_dim = int(getattr(configs, 'pred_len', configs.d_model))
        self.text_in_proj = nn.Linear(self.text_dim, self.d_model) if self.text_dim != self.d_model else None

        # N↔L: 初始形状用 configs 判断；真正 in/out features 在运行时通过 _ensure_linear 懒初始化
        self.text_len2var = nn.Linear(configs.seq_len, 1, bias=True)
        self.ts_var2time = nn.Linear(1, configs.seq_len, bias=True)
        self.ts_time2var = nn.Linear(configs.seq_len, 1, bias=True)

        # 对齐投影
        # self.shared_proj2_time = _SharedProj(d_ts=self.align_dim, d_txt=self.align_dim, d_shared=self.align_dim)
        self.shared_proj1 = _SharedProj(d_ts=self.align_dim, d_txt=self.text_dim, d_shared=self.align_dim)
        self.shared_proj2 = _SharedProj(d_ts=self.align_dim, d_txt=self.text_dim, d_shared=self.align_dim)

        self.shared_proj2_time = _SharedProj(d_ts=configs.enc_in, d_txt=self.text_dim, d_shared=self.align_dim)

        # __init__ 末尾附近，和其他层一起注册
        self.ts_da_proj = nn.Linear(self.d_model, self.align_dim)

        self.use_automatic_fuse_weight_for_alignment = getattr(
            configs, 'use_automatic_fuse_weight_for_alignment', True
        )
        if self.use_automatic_fuse_weight_for_alignment:
            init_gate_2 = 1.0
            self.fuse2_ts2ts = nn.Parameter(torch.tensor([init_gate_2]))
            self.fuse2_txt2ts = nn.Parameter(torch.tensor([init_gate_2]))
            self.fuse2_txt2txt = nn.Parameter(torch.tensor([init_gate_2]))
        else:
            self.fuse2_ts2ts = float(getattr(configs, 'alignment2_ts2ts_weight', 1))
            self.fuse2_txt2ts = float(getattr(configs, 'alignment2_txt2ts_weight', 1))
            self.fuse2_txt2txt = float(getattr(configs, 'alignment2_txt2txt_weight', 1))

        # Cross-Modality + 文本分支 MLP
        self.cross_mod = _CrossModality(
            d_model=configs.d_model,
            n_heads=configs.cross_modality_n_heads,
            dropout=configs.dropout
        )
        self.text2pred_head = _Text2PredHead(
            self.d_model, self.pred_len,
            hidden=getattr(configs, 'd_ff', None),
            dropout=getattr(configs, 'dropout', 0.0)
        )

        # ===== Time-series self-attention（沿“时间/变量”维），使用 align_dim =====
        self.use_da_self_attention = bool(getattr(configs, 'da_self_attention', True))
        self.da_heads = int(getattr(configs, 'da_n_heads', 1))  # 默认 1 头

        if self.use_da_self_attention:
            self.da_self_attn = nn.MultiheadAttention(
                embed_dim=self.align_dim,
                num_heads=self.da_heads,
                dropout=configs.dropout,
                batch_first=True
            )
            self.da_self_attn_ln = nn.LayerNorm(self.align_dim)
        else:
            self.da_self_attn = None
            self.da_self_attn_ln = None

        # TS self-attn 后投到 alignment1 维度（这里输入/输出都是 align_dim）
        self.align1_ts_mlp = nn.Sequential(
            nn.LayerNorm(self.align_dim),
            nn.Linear(self.align_dim, self.align_dim),
            nn.GELU()
        )
        self.last_align1_ts: torch.Tensor | None = None

        # fues final repre of ts
        fused_w = 0.2
        self.fuse_align_cross = nn.Parameter(torch.tensor([fused_w]))

        final_fuse_text_w = 0.1
        self.final_fuse_text = nn.Parameter(torch.tensor([final_fuse_text_w]))

        final_fuse_ts_w = 0.9
        self.final_fuse_ts = nn.Parameter(torch.tensor([final_fuse_ts_w]))

    def _apply_alignment_and_cross(
            self, enc_out: torch.Tensor,  # [32,5,512]
            z_text: torch.Tensor = None,
            mask: torch.Tensor = None,
            *,
            # 对齐1
            use_alignment_1: bool = None, alignment_mode=None, alignment_delta=None, alignment_tau=None,
            # 对齐2
            use_alignment_2: bool = None, y_true: torch.Tensor = None,
            # 跨模态
            use_cross=None, cross_mode=None,
            enc_out_original=None
    ):
        """
        输入:
          - enc_out: [B, N, d_model] 作为 TS 变量 token 表示（对齐&跨模态的基础表示）
          - z_text: 文本 token 表示 [B, L, D_txt]
        返回:
          - enc_out_time_first: [B, d_model, N]（主要给“时间版”路径使用；当前 forward 中我们只用 align_loss/extra）
          - align_loss: 若启用对齐则是标量 loss，否则 None
          - extra: {"text_pool": [B, d_model]} 当 cross_mode='seperate_cross' 时提供给解码侧融合
        """
        # -------- 读默认值 --------
        use_alignment_1 = self.use_alignment1_default if use_alignment_1 is None else use_alignment_1
        use_alignment_2 = self.use_alignment2_default if use_alignment_2 is None else use_alignment_2
        alignment_mode = self.alignment_mode_default if alignment_mode is None else alignment_mode
        alignment_delta = self.alignment_delta_default if alignment_delta is None else alignment_delta
        alignment_tau = self.alignment_tau_default if alignment_tau is None else alignment_tau
        use_cross = self.use_cross_default if use_cross is None else use_cross
        cross_mode = self.cross_mode_default if cross_mode is None else cross_mode

        align_loss = None
        extra = {}

        if z_text is None:
            return self.align2model_ts(enc_out), None, extra  # [B, L, d_model]

        device = enc_out.device
        dtype = enc_out.dtype

        _, N, _ = enc_out.shape  # [32,5,512]
        _, L, Dtxt = z_text.shape  # [32,40,12]

        # -------- 小工具：按运行时 N/L 懒初始化/重建 映射层 --------
        def _ensure_linear(name, in_f, out_f, p=None):
            p = self.configs.dropout_DA if p is None else float(p)

            cur = getattr(self, name, None)

            def _mk_new():
                return nn.Sequential(
                    nn.Linear(in_f, out_f, bias=True),
                    nn.Dropout(p)
                ).to(device=device, dtype=dtype)

            # ---- Case 0: not exist ----
            if cur is None:
                mod = _mk_new()
                setattr(self, name, mod)
                return mod

            # ---- Case 1: already Sequential ----
            if isinstance(cur, nn.Sequential):
                # 1.1 Sequential(Linear, Dropout) : normal path
                if (
                        len(cur) >= 2
                        and isinstance(cur[0], nn.Linear)
                        and isinstance(cur[1], nn.Dropout)
                        and cur[0].in_features == in_f
                        and cur[0].out_features == out_f
                        and float(cur[1].p) == p
                ):
                    # make sure device/dtype
                    cur = cur.to(device=device, dtype=dtype)
                    setattr(self, name, cur)
                    return cur

                # 1.2 Sequential(Linear) : patch by appending Dropout, keep weights
                if (
                        len(cur) == 1
                        and isinstance(cur[0], nn.Linear)
                        and cur[0].in_features == in_f
                        and cur[0].out_features == out_f
                ):
                    lin = cur[0].to(device=device, dtype=dtype)
                    mod = nn.Sequential(lin, nn.Dropout(p)).to(device=device, dtype=dtype)
                    setattr(self, name, mod)
                    return mod

                # 1.3 Other Sequential shapes: rebuild
                mod = _mk_new()
                setattr(self, name, mod)
                return mod

            # ---- Case 2: plain Linear ----
            if isinstance(cur, nn.Linear):
                if cur.in_features == in_f and cur.out_features == out_f:
                    lin = cur.to(device=device, dtype=dtype)
                    mod = nn.Sequential(lin, nn.Dropout(p)).to(device=device, dtype=dtype)
                    setattr(self, name, mod)
                    return mod
                # mismatch -> rebuild
                mod = _mk_new()
                setattr(self, name, mod)
                return mod

            # ---- Case 3: unknown type -> rebuild ----
            mod = _mk_new()
            setattr(self, name, mod)
            return mod

        _ensure_linear("text_len2var", L, N)
        _ensure_linear("ts_var2time", N, L)
        _ensure_linear("ts_time2var", L, N)

        enc_out = F.normalize(enc_out, dim=-1)  # [32,5,512]
        enc_var = enc_out  # [32,5,512]

        h_txt_for_cross = z_text  # (32,40,12)

        # ==========================================================
        # 对齐1（变量 token N）
        # ==========================================================
        if use_alignment_1:
            #print("========================Creatalign1")
            proj_ts1 = _ensure_linear("_proj_ts1", N, Dtxt)
            enc_out = proj_ts1(enc_out.transpose(1, 2))  # (32,512,12)

            proj_ts2 = _ensure_linear("_proj_ts2", self.d_model, L)
            enc_out = proj_ts2(enc_out.transpose(1, 2)).transpose(1, 2)  # (32,40,12)

            h_txt_align1 = z_text  # [B, L, align_dim] (32,40,12)
            # 若希望 TS 侧直接用 d_model 表示做对齐，可改为：
            h_ts_align1 = enc_out  # (32,40,12)

            if alignment_mode == 'cosine':
                cur_loss = _cosine_align_loss(h_ts_align1, h_txt_align1, mask=mask)
            elif alignment_mode == 'infonce':
                cur_loss = _info_nce_temporal(
                    h_ts_align1, h_txt_align1,
                    delta=alignment_delta, tau=alignment_tau, mask=mask
                )
            else:
                raise ValueError(f"Unknown alignment_mode: {alignment_mode}")
            cur_loss = self.alignment1_weight * cur_loss
            align_loss = cur_loss if align_loss is None else (align_loss + cur_loss)

        # ==========================================================
        # 对齐2（时间 token L）
        # ==========================================================
        if use_alignment_2:
            txt_time = z_text  # (32,40,12)
            h_ts_t2, h_txt_t2 = self.shared_proj2_time(enc_out_original, txt_time)  # (32,40,64)

            p = float(getattr(self.configs, "dropout_DA"))
            if p > 0:
                h_ts_t2 = F.dropout(h_ts_t2, p=p, training=self.training)
                h_txt_t2 = F.dropout(h_txt_t2, p=p, training=self.training)

            if self.use_align_repre:
                align_2_proj_ts1 = _ensure_linear("align_2_proj_ts1", self.align_dim, N)
                h_ts_t2 = align_2_proj_ts1(h_ts_t2)  # (32,40,5)
                align_2_proj_ts2 = _ensure_linear("align_2_proj_ts2", L, self.d_model)
                ts_inj_var = align_2_proj_ts2(h_ts_t2.transpose(1, 2))  # (32,5,512)

                txt_inj_var = h_txt_t2  # (32,40,64)
                if enc_var.shape[-1] != ts_inj_var.shape[-1]:
                    proj_ts3 = _ensure_linear("proj_ts3", enc_var.shape[-1], ts_inj_var.shape[-1])
                    enc_var = proj_ts3(enc_var)
                enc_var = enc_var + self.fuse2_ts2ts * ts_inj_var

                if h_txt_for_cross is not None:
                    mlp_for_txt = _ensure_linear("_mlp_for_txt", Dtxt, self.align_dim)
                    h_txt_for_cross = mlp_for_txt(h_txt_for_cross) + self.fuse2_txt2txt * txt_inj_var  # (32,40,64)

            # -------- 用 enc_out_original 构造时间 movement 标签（替代 y_true）--------
            with torch.no_grad():
                # enc_out_original 期望形状: [B, L, N]，与当前对齐2的 L 一致
                src = enc_out_original  # [B, L, N]
                if src is None:
                    labels_time = None
                else:
                    eps = 1e-5
                    # 按时间维逐变量标准化
                    s_mean = src.mean(dim=1, keepdim=True)  # [B, 1, N]
                    s_std = src.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)  # [B, 1, N]
                    s_norm = (src - s_mean) / s_std  # [B, L, N]

                    # 变量维等权聚合 -> 得到标量时间序列
                    y_scalar = s_norm.mean(dim=-1)  # [B, L]

                    if y_scalar.shape[1] < 2:
                        labels_time = torch.zeros(
                            y_scalar.shape[0], y_scalar.shape[1],
                            dtype=torch.long, device=device
                        )
                    else:
                        # 一阶差分并符号化 ∈ {-1,0,+1}，首位补 0
                        diff = y_scalar[:, 1:] - y_scalar[:, :-1]  # [B, L-1]
                        signs = torch.sign(diff)  # [B, L-1]
                        labels_time = torch.zeros(
                            y_scalar.shape[0], y_scalar.shape[1],
                            dtype=torch.long, device=device
                        )
                        labels_time[:, 1:] = torch.where(
                            signs > 0, torch.ones_like(signs, dtype=torch.long),
                            torch.where(signs < 0, -torch.ones_like(signs, dtype=torch.long),
                                        torch.zeros_like(signs, dtype=torch.long))
                        )

            if labels_time is not None:
                ts_loss = _supcon_tokens(h_ts_t2, labels_time, tau=self.alignment2_tau_default)
                txt_loss = _supcon_tokens(h_txt_t2, labels_time, tau=self.alignment2_tau_default)
                cur_loss = self.alignment2_weight * (ts_loss + txt_loss)
            else:
                cur_loss = self.alignment2_weight * _info_nce_temporal(
                    h_ts_t2, h_txt_t2,
                    delta=alignment_delta, tau=self.alignment2_tau_default, mask=None
                )
            # print("==========================================")
            # print(self.alignment2_weight, self.alignment1_weight)
            align_loss = cur_loss if align_loss is None else (align_loss + cur_loss)

        # ==========================================================
        # 跨模态
        # ==========================================================
        # 2) h_txt_for_cross 统一到 align_dim
        if h_txt_for_cross.shape[-1] != self.align_dim:
            h_txt_for_cross = _ensure_linear("_pre_cross_txt_to_align", h_txt_for_cross.shape[-1], self.align_dim)(
                h_txt_for_cross)

        if use_cross and h_txt_for_cross is not None:
            txt_for_cross1 = _ensure_linear("_txt_for_cross1", self.align_dim, N)
            txt_for_cross2 = _ensure_linear("_txt_for_cross2", L, self.d_model)
            h_txt_for_cross = txt_for_cross1(h_txt_for_cross)  # (32,40,5)
            h_txt_for_cross = txt_for_cross2(h_txt_for_cross.transpose(1, 2))  # (32,5,512)

            if cross_mode == 'seperate_cross':
                out_ts, out_txt = self.cross_mod(enc_var, h_txt_for_cross, mode='seperate_cross')
                enc_var = out_ts

                txt_for_cross3 = _ensure_linear("_txt_for_cross2_n", N, L)
                out_txt = txt_for_cross3(out_txt.transpose(1, 2)).transpose(1, 2)  # [32,40,512]
                extra['text_pool'] = out_txt.mean(dim=1)  # [B, d_model]
            else:
                enc_var = self.cross_mod(enc_var, h_txt_for_cross, mode=cross_mode)

        # ---- 兜底：确保输出最后一维 = d_model（给主干 projection 用）
        if enc_var.shape[-1] != self.d_model:
            to_dmodel = _ensure_linear("_to_dmodel_tail", enc_var.shape[-1], self.d_model)
            enc_var = to_dmodel(enc_var)

        enc_out_time_first = enc_var  # [32,5,512]
        return enc_out_time_first, align_loss, extra

    # ---------- 其他任务 ----------
    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = x_enc.mean(1, keepdim=True).detach()
        x_ = x_enc - means
        stdev = torch.sqrt(torch.var(x_, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_ /= stdev
        _, L, N = x_.shape
        enc_out = self.enc_embedding(x_, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_ = x_enc - means
        stdev = torch.sqrt(torch.var(x_, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_ /= stdev
        _, L, N = x_.shape
        enc_out = self.enc_embedding(x_, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, L, 1))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        output = F.gelu(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    # ---------- forward ----------
    def forward(
            self, x_enc, x_mark_enc, x_dec, x_mark_dec,
            mask: torch.Tensor = None, z_text: torch.Tensor = None,
            *,
            use_alignment_1: bool = None, alignment_mode: str = None,
            alignment_delta: int = None, alignment_tau: float = None,
            use_alignment_2: bool = None, y_true: torch.Tensor = None,
            use_cross: bool = None, cross_mode: str = None
    ):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            if self.use_detrend_reverse:
                level = x_enc[:, -1, :].detach()
                x_ = x_enc - level.unsqueeze(1)
                means = None
                stdev = None
            else:
                level = None
                means = x_enc.mean(1, keepdim=True).detach()
                x_ = x_enc - means
                stdev = torch.sqrt(torch.var(x_, dim=1, keepdim=True, unbiased=False) + 1e-5)
                x_ /= stdev

            _, W, N = x_.shape

            enc_out = self.enc_embedding(x_, x_mark_enc)  # [32, 5, 512]
            enc_out, _ = self.encoder(enc_out, attn_mask=None)  # [32, 5, 512]
            _, L, _ = enc_out.shape
            enc_tokens = enc_out  # [32,5,512]

            # ===== TS self-attention：d_model -> align_dim -> MHA =====
            ts_time = enc_tokens

            if self.use_da_self_attention and self.da_self_attn is not None:
                pass
                # sa_out, _ = self.da_self_attn(ts_time, ts_time, ts_time)
                # ts_time = self.da_self_attn_ln(ts_time + sa_out)

            # ===== 对齐 + 跨模态（仅影响损失 & 文本分支）=====
            enc_final, align_loss, extra = self._apply_alignment_and_cross(
                ts_time,
                z_text=z_text, mask=mask,
                use_alignment_1=use_alignment_1, alignment_mode=alignment_mode,
                alignment_delta=alignment_delta, alignment_tau=alignment_tau,
                use_alignment_2=use_alignment_2, y_true=y_true,
                use_cross=use_cross, cross_mode=cross_mode,
                enc_out_original=x_
            )  # [32,5,512]

            enc_final = enc_out + self.fuse_align_cross * enc_final

            # ===== 数值预测主干 =====
            dec_seq = self.projection(enc_final).permute(0, 2, 1)[:, :, :N]  # [32,40,1]

            if self.use_detrend_reverse:
                dec_seq = dec_seq + level.unsqueeze(1)
            else:
                dec_seq = dec_seq * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
                dec_seq = dec_seq + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

            dec_seq = dec_seq[:, -self.pred_len:, :]

            if (use_cross if use_cross is not None else self.use_cross_default) and \
                    (cross_mode if cross_mode is not None else self.cross_mode_default) == 'seperate_cross' and \
                    ('text_pool' in extra):
                text_pool = extra['text_pool']  # [B, d_model]
                mlp_pred = self.text2pred_head(text_pool)  # [B, pred_len]
                mlp_pred = mlp_pred.unsqueeze(-1).expand(-1, self.pred_len, N)
                w = self.prompt_weight
                y_hat = (1.0 - w) * dec_seq + w * mlp_pred
                # y_hat = self.final_fuse_ts * dec_seq + self.final_fuse_text * mlp_pred
                return (y_hat, align_loss) if align_loss is not None else y_hat

            return (dec_seq, align_loss) if align_loss is not None else dec_seq

        if self.task_name == 'imputation':
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc, x_mark_enc)
        return None
