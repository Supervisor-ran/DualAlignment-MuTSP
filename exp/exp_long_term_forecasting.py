# ------------------------------------------------------------------
# exp/exp_long_term_forecasting.py
# Cleaned official version: DATransformer + BERT only.
# ------------------------------------------------------------------

from data_provider.data_factory import data_provider, get_enc_in
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, visual
from utils.metrics import metric_masked

import os
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from transformers import BertConfig, BertModel, BertTokenizer

from exp.exp_tools.tensor_ops import (
    norm as _norm_tool,
    info_nce_loss as _info_nce_loss_tool,
    time_align_text_emb as _time_align_text_emb_tool,
    feat_slice as _feat_slice_tool,
)
from exp.exp_tools.dataset_ops import (
    get_prior_y_safe as _get_prior_y_safe_tool,
    get_text_inputs as _get_text_inputs_tool,
)
from exp.exp_tools.text_encoder import (
    safe_from_pretrained as _safe_from_pretrained_tool,
    ensure_text_encoder_ready as _ensure_text_encoder_ready_tool,
    encode_text_list as _encode_text_list_tool,
)
from exp.exp_tools.text_cache import (
    cache_meta as _cache_meta_tool,
    cache_file_path as _cache_file_path_tool,
    load_text_cache as _load_text_cache_tool,
    save_text_cache as _save_text_cache_tool,
    hash_text as _hash_text_tool,
    vec_from_cache as _vec_from_cache_tool,
)
from exp.exp_tools.exp_extend import MLP, MultiVarIndepWrapper

warnings.filterwarnings("ignore")


class Exp_Long_Term_Forecast(Exp_Basic):
    """Long-term forecasting experiment for DATransformer only."""

    def __init__(self, args):
        if getattr(args, "model", None) != "DATransformer":
            raise ValueError("This cleaned experiment only supports DATransformer.")
        if getattr(args, "llm_model", None) != "BERT":
            raise ValueError("This cleaned experiment only supports --llm_model BERT.")

        if not hasattr(args, "enc_in") or args.enc_in is None:
            x0 = get_enc_in(args, flag="train")
            args.enc_in = x0

        super(Exp_Long_Term_Forecast, self).__init__(args)
        configs = args

        # ====== configuration ======
        self.text_path = configs.text_path
        self.prompt_weight = configs.prompt_weight
        self.attribute = "final_sum"
        self.type_tag = configs.type_tag
        self.text_len = configs.text_len
        self.d_llm = configs.llm_dim
        self.pred_len = configs.pred_len
        self.text_embedding_dim = configs.text_emb
        self.pool_type = configs.pool_type
        self.use_fullmodel = configs.use_fullmodel
        self.hug_token = configs.huggingface_token

        # ====== cache dir ======
        try:
            base_dir = Path(__file__).resolve().parents[2]
        except NameError:
            cwd = Path.cwd().resolve()
            base_dir = cwd.parents[2] if len(cwd.parents) >= 3 else cwd

        self.cache_dir = str(base_dir / "pretrained_models")
        os.makedirs(self.cache_dir, exist_ok=True)

        # ====== text-side MLP ======
        mlp_sizes = [self.d_llm, int(self.d_llm / 8), self.text_embedding_dim]
        self.llm_model = None
        self.tokenizer = None
        self.Doc2Vec = False
        self.mlp = MLP(mlp_sizes, dropout_rate=0.3) if mlp_sizes is not None else None

        self.args.freeze_text_mlp = getattr(self.args, "freeze_text_mlp", False)
        self.args.freeze_text_mlp_last = getattr(self.args, "freeze_text_mlp_last", False)

        if self.mlp is not None:
            if self.args.freeze_text_mlp:
                for p in self.mlp.parameters():
                    p.requires_grad = False
                print(f"[{self.args.model}] Froze text MLP (self.mlp).")
            elif self.args.freeze_text_mlp_last:
                for p in self.mlp.parameters():
                    p.requires_grad = False
                if hasattr(self.mlp, "layers") and len(self.mlp.layers) > 0:
                    for p in self.mlp.layers[-1].parameters():
                        p.requires_grad = True
                print(f"[{self.args.model}] Unfroze ONLY the last layer of text MLP.")

        # ====== output projection MLP ======
        mlp_sizes2 = [self.text_embedding_dim + self.args.pred_len, self.args.pred_len]
        self.mlp_proj = MLP(mlp_sizes2, dropout_rate=0.3) if mlp_sizes2 is not None else None

        # ====== BERT text encoder only ======
        self.bert_config = BertConfig.from_pretrained("google-bert/bert-base-uncased")
        self.bert_config.num_hidden_layers = configs.llm_layers
        self.bert_config.output_attentions = True
        self.bert_config.output_hidden_states = True

        try:
            self.llm_model = BertModel.from_pretrained(
                "google-bert/bert-base-uncased",
                trust_remote_code=True,
                local_files_only=True,
                config=self.bert_config,
            )
        except EnvironmentError:
            print("Local model files not found. Attempting to download...")
            self.llm_model = BertModel.from_pretrained(
                "google-bert/bert-base-uncased",
                trust_remote_code=True,
                local_files_only=False,
                config=self.bert_config,
            )

        try:
            self.tokenizer = BertTokenizer.from_pretrained(
                "google-bert/bert-base-uncased",
                trust_remote_code=True,
                local_files_only=True,
            )
        except EnvironmentError:
            print("Local tokenizer files not found. Attempting to download them..")
            self.tokenizer = BertTokenizer.from_pretrained(
                "google-bert/bert-base-uncased",
                trust_remote_code=True,
                local_files_only=False,
            )

        _tok = getattr(self, "tokenizer", None)
        if _tok is not None and getattr(_tok, "pad_token", None) is None:
            eos = getattr(_tok, "eos_token", None)
            if eos:
                _tok.pad_token = eos
            else:
                pad_token = "[PAD]"
                _tok.add_special_tokens({"pad_token": pad_token})
                _tok.pad_token = pad_token

        if self.llm_model is not None:
            for param in self.llm_model.parameters():
                param.requires_grad = False
            self.llm_model = self.llm_model.to(self.device)

        # ====== learnable weights ======
        if args.init_method == "uniform":
            self.weight1 = nn.Embedding(1, self.args.pred_len)
            self.weight2 = nn.Embedding(1, self.args.pred_len)
            nn.init.uniform_(self.weight1.weight)
            nn.init.uniform_(self.weight2.weight)
            self.weight1.weight.requires_grad = True
            self.weight2.weight.requires_grad = True
        elif args.init_method == "normal":
            self.weight1 = nn.Embedding(1, self.args.pred_len)
            self.weight2 = nn.Embedding(1, self.args.pred_len)
            nn.init.normal_(self.weight1.weight)
            nn.init.normal_(self.weight2.weight)
            self.weight1.weight.requires_grad = True
            self.weight2.weight.requires_grad = True
        else:
            raise ValueError("Unsupported initialization method")

        if self.mlp is not None:
            self.mlp = self.mlp.to(self.device)
        if self.mlp_proj is not None:
            self.mlp_proj = self.mlp_proj.to(self.device)

        self.learning_rate2 = 1e-2
        self.learning_rate3 = 1e-3
        self._best_val = float("inf")

        setattr(self.args, "text_dim", int(self.text_embedding_dim))
        self.args.use_alignment_1 = getattr(self.args, "use_alignment_1", False)
        self.args.use_alignment_2 = getattr(self.args, "use_alignment_2", False)
        self.args.alignment_mode = getattr(self.args, "alignment_mode", "cosine")
        self.args.alignment_weight = getattr(self.args, "alignment_weight", 1.0)
        self.args.alignment_delta = getattr(self.args, "alignment_delta", 0)
        self.args.alignment_tau = getattr(self.args, "alignment_tau", 0.07)
        self.args.alignment2_weight = getattr(self.args, "alignment2_weight", self.args.alignment_weight)
        self.args.alignment2_tau = getattr(self.args, "alignment2_tau", self.args.alignment_tau)
        self.args.use_cross_modality = getattr(self.args, "use_cross_modality", True)
        self.args.cross_modality_mode = getattr(self.args, "cross_modality_mode", "seperate_cross")

        self.text_cache_train_write = bool(getattr(self.args, "text_cache_train_write", False))
        self.text_cache_max_items = int(getattr(self.args, "text_cache_max_items", 200000))
        self.gc_interval = int(getattr(self.args, "gc_interval", 200))

    # ============================================================
    # exp_tools wrappers
    # ============================================================
    def _norm(self, x):
        return _norm_tool(x)

    def _info_nce_loss(self, z_time, z_text, tau=0.07):
        return _info_nce_loss_tool(z_time, z_text, tau=tau)

    def _time_align_text_emb(self, txt_emb, target_len: int):
        return _time_align_text_emb_tool(txt_emb, target_len)

    def _feat_slice(self):
        return _feat_slice_tool(self.args)

    def _get_prior_y_safe(self, dataset, index):
        return _get_prior_y_safe_tool(self, dataset, index)

    def _get_text_inputs(self, dataset, index):
        return _get_text_inputs_tool(dataset, index)

    def _safe_from_pretrained(self, model_cls, repo, *, config=None, tokenizer=False, cache_dir=None, **kw):
        return _safe_from_pretrained_tool(model_cls, repo, config=config, tokenizer=tokenizer, cache_dir=cache_dir, **kw)

    def _ensure_text_encoder_ready(self):
        return _ensure_text_encoder_ready_tool(self)

    def _encode_text_list(self, text_list, pool_token: str = "avg", reduce_time: str = None):
        return _encode_text_list_tool(self, text_list, pool_token=pool_token, reduce_time=reduce_time)

    def _cache_meta(self):
        return _cache_meta_tool(self)

    def _cache_file_path(self):
        return _cache_file_path_tool(self)

    def _load_text_cache(self):
        return _load_text_cache_tool(self)

    def _save_text_cache(self, force=False):
        return _save_text_cache_tool(self, force=force)

    @staticmethod
    def _hash_text(t: str) -> str:
        return _hash_text_tool(t)

    def _vec_from_cache(self, texts, make_missing):
        return _vec_from_cache_tool(self, texts, make_missing)

    # ============================================================
    # train / vali / test utilities
    # ============================================================
    def _build_model(self):
        if getattr(self.args, "model", "") != "DATransformer":
            raise ValueError("This cleaned experiment only supports DATransformer.")

        base_cls = self.model_dict[self.args.model].Model

        use_indep = bool(getattr(self.args, "multivar_indep", False)) \
            and (getattr(self.args, "features", None) == "M") \
            and (getattr(self.args, "model", "") == "DATransformer")

        if use_indep:
            n_vars = int(getattr(self.args, "enc_in", 0))
            assert n_vars >= 1, f"args.enc_in must be >=1, got {n_vars}"
            model = MultiVarIndepWrapper(base_cls, self.args, n_vars=n_vars).to(self.device)
            print(f"[DATransformer] MultiVarIndepWrapper enabled: enc_in={n_vars} -> {n_vars} independent models.")
        else:
            model = base_cls(self.args).float()

        def _apply_freeze_to_one(m):
            if getattr(self.args, "freeze_p_txt", True):
                try:
                    sp1 = getattr(m, "shared_proj1", None)
                    if sp1 is not None and hasattr(sp1, "p_txt"):
                        for p in sp1.p_txt.parameters():
                            p.requires_grad = False
                        print(f"[{self.args.model}] Froze shared_proj1.p_txt (freeze_p_txt=True).")
                except Exception as e:
                    print(f"[{self.args.model}] WARN: freezing shared_proj1.p_txt failed: {e}")

            freeze_list = getattr(self.args, "freeze_mlp_list", None)
            if freeze_list:
                target = m.module if hasattr(m, "module") else m
                if hasattr(target, "freeze_from_list"):
                    hit = target.freeze_from_list(freeze_list)
                    print(f"[DATransformer] freeze_mlp_list cfg = {list(freeze_list)}")
                    print(f"[{self.args.model}] Freeze from list: {sorted(hit)}")
                else:
                    print(f"[{self.args.model}] WARN: model has no freeze_from_list, skip freeze_mlp_list.")

        if isinstance(model, MultiVarIndepWrapper):
            for sm in model.models:
                _apply_freeze_to_one(sm)
        else:
            _apply_freeze_to_one(model)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_optimizer_indep(self):
        m = self.model.module if hasattr(self.model, "module") else self.model
        assert hasattr(m, "models"), "MultiVarIndepWrapper expected: .models"
        opts = []
        for sm in m.models:
            opts.append(optim.Adam(sm.parameters(), lr=self.args.learning_rate))
        return opts

    def _select_optimizer_mlp(self):
        lr = getattr(self.args, "learning_rate2", getattr(self, "learning_rate2", 1e-2))
        params = [p for p in self.mlp.parameters() if p.requires_grad] if self.mlp is not None else []
        return optim.Adam(params, lr=lr) if len(params) > 0 else None

    def _select_optimizer_proj(self):
        lr = getattr(self.args, "learning_rate3", getattr(self, "learning_rate3", 1e-3))
        return optim.Adam(self.mlp_proj.parameters(), lr=lr) if self.mlp_proj is not None else None

    def _select_optimizer_weight(self):
        return optim.Adam(
            [{"params": self.weight1.parameters()}, {"params": self.weight2.parameters()}],
            lr=self.args.learning_rate_weight,
        )

    def _select_criterion(self):
        if self.args.loss == "MSE":
            return nn.MSELoss()
        raise ValueError("Unsupported loss function")

    def _encode_batch_text(self, dataset, index, *, nan_to_num=False):
        hist_text, fut_text = self._get_text_inputs(dataset, index)
        hist_tok_emb = self._encode_text_list(hist_text)
        fut_tok_emb = self._encode_text_list(fut_text) if fut_text is not None else None

        if hist_tok_emb is not None and self.mlp is not None:
            hist_tok_emb = self.mlp(hist_tok_emb)
            if nan_to_num:
                hist_tok_emb = torch.nan_to_num(hist_tok_emb, nan=0.0, posinf=0.0, neginf=0.0)
        if fut_tok_emb is not None and self.mlp is not None:
            fut_tok_emb = self.mlp(fut_tok_emb)
            if nan_to_num:
                fut_tok_emb = torch.nan_to_num(fut_tok_emb, nan=0.0, posinf=0.0, neginf=0.0)

        return hist_text, fut_text, hist_tok_emb, fut_tok_emb

    # ============================================================
    # validation
    # ============================================================
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        if self.mlp is not None:
            self.mlp.eval()
        if self.mlp_proj is not None:
            self.mlp_proj.eval()

        use_dt_indep = (getattr(self.args, "model", "") == "DATransformer") \
            and bool(getattr(self.args, "multivar_indep", False)) \
            and (getattr(self.args, "features", None) == "M")
        use_dt_loss_indep = use_dt_indep and bool(getattr(self.args, "multivar_loss_indep", False))

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                batch_x = torch.nan_to_num(batch_x, nan=0.0, posinf=0.0, neginf=0.0)
                batch_y = torch.nan_to_num(batch_y, nan=0.0, posinf=0.0, neginf=0.0)
                batch_x_mark = torch.nan_to_num(batch_x_mark, nan=0.0, posinf=0.0, neginf=0.0)
                batch_y_mark = torch.nan_to_num(batch_y_mark, nan=0.0, posinf=0.0, neginf=0.0)

                prior_y = self._get_prior_y_safe(vali_data, index)
                if prior_y is not None:
                    prior_y = torch.nan_to_num(prior_y, nan=0.0, posinf=0.0, neginf=0.0)

                hist_text, fut_text, hist_tok_emb, fut_tok_emb = self._encode_batch_text(
                    vali_data, index, nan_to_num=True
                )

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

                T_enc = batch_x.shape[1]
                hist_tok_emb = self._time_align_text_emb(hist_tok_emb, T_enc)
                align_w = float(getattr(self.args, "alignment_weight", 1.0))

                if use_dt_loss_indep:
                    mwrap = self.model.module if hasattr(self.model, "module") else self.model
                    n_vars = int(getattr(self.args, "enc_in", batch_x.shape[-1]))
                    assert hasattr(mwrap, "models"), "multivar_loss_indep=True requires MultiVarIndepWrapper"
                    assert n_vars == batch_x.shape[-1], f"enc_in({n_vars}) != batch_x.shape[-1]({batch_x.shape[-1]})"
                    assert len(mwrap.models) == n_vars, f"wrapper models({len(mwrap.models)}) != n_vars({n_vars})"

                    loss_list = []
                    for v in range(n_vars):
                        bx = batch_x[..., v:v + 1]
                        by = batch_y[..., v:v + 1]
                        di = dec_inp[..., v:v + 1]

                        out = mwrap.models[v](
                            bx, batch_x_mark, di, batch_y_mark,
                            z_text=hist_tok_emb,
                            use_alignment_1=getattr(self.args, "use_alignment_1", True),
                            alignment_mode=getattr(self.args, "alignment_mode", "cosine"),
                            alignment_delta=getattr(self.args, "alignment_delta", 0),
                            alignment_tau=getattr(self.args, "alignment_tau", 0.07),
                            use_alignment_2=getattr(self.args, "use_alignment_2", False),
                            y_true=by,
                            use_cross=getattr(self.args, "use_cross_modality", True),
                            cross_mode=getattr(self.args, "cross_modality_mode", "seperate_cross"),
                        )

                        if isinstance(out, tuple):
                            yhat, align_loss = out
                        else:
                            yhat, align_loss = out, None

                        yhat = yhat[:, -self.args.pred_len:, :]
                        by2 = by[:, -self.args.pred_len:, :]

                        loss_v = criterion(yhat, by2)
                        if align_loss is not None:
                            loss_v = loss_v + align_w * align_loss

                        loss_list.append(loss_v)

                    loss = torch.stack(loss_list).mean()

                else:
                    out = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark,
                        z_text=hist_tok_emb,
                        use_alignment_1=getattr(self.args, "use_alignment_1", True),
                        alignment_mode=getattr(self.args, "alignment_mode", "cosine"),
                        alignment_delta=getattr(self.args, "alignment_delta", 0),
                        alignment_tau=getattr(self.args, "alignment_tau", 0.07),
                        use_alignment_2=getattr(self.args, "use_alignment_2", False),
                        y_true=batch_y,
                        use_cross=getattr(self.args, "use_cross_modality", True),
                        cross_mode=getattr(self.args, "cross_modality_mode", "seperate_cross"),
                    )

                    if isinstance(out, tuple):
                        outputs, align_loss = out
                    else:
                        outputs, align_loss = out, None

                    f_slice = self._feat_slice()
                    outputs = outputs[:, -self.args.pred_len:, f_slice]
                    batch_y2 = batch_y[:, -self.args.pred_len:, f_slice].to(self.device)

                    loss = criterion(outputs, batch_y2)
                    if align_loss is not None:
                        loss = loss + align_w * align_loss

                total_loss.append(loss)

        vals = []
        for l in total_loss:
            v = float(l.detach().cpu().item())
            if np.isfinite(v):
                vals.append(v)
            else:
                print(f"[vali] drop NaN/Inf loss item: {v}")

        total_loss = np.mean(vals) if len(vals) > 0 else float("inf")

        self.model.train()
        if self.mlp is not None:
            self.mlp.train()
        if self.mlp_proj is not None:
            self.mlp_proj.train()

        return total_loss

    # ============================================================
    # training
    # ============================================================
    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        use_dt_indep = (getattr(self.args, "model", "") == "DATransformer") \
            and bool(getattr(self.args, "multivar_indep", False)) \
            and (getattr(self.args, "features", None) == "M")
        use_dt_loss_indep = use_dt_indep and bool(getattr(self.args, "multivar_loss_indep", False))

        if use_dt_loss_indep:
            model_optim = None
            model_optim_list = self._select_optimizer_indep()
        else:
            model_optim = self._select_optimizer()
            model_optim_list = None

        model_optim_mlp = self._select_optimizer_mlp()
        model_optim_proj = self._select_optimizer_proj()
        criterion = self._select_criterion()

        if model_optim_mlp is None:
            print("[OPT] self.mlp has no trainable parameters; skip its optimizer/steps.")

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        print(f"[DATransformer] multivar_indep={bool(getattr(self.args, 'multivar_indep', False))}, "
              f"multivar_loss_indep={bool(getattr(self.args, 'multivar_loss_indep', False))}, "
              f"features={getattr(self.args, 'features', None)}, enc_in={getattr(self.args, 'enc_in', None)}")

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            if self.mlp is not None:
                self.mlp.train()
            if self.mlp_proj is not None:
                self.mlp_proj.train()

            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(train_loader):
                iter_count += 1

                if model_optim is not None:
                    model_optim.zero_grad()

                if model_optim_mlp is not None:
                    model_optim_mlp.zero_grad()
                if model_optim_proj is not None:
                    model_optim_proj.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                prior_y = self._get_prior_y_safe(train_data, index)
                hist_text, fut_text, hist_tok_emb, fut_tok_emb = self._encode_batch_text(train_data, index)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

                T_enc = batch_x.shape[1]
                hist_tok_emb = self._time_align_text_emb(hist_tok_emb, T_enc)
                align_w = float(getattr(self.args, "alignment_weight", 1.0))

                if use_dt_loss_indep:
                    mwrap = self.model.module if hasattr(self.model, "module") else self.model
                    n_vars = int(getattr(self.args, "enc_in", batch_x.shape[-1]))
                    assert hasattr(mwrap, "models"), "multivar_loss_indep=True requires MultiVarIndepWrapper"
                    assert n_vars == batch_x.shape[-1], f"enc_in({n_vars}) != batch_x.shape[-1]({batch_x.shape[-1]})"
                    assert len(mwrap.models) == n_vars, f"wrapper models({len(mwrap.models)}) != n_vars({n_vars})"

                    loss_list = []

                    for v in range(n_vars):
                        opt_v = model_optim_list[v]
                        opt_v.zero_grad()

                        bx = batch_x[..., v:v + 1]
                        by = batch_y[..., v:v + 1]
                        di = dec_inp[..., v:v + 1]

                        retain = (v < n_vars - 1)

                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():
                                out = mwrap.models[v](
                                    bx, batch_x_mark, di, batch_y_mark,
                                    z_text=hist_tok_emb,
                                    use_alignment_1=getattr(self.args, "use_alignment_1", True),
                                    alignment_mode=getattr(self.args, "alignment_mode", "cosine"),
                                    alignment_delta=getattr(self.args, "alignment_delta", 0),
                                    alignment_tau=getattr(self.args, "alignment_tau", 0.07),
                                    use_alignment_2=getattr(self.args, "use_alignment_2", False),
                                    y_true=by,
                                    use_cross=getattr(self.args, "use_cross_modality", True),
                                    cross_mode=getattr(self.args, "cross_modality_mode", "seperate_cross"),
                                )
                            if isinstance(out, tuple):
                                yhat, align_loss = out
                            else:
                                yhat, align_loss = out, None

                            yhat = yhat[:, -self.args.pred_len:, :]
                            by2 = by[:, -self.args.pred_len:, :]

                            loss_v = criterion(yhat, by2)
                            if align_loss is not None:
                                loss_v = loss_v + align_w * align_loss

                            scaler.scale(loss_v).backward(retain_graph=retain)
                            scaler.step(opt_v)
                            loss_list.append(loss_v.detach())
                        else:
                            out = mwrap.models[v](
                                bx, batch_x_mark, di, batch_y_mark,
                                z_text=hist_tok_emb,
                                use_alignment_1=getattr(self.args, "use_alignment_1", True),
                                alignment_mode=getattr(self.args, "alignment_mode", "cosine"),
                                alignment_delta=getattr(self.args, "alignment_delta", 0),
                                alignment_tau=getattr(self.args, "alignment_tau", 0.07),
                                use_alignment_2=getattr(self.args, "use_alignment_2", False),
                                y_true=by,
                                use_cross=getattr(self.args, "use_cross_modality", True),
                                cross_mode=getattr(self.args, "cross_modality_mode", "seperate_cross"),
                            )

                            if isinstance(out, tuple):
                                yhat, align_loss = out
                            else:
                                yhat, align_loss = out, None

                            yhat = yhat[:, -self.args.pred_len:, :]
                            by2 = by[:, -self.args.pred_len:, :]

                            loss_v = criterion(yhat, by2)
                            if align_loss is not None:
                                loss_v = loss_v + align_w * align_loss

                            loss_v.backward(retain_graph=retain)
                            opt_v.step()
                            loss_list.append(loss_v.detach())

                    if self.args.use_amp:
                        scaler.update()

                    if model_optim_mlp is not None:
                        model_optim_mlp.step()
                    if model_optim_proj is not None:
                        model_optim_proj.step()

                    loss = torch.stack(loss_list).mean()
                    outputs = None

                else:
                    if self.args.use_amp:
                        with torch.cuda.amp.autocast():
                            out = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                z_text=hist_tok_emb,
                                use_alignment_1=getattr(self.args, "use_alignment_1", True),
                                alignment_mode=getattr(self.args, "alignment_mode", "cosine"),
                                alignment_delta=getattr(self.args, "alignment_delta", 0),
                                alignment_tau=getattr(self.args, "alignment_tau", 0.07),
                                use_alignment_2=getattr(self.args, "use_alignment_2", False),
                                y_true=batch_y,
                                use_cross=getattr(self.args, "use_cross_modality", True),
                                cross_mode=getattr(self.args, "cross_modality_mode", "seperate_cross"),
                            )
                    else:
                        out = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark,
                            z_text=hist_tok_emb,
                            use_alignment_1=getattr(self.args, "use_alignment_1", True),
                            alignment_mode=getattr(self.args, "alignment_mode", "cosine"),
                            alignment_delta=getattr(self.args, "alignment_delta", 0),
                            alignment_tau=getattr(self.args, "alignment_tau", 0.07),
                            use_alignment_2=getattr(self.args, "use_alignment_2", False),
                            y_true=batch_y,
                            use_cross=getattr(self.args, "use_cross_modality", True),
                            cross_mode=getattr(self.args, "cross_modality_mode", "seperate_cross"),
                        )

                    if isinstance(out, tuple):
                        outputs, align_loss = out
                    else:
                        outputs, align_loss = out, None

                    f_slice = self._feat_slice()
                    outputs = outputs[:, -self.args.pred_len:, f_slice]
                    batch_y2 = batch_y[:, -self.args.pred_len:, f_slice].to(self.device)
                    loss = criterion(outputs, batch_y2)
                    if align_loss is not None:
                        loss = loss + align_w * align_loss

                if not use_dt_loss_indep:
                    if self.args.use_amp:
                        scaler.scale(loss).backward()
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        loss.backward()
                        model_optim.step()

                    if model_optim_mlp is not None:
                        model_optim_mlp.step()
                    if model_optim_proj is not None:
                        model_optim_proj.step()

                train_loss.append(float(loss.detach().cpu().item()) if torch.is_tensor(loss) else float(loss))

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, float(loss)))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                    try:
                        del outputs
                        del batch_x, batch_y, batch_x_mark, batch_y_mark
                        del dec_inp
                        del hist_tok_emb, fut_tok_emb
                        del hist_text, fut_text
                        del prior_y
                    except Exception:
                        pass

                if (i + 1) % max(1, self.gc_interval) == 0:
                    import gc
                    gc.collect()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss
            ))

            if vali_loss < self._best_val:
                self._best_val = vali_loss
                torch.save({
                    "model": self.model.state_dict(),
                    "mlp": self.mlp.state_dict() if self.mlp is not None else None,
                    "mlp_proj": self.mlp_proj.state_dict() if self.mlp_proj is not None else None,
                    "weight1": self.weight1.state_dict(),
                    "weight2": self.weight2.state_dict(),
                    "args": vars(self.args) if hasattr(self.args, "__dict__") else self.args,
                }, os.path.join(path, "checkpoint_all.pth"))

            early_stopping(vali_loss, self.model, path)
            import gc
            gc.collect()
            self._save_text_cache(force=True)

            m = self.model.module if hasattr(self.model, "module") else self.model
            if hasattr(m, "flush_prompt_cache"):
                m.flush_prompt_cache(force=True)

            if early_stopping.early_stop:
                print("Early stopping")
                break

        ckpt_all_path = os.path.join(path, "checkpoint_all.pth")
        if not os.path.exists(ckpt_all_path):
            try:
                torch.save({
                    "model": self.model.state_dict(),
                    "mlp": self.mlp.state_dict() if self.mlp is not None else None,
                    "mlp_proj": self.mlp_proj.state_dict() if self.mlp_proj is not None else None,
                    "weight1": self.weight1.state_dict(),
                    "weight2": self.weight2.state_dict(),
                    "args": vars(self.args) if hasattr(self.args, "__dict__") else self.args,
                }, ckpt_all_path)
                print(f"[C1] Wrote fallback checkpoint_all.pth with args -> {ckpt_all_path}")
            except Exception as e:
                print(f"[C1] WARN: failed to write fallback checkpoint_all.pth: {repr(e)}")

        ckpt_all = os.path.join(path, "checkpoint_all.pth")
        if os.path.exists(ckpt_all):
            state = torch.load(ckpt_all, map_location=self.device)
            self.model.load_state_dict(state["model"])
            if self.mlp is not None and state.get("mlp") is not None:
                self.mlp.load_state_dict(state["mlp"])
            if self.mlp_proj is not None and state.get("mlp_proj") is not None:
                self.mlp_proj.load_state_dict(state["mlp_proj"])
            self.weight1.load_state_dict(state["weight1"])
            self.weight2.load_state_dict(state["weight2"])
        else:
            best_model_path = os.path.join(path, "checkpoint.pth")
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        m = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(m, "flush_prompt_cache"):
            m.flush_prompt_cache(force=True)
        return self.model

    # ============================================================
    # testing
    # ============================================================
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        if test:
            print("loading model")
            p_all = os.path.join("./checkpoints/" + setting, "checkpoint_all.pth")
            if os.path.exists(p_all):
                state = torch.load(p_all, map_location=self.device)
                self.model.load_state_dict(state["model"])
                if self.mlp is not None and state.get("mlp") is not None:
                    self.mlp.load_state_dict(state["mlp"])
                if self.mlp_proj is not None and state.get("mlp_proj") is not None:
                    self.mlp_proj.load_state_dict(state["mlp_proj"])
                self.weight1.load_state_dict(state["weight1"])
                self.weight2.load_state_dict(state["weight2"])
            else:
                self.model.load_state_dict(torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"),
                                                      map_location=self.device))

        preds = []
        trues = []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        if self.mlp is not None:
            self.mlp.eval()
        if self.mlp_proj is not None:
            self.mlp_proj.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, index) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                prior_y = self._get_prior_y_safe(test_data, index)
                hist_text, fut_text, hist_tok_emb, fut_tok_emb = self._encode_batch_text(test_data, index)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).to(self.device)

                T_enc = batch_x.shape[1]
                hist_tok_emb = self._time_align_text_emb(hist_tok_emb, T_enc)

                out = self.model(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark,
                    z_text=hist_tok_emb,
                    use_alignment_1=getattr(self.args, "use_alignment_1", True),
                    alignment_mode=getattr(self.args, "alignment_mode", "cosine"),
                    alignment_delta=getattr(self.args, "alignment_delta", 0),
                    alignment_tau=getattr(self.args, "alignment_tau", 0.07),
                    use_alignment_2=getattr(self.args, "use_alignment_2", False),
                    y_true=batch_y,
                    use_cross=getattr(self.args, "use_cross_modality", True),
                    cross_mode=getattr(self.args, "cross_modality_mode", "seperate_cross"),
                )

                outputs = out[0] if isinstance(out, tuple) else out
                f_slice = self._feat_slice()
                outputs = outputs[:, -self.args.pred_len:, f_slice]
                batch_y = batch_y[:, -self.args.pred_len:, f_slice].to(self.device)

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                f_slice = self._feat_slice()
                outputs = outputs[:, -self.args.pred_len:, f_slice]
                batch_y = batch_y[:, -self.args.pred_len:, f_slice]

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input_arr = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input_arr.shape
                        input_arr = test_data.inverse_transform(input_arr.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input_arr[0, :, -1], true[0, :, -1]), axis=0)
                    pd_ = np.concatenate((input_arr[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd_, os.path.join(folder_path, str(i) + ".pdf"))

        m = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(m, "flush_prompt_cache"):
            m.flush_prompt_cache(force=True)

        preds = np.array(preds)
        trues = np.array(trues)
        trues = trues[:, :, -self.args.pred_len:, :]
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        dtw = -999
        mae, mse, rmse, mape, mspe = metric_masked(preds, trues)
        print("mse:{}, mae:{}, dtw:{}".format(mse, mae, dtw))
        with open(self.args.save_name, "a") as f:
            f.write(setting + " \n")
            f.write("mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}".format(mse, mae, rmse, mape, mspe))
            f.write("\n\n")

        np.save(folder_path + "metrics.npy", np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + "pred.npy", preds)
        np.save(folder_path + "true.npy", trues)

        self._save_text_cache(force=True)

        if hasattr(m, "flush_prompt_cache"):
            m.flush_prompt_cache(force=True)

        return mse
