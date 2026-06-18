# exp/exp_tools/text_encoder.py

from __future__ import annotations

import os
import re
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

from .text_cache import vec_from_cache

# BERTopic + FinBERT 相关：只有你选该模式才会用到
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSequenceClassification
    import nltk
    from nltk.tokenize import sent_tokenize
except Exception:
    BERTopic = None
    SentenceTransformer = None
    AutoModelForSequenceClassification = None
    nltk = None
    sent_tokenize = None


def safe_from_pretrained(model_cls, repo, *, config=None, tokenizer=False, cache_dir=None, **kw):
    """
    优先从本地 cache_dir 读取（local_files_only=True），
    如果本地没有则退回到允许下载，并把权重缓存到 cache_dir。
    """
    try:
        if tokenizer:
            return AutoTokenizer.from_pretrained(repo, local_files_only=True, cache_dir=cache_dir, **kw)
        else:
            return model_cls.from_pretrained(repo, local_files_only=True, cache_dir=cache_dir, config=config, **kw)
    except Exception:
        if tokenizer:
            return AutoTokenizer.from_pretrained(repo, local_files_only=False, cache_dir=cache_dir, **kw)
        else:
            return model_cls.from_pretrained(repo, local_files_only=False, cache_dir=cache_dir, config=config, **kw)


def ensure_text_encoder_ready(exp) -> None:
    """
    懒加载文本编码器（SBERT权重 + mean pooling）
    - Doc2Vec 或 BERTopic+FinBERT 路径不需要 HF tokenizer
    - 默认模型 sentence-transformers/all-MiniLM-L6-v2
    - 如隐藏维 != exp.d_llm，则懒建 exp._proj_text2dllm 做线性投影
    """
    if getattr(exp, "Doc2Vec", False):
        return
    if getattr(exp, "use_bertopic_finbert", False):
        return

    tok = getattr(exp, "tokenizer", None)
    mdl = getattr(exp, "llm_model", None)
    if tok is not None and mdl is not None:
        return

    model_name = getattr(exp.args, "text_encoder_name", None) or getattr(exp, "text_encoder_name", None) \
                 or "sentence-transformers/all-MiniLM-L6-v2"

    device = exp.device

    try:
        exp.tokenizer = AutoTokenizer.from_pretrained(model_name)
        exp.llm_model = AutoModel.from_pretrained(model_name).to(device)
        exp.llm_model.eval()
        exp.text_encoder_name = model_name
        exp.text_device = device
    except Exception as e:
        raise RuntimeError(f"[text-encoder lazy init] failed to load '{model_name}': {e}")

    try:
        hidden = int(getattr(exp.llm_model.config, "hidden_size", 0)) or int(exp.llm_model.get_input_embeddings().embedding_dim)
    except Exception:
        hidden = int(exp.llm_model.get_input_embeddings().embedding_dim)

    exp._text_hidden_dim = hidden
    need = int(getattr(exp, "d_llm", hidden))
    if hidden != need:
        if not hasattr(exp, "_proj_text2dllm"):
            exp._proj_text2dllm = nn.Linear(hidden, need, bias=True).to(device)


def encode_text_list(exp, text_list, pool_token: str = "avg", reduce_time: str = None):
    """
    text_list: List[str] 或 List[List[str]]
    返回: [B, L, d] 或 [B, d]
    会通过 vec_from_cache 做缓存（训练期默认不写回缓存，避免内存涨）
    """
    if text_list is None:
        return None

    ensure_text_encoder_ready(exp)

    is_grid = isinstance(text_list, (list, tuple)) and len(text_list) > 0 and isinstance(text_list[0], (list, tuple))
    if is_grid:
        B = len(text_list)
        L = len(text_list[0]) if B > 0 else 0
        flat_texts = [t for row in text_list for t in row]
        if L == 0:
            emb_dim = int(getattr(exp, "text_embedding_dim", getattr(exp, "d_llm", 768)))
            return torch.zeros(B, 0, emb_dim, device=exp.device)
    else:
        B = len(text_list)
        L = 1
        flat_texts = list(text_list)

    def _encode_batch(missing_texts: List[str]) -> torch.Tensor:
        if len(missing_texts) == 0:
            d = int(getattr(exp, "d_llm", 768))
            return torch.zeros(0, d, device=exp.device)

        # Doc2Vec 分支（一般走不到这里，因为 ensure_text_encoder_ready 已 return）
        if getattr(exp, "Doc2Vec", False):
            vecs = torch.tensor([exp.text_model.infer_vector(t) for t in missing_texts],
                                device=exp.device, dtype=torch.float32)
            return vecs

        toks = exp.tokenizer(
            missing_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        for k, v in list(toks.items()):
            if isinstance(v, torch.Tensor):
                toks[k] = v.to(exp.device)

        if "input_ids" in toks:
            toks["input_ids"] = toks["input_ids"].to(torch.long)
        if "attention_mask" in toks:
            toks["attention_mask"] = toks["attention_mask"].to(torch.long)

        inp_emb = exp.llm_model.get_input_embeddings()(toks["input_ids"])

        if getattr(exp, "use_fullmodel", False):
            outputs = exp.llm_model(inputs_embeds=inp_emb, attention_mask=toks.get("attention_mask", None))
            tok_feat = outputs.last_hidden_state
        else:
            tok_feat = inp_emb

        attn_mask = toks.get("attention_mask", None)
        if attn_mask is not None:
            mask = attn_mask.unsqueeze(-1).float()
            if pool_token == "avg":
                summed = (tok_feat * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                per_text_vec = summed / denom
            elif pool_token == "max":
                tok_feat_masked = tok_feat.masked_fill(mask == 0, float("-inf"))
                per_text_vec, _ = tok_feat_masked.max(dim=1)
            elif pool_token == "min":
                tok_feat_masked = tok_feat.masked_fill(mask == 0, float("inf"))
                per_text_vec, _ = tok_feat_masked.min(dim=1)
            else:
                raise ValueError(f"Unsupported pool_token: {pool_token}")
        else:
            if pool_token == "avg":
                per_text_vec = tok_feat.mean(dim=1)
            elif pool_token == "max":
                per_text_vec, _ = tok_feat.max(dim=1)
            elif pool_token == "min":
                per_text_vec, _ = tok_feat.min(dim=1)
            else:
                raise ValueError(f"Unsupported pool_token: {pool_token}")

        need = int(getattr(exp, "d_llm", per_text_vec.shape[-1]))
        if per_text_vec.shape[-1] != need:
            if not hasattr(exp, "_proj_text2dllm"):
                exp._proj_text2dllm = nn.Linear(per_text_vec.shape[-1], need, bias=True).to(per_text_vec.device)
            per_text_vec = exp._proj_text2dllm(per_text_vec)

        return per_text_vec  # [N, d]

    per_text_vec = vec_from_cache(exp, flat_texts, _encode_batch)  # [B*L, d]
    d = per_text_vec.shape[-1]
    feat = per_text_vec.view(B, L, d)

    if reduce_time is None:
        return feat
    elif reduce_time == "avg":
        return feat.mean(dim=1)
    elif reduce_time == "max":
        return feat.max(dim=1).values
    elif reduce_time == "min":
        return feat.min(dim=1).values
    else:
        raise ValueError(f"Unsupported reduce_time: {reduce_time}")


def encode_with_bertopic_finbert(exp, text_list, reduce_time=None):
    """
    返回:
      - 若 reduce_time is None: [B, L, K]，K=exp.num_topics
      - 否则：按时间维聚合 -> [B, K]
    同样通过 vec_from_cache 缓存每个文本的 [K] 主题情感向量

    依赖 exp 在 __init__ 里已经初始化：
      exp.st_model, exp.finbert_tok, exp.finbert_cls, exp.bertopic, exp.num_topics, ...
    """
    if BERTopic is None or SentenceTransformer is None or AutoModelForSequenceClassification is None:
        raise RuntimeError("BERTopic/FinBERT deps not available. Please ensure bertopic, sentence-transformers, nltk are installed.")

    def _safe_sent_tokenize(txt: str) -> List[str]:
        if not isinstance(txt, str) or not txt:
            return []
        try:
            return sent_tokenize(txt)
        except LookupError:
            # 下载 nltk punkt
            if nltk is not None:
                for pkg in ("punkt", "punkt_tab"):
                    try:
                        nltk.data.find(f"tokenizers/{pkg}")
                    except LookupError:
                        try:
                            nltk.download(pkg, quiet=True)
                        except Exception:
                            pass
            try:
                return sent_tokenize(txt)
            except Exception:
                return [s for s in re.split(r'(?<=[\.!\?。！？])\s+', txt.strip()) if s]

    is_grid = isinstance(text_list, (list, tuple)) and len(text_list) > 0 and isinstance(text_list[0], (list, tuple))
    if is_grid:
        B = len(text_list)
        L = len(text_list[0]) if B > 0 else 0
        flat_texts = [t for row in text_list for t in row]
        if L == 0:
            return torch.zeros(B, 0, exp.num_topics, device=exp.device)
    else:
        B = len(text_list)
        L = 1
        flat_texts = list(text_list)

    K = int(exp.num_topics)

    def _encode_btf_batch(missing_texts: List[str]) -> torch.Tensor:
        if len(missing_texts) == 0:
            return torch.zeros(0, K, device=exp.device)

        all_sents: List[str] = []
        offsets = []
        for txt in missing_texts:
            sents = _safe_sent_tokenize(txt)
            st = len(all_sents)
            all_sents.extend(sents if len(sents) > 0 else [""])
            ed = len(all_sents)
            offsets.append((st, ed))

        # lazy fit（可选）
        if not getattr(exp, "_bertopic_fitted", False) and getattr(exp, "bertopic_lazy_fit", False):
            try:
                _ = exp.bertopic.fit(all_sents)
                exp._bertopic_fitted = True
                exp._topic_labels = [t for t in range(len(exp.bertopic.get_topics())) if t != -1]
            except Exception as e:
                print(f"[WARN] BERTopic lazy fit failed: {e}")

        try:
            sentence_topics, _ = exp.bertopic.transform(all_sents)
        except Exception:
            sentence_topics = [0 for _ in all_sents]

        def _finbert_score(batch_sents: List[str]) -> torch.Tensor:
            toks = exp.finbert_tok(batch_sents, return_tensors="pt", padding=True, truncation=True, max_length=256)
            toks = {k: v.to(exp.device) for k, v in toks.items()}
            with torch.no_grad():
                logits = exp.finbert_cls(**toks).logits
                probs = torch.softmax(logits, dim=-1)
                score = probs[:, 2] - probs[:, 0]
            return score.detach().cpu()

        scores = []
        step = 64
        for i in range(0, len(all_sents), step):
            scores.append(_finbert_score(all_sents[i:i + step]))
        scores = torch.cat(scores, dim=0)  # [N_sent] on CPU

        vecs = []
        for (st, ed) in offsets:
            t_ids = sentence_topics[st:ed]
            t_scores = scores[st:ed]
            sums = torch.zeros(K, dtype=torch.float32)
            cnts = torch.zeros(K, dtype=torch.float32)
            for tid, sc in zip(t_ids, t_scores):
                if tid == -1:
                    continue
                if 0 <= tid < K:
                    sums[tid] += sc
                    cnts[tid] += 1.0
                else:
                    sums[K - 1] += sc
                    cnts[K - 1] += 1.0
            vec = torch.where(cnts > 0, sums / torch.clamp(cnts, min=1.0), torch.zeros_like(sums))
            vecs.append(vec)

        return torch.stack(vecs, dim=0).to(exp.device)  # [N_docs, K]

    per_text_vec = vec_from_cache(exp, flat_texts, _encode_btf_batch)  # [B*L, K]
    feat = per_text_vec.view(B, L, K)

    if reduce_time is None:
        return feat
    elif reduce_time == "avg":
        return feat.mean(dim=1)
    elif reduce_time == "max":
        return feat.max(dim=1).values
    elif reduce_time == "min":
        return feat.min(dim=1).values
    else:
        raise ValueError(f"Unsupported reduce_time: {reduce_time}")
