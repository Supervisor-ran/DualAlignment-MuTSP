# exp/exp_tools/text_cache.py

from __future__ import annotations

import os
import pickle
import hashlib
import pathlib
from typing import Callable, List, Dict, Any

import numpy as np
import torch


def cache_meta(exp) -> Dict[str, Any]:
    return dict(
        data_path=os.path.basename(exp.args.data_path),
        llm_model=getattr(exp.args, "llm_model", None),
        llm_layers=getattr(exp.args, "llm_layers", None),
        use_fullmodel=int(getattr(exp, "use_fullmodel", False)),
        pool_type=getattr(exp, "pool_type", "avg"),
        d_llm=int(getattr(exp, "d_llm", 0)),
        text_len=int(getattr(exp, "text_len", 0)),
        use_btf=int(getattr(exp, "use_bertopic_finbert", False)),
        num_topics=int(getattr(exp, "num_topics", 0)) if getattr(exp, "use_bertopic_finbert", False) else None,
    )


def cache_file_path(exp) -> str:
    root = getattr(exp.args, "root_path", ".")
    base = os.path.splitext(os.path.basename(exp.args.data_path))[0]
    tag = [
        base,
        f"LLM-{getattr(exp.args, 'llm_model', 'NA')}",
        f"L{getattr(exp.args, 'llm_layers', 0)}",
        "full" if getattr(exp, "use_fullmodel", False) else "emb",
        f"pool-{getattr(exp, 'pool_type', 'avg')}",
        f"d{getattr(exp, 'd_llm', 0)}",
        f"T{getattr(exp, 'text_len', 0)}",
    ]
    if getattr(exp, "use_bertopic_finbert", False):
        tag.append(f"btf-K{getattr(exp, 'num_topics', 0)}")
    fname = "__".join(tag) + ".pkl"
    return os.path.join(root, fname)


def load_text_cache(exp) -> Dict[str, Any]:
    """
    在 exp 上挂载：
      - exp._text_vec_cache
      - exp._cache_path
      - exp._cache_dirty
      - exp._cache_flush_every
      - exp._text_cache_max_items
    """
    if hasattr(exp, "_text_vec_cache"):
        return exp._text_vec_cache

    path = cache_file_path(exp)
    cache = {"__meta__": cache_meta(exp), "kv": {}}
    print(f"[cache] Loading text cache from {path}")

    obj = None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except FileNotFoundError:
        print(f"[cache] No cache file at {path}; starting fresh.")
    except Exception as e:
        print(f"[cache] Failed to read cache {path}: {e}; starting fresh.")

    if isinstance(obj, dict) and obj.get("__meta__") == cache_meta(exp) and "kv" in obj:
        cache = obj
    elif obj is not None:
        print(f"[cache] Cache meta mismatch at {path}; ignoring old cache.")

    exp._text_vec_cache = cache
    exp._cache_path = path
    exp._cache_dirty = False
    exp._cache_flush_every = int(getattr(exp.args, "cache_flush_every", 1000))
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

    exp._text_cache_max_items = int(getattr(exp, "text_cache_max_items", 200000))
    return cache


def save_text_cache(exp, force: bool = False) -> None:
    if not getattr(exp, "_cache_dirty", False) and not force:
        return
    try:
        with open(exp._cache_path, "wb") as f:
            pickle.dump(exp._text_vec_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved text vector cache in {exp._cache_path}")
        exp._cache_dirty = False
    except Exception as e:
        print(f"[WARN] save cache failed: {e}")


def hash_text(t: str) -> str:
    if not isinstance(t, str):
        t = "" if t is None else str(t)
    return hashlib.sha1(t.encode("utf-8")).hexdigest()


def vec_from_cache(exp, texts: List[str], make_missing: Callable[[List[str]], torch.Tensor]) -> torch.Tensor:
    """
    从缓存加载一批文本向量。
    - 训练期：默认不把缺失项写回缓存（exp.text_cache_train_write=False），避免缓存键无限增长导致内存暴涨。
    - 验证/测试期：写回缓存，并对缓存体量做上限控制（超限简单淘汰最早的若干个键）。

    返回: torch.Tensor [N, d]（在 exp.device 上）
    """
    cache = load_text_cache(exp)
    kv: Dict[str, np.ndarray] = cache["kv"]

    # 1) 去重
    keys = [hash_text(t if isinstance(t, str) else ("" if t is None else str(t))) for t in texts]
    uniq_map: Dict[str, str] = {}
    uniq_texts: List[str] = []
    for t, k in zip(texts, keys):
        if k not in uniq_map:
            uniq_map[k] = t
            uniq_texts.append(t)

    # 2) 缺失项
    missing_keys = [k for k in uniq_map.keys() if k not in kv]
    missing_texts = [uniq_map[k] for k in missing_keys]

    new_kv_local: Dict[str, torch.Tensor] = {}

    # 3) 编码缺失项
    if len(missing_texts) > 0:
        with torch.no_grad():
            new_vecs = make_missing(missing_texts)  # torch [M, d] on device
            if not isinstance(new_vecs, torch.Tensor):
                raise TypeError("vec_from_cache: make_missing did not return a torch.Tensor")
            new_vecs = new_vecs.detach()

        # 4) 决定是否写回缓存
        write_back = (not exp.model.training) and True
        if exp.model.training and bool(getattr(exp, "text_cache_train_write", False)):
            write_back = True

        if write_back:
            as_np = new_vecs.cpu().numpy().astype("float32")
            for k, v in zip(missing_keys, as_np):
                kv[k] = v
            exp._cache_dirty = True

            # 超限淘汰（FIFO 近似）
            max_items = int(getattr(exp, "_text_cache_max_items", 200000))
            overflow = max(0, len(kv) - max_items)
            if overflow > 0:
                it = iter(kv.keys())
                for _ in range(overflow):
                    try:
                        first_k = next(it)
                    except StopIteration:
                        break
                    kv.pop(first_k, None)

            # 分批落盘
            if len(missing_texts) >= int(getattr(exp, "_cache_flush_every", 1000)):
                save_text_cache(exp)
        else:
            # 训练期不写缓存：只存在本批局部 dict
            for k, v in zip(missing_keys, new_vecs):
                new_kv_local[k] = v

    # 5) 组装输出
    out_list = []
    d_hint = None

    for k in keys:
        if k in new_kv_local:
            v = new_kv_local[k]
            if d_hint is None:
                d_hint = int(v.shape[-1])
            out_list.append(v)
        else:
            v = kv.get(k, None)
            if v is None:
                if d_hint is None:
                    d_hint = int(getattr(exp, "d_llm", 768))
                out_list.append(torch.zeros(d_hint, device=exp.device, dtype=torch.float32))
            else:
                if d_hint is None:
                    d_hint = int(v.shape[-1])
                out_list.append(torch.from_numpy(v).to(exp.device))

    if len(out_list) == 0:
        d = int(getattr(exp, "d_llm", 1))
        return torch.zeros(0, d, device=exp.device, dtype=torch.float32)

    return torch.stack(out_list, dim=0)  # [N, d]
