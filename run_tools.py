import os
import gc
import random
import numpy as np
import torch
import multiprocessing as mp

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def setup_device_args(args):
    args.use_gpu = True if torch.cuda.is_available() else False
    if args.use_gpu and getattr(args, "use_multi_gpu", False):
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(i) for i in device_ids]
        args.gpu = args.device_ids[0]
    return args

def apply_llm_dim_defaults(args):
    mapping = {
        "BERT": 768, "GPT2": 768, "LLAMA2": 4096, "LLAMA3": 4096,
        "GPT2M": 1024, "GPT2L": 1280, "GPT2XL": 1600, "Doc2Vec": 64
    }
    if args.llm_model == "ClosedLLM":
        args.llm_model = "BERT"
        args.llm_dim = 768
        args.use_closedllm = 1
        return args
    if args.llm_model in mapping:
        args.llm_dim = mapping[args.llm_model]
    return args

def safe_close_loader(loader):
    if loader is None:
        return
    try:
        it = getattr(loader, "_iterator", None)
        if it is not None:
            try:
                it._shutdown_workers()
            except Exception:
                pass
            try:
                loader._iterator = None
            except Exception:
                pass
    except Exception:
        pass

def cleanup_exp_cached_loaders(exp):
    if exp is None:
        return
    names = [
        "train_loader", "vali_loader", "val_loader", "test_loader",
        "loader", "data_loader", "train_data_loader", "valid_data_loader", "test_data_loader"
    ]
    for n in names:
        if hasattr(exp, n):
            try:
                safe_close_loader(getattr(exp, n))
            except Exception:
                pass
            try:
                delattr(exp, n)
            except Exception:
                pass

def reap_dataloader_workers():
    try:
        children = mp.active_children()
        for p in children:
            try:
                p.terminate()
            except Exception:
                pass
        for p in children:
            try:
                p.join(timeout=1.0)
            except Exception:
                pass
    except Exception:
        pass

def hard_cleanup(exp):
    cleanup_exp_cached_loaders(exp)

    try:
        if hasattr(exp, "model") and exp.model is not None:
            try:
                exp.model.to("cpu")
            except Exception:
                pass
    except Exception:
        pass

    big_attrs = [
        'model', 'mlp', 'mlp_proj', 'weight1', 'weight2',
        'llm_model', 'tokenizer', 'st_model', 'finbert_cls', 'finbert_tok',
        'bertopic', 'text_model'
    ]
    for name in big_attrs:
        if hasattr(exp, name):
            try:
                obj = getattr(exp, name)
                try:
                    if hasattr(obj, "to"):
                        obj.to("cpu")
                except Exception:
                    pass
                del obj
            except Exception:
                pass
            try:
                delattr(exp, name)
            except Exception:
                pass

    try:
        import matplotlib.pyplot as plt
        plt.close("all")
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def final_cleanup(exp):
    cleanup_exp_cached_loaders(exp)
    hard_cleanup(exp)
    reap_dataloader_workers()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def decide_enc_in(root_path, data_path):
    """
    Build csv path from root_path + data_path, read it, drop specified columns if they exist,
    and return the remaining column count as enc_in.
    """
    import os
    import pandas as pd

    csv_path = os.path.join(root_path, data_path)
    df = pd.read_csv(csv_path)

    drop_cols = ['date', 'start_date', 'end_date', 'prior_history_avg', 'Final_Search_4']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    enc_in = df.shape[1]
    return enc_in

