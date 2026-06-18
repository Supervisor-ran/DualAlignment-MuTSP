# exp/exp_tools/__init__.py

from .tensor_ops import (
    norm,
    info_nce_loss,
    apply_time_mask,
    align_dec_mark,
    time_align_text_emb,
    feat_slice,
)

from .dataset_ops import (
    get_prior_y_safe,
    get_text_inputs,
)

from .text_cache import (
    cache_meta,
    cache_file_path,
    load_text_cache,
    save_text_cache,
    hash_text,
    vec_from_cache,
)

from .text_encoder import (
    safe_from_pretrained,
    ensure_text_encoder_ready,
    encode_text_list,
    encode_with_bertopic_finbert,
)
