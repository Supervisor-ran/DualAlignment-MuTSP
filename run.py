import argparse
import os

import torch

from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_imputation import Exp_Imputation
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from run_tools import (
    apply_llm_dim_defaults,
    decide_enc_in,
    final_cleanup,
    reap_dataloader_workers,
    set_seed,
    setup_device_args,
)
from utils.print_args import print_args


os.environ["TOKENIZERS_PARALLELISM"] = "true"


def str2bool(value):
    """Parse boolean command-line values safely."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_setting(args):
    """Build the experiment setting name.

    Keep the original training-setting fields so existing result/checkpoint names
    remain compatible with the previous run.py behavior.
    """
    return (
        "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expa{}_dc{}_"
        "fc{}_eb{}_d{}_{}_{}_{}"
    ).format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        args.use_alignment_1,
        args.use_alignment_2,
        args.use_cross_modality,
    )


def get_experiment_class(task_name):
    if task_name == "long_term_forecast":
        return Exp_Long_Term_Forecast
    if task_name == "short_term_forecast":
        return Exp_Short_Term_Forecast
    if task_name == "imputation":
        return Exp_Imputation
    if task_name == "anomaly_detection":
        return Exp_Anomaly_Detection
    if task_name == "classification":
        return Exp_Classification
    raise ValueError(f"Unsupported task_name: {task_name}")


def prepare_args(args):
    if args.model != "DATransformer":
        raise ValueError("This cleaned entry point only supports DATransformer.")

    args = apply_llm_dim_defaults(args)

    # Keep the original univariate intent from run.py:
    # when features='S', the dataloader feeds one numeric channel, so the model
    # projection input must be 1. Do not infer the full CSV column count here.
    if args.features == "S":
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
        print("Using univariate mode: features=S, enc_in=1, dec_in=1, c_out=1")
    elif getattr(args, "enc_in", 0) in (0, None):
        args.enc_in = decide_enc_in(args.root_path, args.data_path)
        print(f"Auto set enc_in={args.enc_in} for {args.root_path}/{args.data_path}")

    args.text_emb = args.pred_len
    return args


def run_once(args):
    domain = os.path.basename(os.path.normpath(args.root_path))
    print(f"Now running domain={domain}, model={args.model}")

    args = prepare_args(args)

    print(f"Now using seed {args.seed}")
    set_seed(args.seed)

    args = setup_device_args(args)
    print(f"CUDA available: {torch.cuda.is_available()}")

    print("Args in experiment:")
    print_args(args)

    Exp = get_experiment_class(args.task_name)
    setting = build_setting(args)

    exp = None
    try:
        if args.is_training:
            for _ in range(args.itr):
                exp = Exp(args)

                print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
                exp.train(setting)

                print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
                _ = exp.test(setting, test=1)

                try:
                    import matplotlib.pyplot as plt

                    plt.close("all")
                except Exception:
                    pass

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                reap_dataloader_workers()
        else:
            exp = Exp(args)
            print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            _ = exp.test(setting, test=1)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        final_cleanup(exp)


def build_parser():
    parser = argparse.ArgumentParser(description="DATransformer runner")

    # Basic config
    parser.add_argument(
        "--task_name",
        type=str,
        default="long_term_forecast",
        choices=[
            "long_term_forecast",
            "short_term_forecast",
            "imputation",
            "classification",
            "anomaly_detection",
        ],
    )
    parser.add_argument("--is_training", type=int, default=1, choices=[0, 1])
    parser.add_argument("--model_id", type=str, default="test")
    parser.add_argument("--model", type=str, default="DATransformer", choices=["DATransformer"])

    # Data loader
    parser.add_argument("--data", type=str, default="custom")
    parser.add_argument("--root_path", type=str, default="./data/Public_Health")
    parser.add_argument("--data_path", type=str, default="US_FLURATIO_Week.csv")
    parser.add_argument("--features", type=str, default="S")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")

    # Forecasting task
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--seasonal_patterns", type=str, default="Monthly")
    parser.add_argument("--inverse", action="store_true", default=False)

    # Imputation / anomaly detection
    parser.add_argument("--mask_rate", type=float, default=0.25)
    parser.add_argument("--anomaly_ratio", type=float, default=0.25)

    # Model definition
    parser.add_argument("--expand", type=int, default=2)
    parser.add_argument("--d_conv", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--num_kernels", type=int, default=6)
    parser.add_argument("--enc_in", type=int, default=0)
    parser.add_argument("--dec_in", type=int, default=7)
    parser.add_argument("--text_in", type=int, default=1)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--distil", action="store_false", default=True)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--dropout_DA", type=float, default=0.1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument("--output_attention", action="store_true")
    parser.add_argument("--channel_independence", type=int, default=1)
    parser.add_argument("--decomp_method", type=str, default="moving_avg")
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--down_sampling_layers", type=int, default=0)
    parser.add_argument("--down_sampling_window", type=int, default=1)
    parser.add_argument("--down_sampling_method", type=str, default=None)
    parser.add_argument("--seg_len", type=int, default=48)

    # Optimization
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--itr", type=int, default=1)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--des", type=str, default="test")
    parser.add_argument("--loss", type=str, default="MSE")
    parser.add_argument("--lradj", type=str, default="type1")
    parser.add_argument("--use_amp", action="store_true", default=False)

    # GPU
    parser.add_argument("--use_gpu", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    # Compatibility options for shared LLM/runtime helpers.
    # They are kept to avoid breaking run_tools/model code that may read these attributes.
    parser.add_argument("--memory_limit_for_time_llm", type=str, default="off")
    parser.add_argument("--prompt_max_tokens", type=int, default=0)
    parser.add_argument("--llm_chunk", type=int, default=0)
    parser.add_argument("--num_tokens", type=int, default=0)

    # De-stationary projector params.
    # Kept because utils.print_args and some shared experiment code expect them.
    parser.add_argument("--p_hidden_dims", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--p_hidden_layers", type=int, default=2)

    # Text / LLM-related config used by DATransformer text branch
    parser.add_argument("--llm_model", type=str, default="BERT")
    parser.add_argument("--llm_dim", type=int, default=768)
    parser.add_argument("--llm_layers", type=int, default=6)
    parser.add_argument("--text_path", type=str, default="None")
    parser.add_argument("--type_tag", type=str, default="#F#")
    parser.add_argument("--text_len", type=int, default=4)
    parser.add_argument("--learning_rate2", type=float, default=1e-2)
    parser.add_argument("--learning_rate3", type=float, default=1e-3)
    parser.add_argument("--prompt_weight", type=float, default=0.1)
    parser.add_argument("--pool_type", type=str, default="avg")
    parser.add_argument("--date_name", type=str, default="end_date")
    parser.add_argument("--addHisRate", type=float, default=0.5)
    parser.add_argument("--init_method", type=str, default="normal")
    parser.add_argument("--learning_rate_weight", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--save_name", type=str, default="result_longterm_forecast")
    parser.add_argument("--use_fullmodel", type=int, default=0)
    parser.add_argument("--use_closedllm", type=int, default=0)
    parser.add_argument("--huggingface_token", type=str, default=None)
    parser.add_argument("--text_dim", type=int, default=12)
    parser.add_argument("--use_detrend_reverse", action="store_true", default=False)

    # Alignment
    parser.add_argument("--use_alignment_1", action="store_true", default=False)
    parser.add_argument("--use_alignment_2", action="store_true", default=False)
    parser.add_argument("--use_align_repre", action="store_true", default=False)
    parser.add_argument(
        "--alignment_mode",
        type=str,
        default="cosine",
        choices=["cosine", "infonce"],
    )
    parser.add_argument("--freeze_text_mlp", action="store_true", default=False)
    parser.add_argument("--freeze_text_mlp_last", action="store_true", default=False)
    parser.add_argument("--freeze_p_txt", action="store_true", default=False)
    parser.add_argument("--da_self_attention", action="store_true", default=False)
    parser.add_argument("--da_n_heads", type=int, default=4)
    parser.add_argument("--da_sa_dropout", type=float, default=0.0)
    parser.add_argument("--alignment1_weight", type=float, default=0.2)
    parser.add_argument("--alignment_delta", type=int, default=0)
    parser.add_argument("--alignment_tau", type=float, default=0.07)
    parser.add_argument("--align_dim", type=int, default=64)
    parser.add_argument("--alignment2_tau", type=float, default=0.07)
    parser.add_argument("--alignment2_weight", type=float, default=0.2)
    parser.add_argument(
        "--use_automatic_fuse_weight_for_alignment",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
    )
    parser.add_argument("--alignment2_ts2ts_weight", type=float, default=1)
    parser.add_argument("--alignment2_txt2ts_weight", type=float, default=1)
    parser.add_argument("--alignment2_txt2txt_weight", type=float, default=1)

    # Cross-modality
    parser.add_argument("--use_cross_modality", action="store_true", default=False)
    parser.add_argument(
        "--cross_modality_mode",
        type=str,
        default="seperate_cross",
        choices=["seperate_cross", "ts-to-text", "text-to-ts"],
    )
    parser.add_argument("--cross_modality_n_heads", type=int, default=8)

    # Compatibility options for shared experiment code.
    parser.add_argument("--universal_max_pred", type=int, default=720)
    parser.add_argument("--mask_ratio", type=float, default=0.5)

    # Multivariate independent channel switches
    parser.add_argument("--multivar_indep", action="store_true", default=False)
    parser.add_argument("--multivar_loss_indep", action="store_true", default=False)

    # Optional module freezing list
    parser.add_argument(
        "--freeze_mlp_list",
        type=str,
        nargs="*",
        default=[],
        help="Names of model MLP/Linear modules to freeze.",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_once(args)
