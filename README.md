# DualAlignment-MuTSP

Official implementation of **Dual Alignment Networks for Cross-domain Multimodal Time Series Prediction**.

This repository provides the cleaned implementation of **DA-Transformer**, a multimodal time series forecasting model that uses numerical time series and timestamped textual information for long-term forecasting.

## Overview

DA-Transformer is designed for cross-domain multimodal time series prediction. The model combines time-series representations and text representations through:

- Semantic alignment between numerical and textual time-token representations
- Movement alignment for temporal change patterns
- Cross-modal attention between time-series and text features
- BERT-based text encoding for timestamped textual inputs

The public version of this repository focuses on **DATransformer** and **BERT**.

## Repository Structure

```text
.
├── data/                    # Time-MMD datasets
├── data_NewsForecast/        # NewsForecast datasets
├── data_TimeCAP/             # TimeCAP datasets
├── data_TTC/                 # TTC datasets
├── data_provider/            # Dataset loading utilities
├── exp/                      # Experiment code
│   ├── exp_long_term_forecasting.py
│   └── exp_tools/            # Helper modules for text encoding, caching, tensor ops, etc.
├── layers/                   # Neural network layers
├── models/                   # Model definitions
│   └── DATransformer.py
├── utils/                    # Metrics, tools, time features, and argument printing
├── run.py                    # Main entry point
├── run_tools.py              # Runtime utilities
├── pyproject.toml            # Project dependencies
└── uv.lock                   # Locked dependency file
```

## Environment

This project uses `uv`.

```bash
uv sync
```

To check the environment:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Example Usage

### Public Health Dataset

```bash
uv run python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id test_1 \
  --model DATransformer \
  --data custom \
  --root_path ./data/Public_Health \
  --data_path US_FLURATIO_Week.csv \
  --features S \
  --seq_len 40 \
  --label_len 12 \
  --pred_len 12 \
  --des Exp \
  --seed 2021 \
  --type_tag "#F#" \
  --text_len 4 \
  --prompt_weight 0.1 \
  --pool_type avg \
  --save_name result_tra_bert \
  --llm_model BERT \
  --huggingface_token NA \
  --use_fullmodel 0 \
  --train_epochs 10 \
  --use_cross_modality \
  --use_alignment_1 \
  --use_alignment_2 \
  --use_align_repre \
  --freeze_p_txt \
  --freeze_text_mlp
```

### TTC Medical Dataset

```bash
uv run python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id test_1 \
  --model DATransformer \
  --data custom \
  --root_path ./data_TTC/Medical \
  --data_path patient_20415.csv \
  --features M \
  --seq_len 40 \
  --label_len 12 \
  --pred_len 12 \
  --des Exp \
  --seed 2021 \
  --type_tag "#F#" \
  --text_len 4 \
  --prompt_weight 0.2 \
  --pool_type avg \
  --save_name result_tra_bert \
  --llm_model BERT \
  --huggingface_token NA \
  --use_fullmodel 0 \
  --train_epochs 3 \
  --use_cross_modality \
  --use_alignment_1 \
  --use_alignment_2 \
  --use_align_repre \
  --freeze_p_txt \
  --freeze_text_mlp
```

## Important Notes

- The cleaned public version supports **DATransformer only**.
- The text encoder is fixed to **BERT**.
- Generated files such as checkpoints, experiment outputs, model weights, and pickle cache files are ignored.
- Dataset CSV files are included for reproducibility.
- Pickle cache files are not included and will be regenerated when needed.

## Outputs

Training checkpoints are saved under:

```text
checkpoints/
```

Test results and metrics are saved under:

```text
results/
test_results/
```

These directories are ignored by Git.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{liu2026dualalignment,
  title={Dual Alignment Networks for Cross-domain Multimodal Time Series Prediction},
  author={Liu, Ran and Mine, Tsunenori},
  booktitle={ECML PKDD},
  year={2026}
}
```
