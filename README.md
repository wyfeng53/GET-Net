# GET-Net: A Geometry-Embedded Time-Series Network for Flight Trajectory Prediction in Air Traffic Management

## Introduction
This repository operationalizes the GET-Net prototype (``GET-Net: A Geometry-Embedded Time-Series Network for Flight Trajectory Prediction in Air Traffic Management``) inside a geometry-aware PatchTST harness. The backbone keeps PatchTST’s temporal modeling while injecting spherical-Earth priors and kinematic cues so forecasts stay on great-circle tracks in cruise yet remain agile through sharp turns or terminal maneuvers.

The code mirrors the paper’s two-branch design—PatchTST for long-range temporal context plus a Geo-Kinematic Feature Extraction (GKFE) branch for great-circle distances, bearings, and multi-scale change rates—fused by Adaptive Confidence Fusion (ACF) to emphasize whichever branch is more reliable per flight phase. Experiments run on the sanitized 113-day ADS-B corpus under `fcz_data/PEK_NOT_ON_GROUND/`, and the same scripts reproduce baseline comparisons, ablations, and phase diagnostics with minimal refactoring for other multi-variate datasets.

## Repository Structure
```
patchTST_changed1/
│  run_longExp.py            # Unified CLI entry for training / evaluation
│  README.md                 # This document
├─data_provider/             # Dataset factories and trajectory loaders
├─exp/                       # Experiment manager (Exp_Main, logging helpers)
├─layers/                    # Embeddings, PatchTST backbone, attention blocks
├─models/                    # Ready-to-use forecasting architectures
├─fcz_data/                  # Sanitized trajectory samples + dataset README
├─checkpoints/               # Saved weights (created at runtime)
├─logs/                      # Training/evaluation logs
└─results/                   # Aggregated metrics & tables
```

## Package Requirements
Only the essential dependencies are listed below (install via `pip` or `conda`).

- Python 3.8.20
- PyTorch 2.4.1+cu124 (plus torchvision 0.19.1+cu124 / torchaudio 2.4.1+cu124)
- numpy 1.24.x
- pandas 2.0.x
- scikit-learn 1.3.x

## Instructions
### 1. Clone & Environment
```bash
git clone <your-repo-url>
cd patchTST_changed1
python -m venv .venv && .venv\Scripts\activate  # or use conda
pip install -r requirements.txt  # or install the packages listed above
```

### 2. Prepare Data
- Place sanitized CSV trajectories inside `fcz_data/PEK_NOT_ON_GROUND/<FLIGHT_ID>_<YY_MM-range>/{train,val,test}/`.
- Refer to `fcz_data/README.md` for the exact CSV schema and naming convention.

### 3. Training
`run_longExp.py` controls every experiment through CLI arguments (see the script for the full list). A typical training launch looks like:
```bash
python run_longExp.py \
	--is_training 1 \
	--model_id geo_patchtst_ft \
	--model PatchTST \
	--data custom \
	--root_path ./fcz_data/PEK_NOT_ON_GROUND/ \
	--data_path PEK_NOT_ON_GROUND \
	--features M \
	--target Height \
	--seq_len 96 --label_len 24 --pred_len 32 \
	--d_model 512 --n_heads 8 --e_layers 3 --d_layers 1 \
	--batch_size 128 --learning_rate 1e-4 --train_epochs 50
```
Key switches:
- `--model`: choose among architectures implemented under `models/` (PatchTST, iTransformer, DLinear, GET, etc.).
- `--root_path` / `--data_path`: point to the dataset root and specific sub-folder.
- `--seq_len`, `--label_len`, `--pred_len`: control the encoder/decoder horizons.
- `--dynamics_type`: toggles optional geometry-aware priors (e.g., `GET`, `neural_ode`, `hamiltonian`).
- `--itr`: number of repeated runs; checkpoints/logs are written to `./checkpoints` and `./logs`.

### 4. Evaluation / Inference
```bash
python run_longExp.py \
	--is_training 0 \
	--model_id geo_patchtst_ft \
	--model PatchTST \
	--data custom \
	--root_path ./fcz_data/PEK_NOT_ON_GROUND/ \
	--data_path PEK_NOT_ON_GROUND \
	--seq_len 96 --label_len 24 --pred_len 32 \
	--model_path ./checkpoints/<experiment>/<best_model>.pth
```
When `--is_training 0`, the script loads the checkpoint specified by `--model_path` and reports the metrics defined in `exp/exp_main.py`.

#### Frequently Used Arguments
- `--features {M,S,MS}`：multi-variate vs single-channel setups.
- `--patch_len`, `--stride`, `--padding_patch`：PatchTST patch extraction rules.
- `--revin`, `--affine`, `--subtract_last`：RevIN normalization switches.
- `--dropout`, `--fc_dropout`, `--head_dropout`：regularization knobs for trunk / projection heads.
- `--learning_rate`, `--lradj`, `--pct_start`：optimizer schedule.
- `--train_epochs`, `--patience`, `--batch_size`：training budget and early stopping.
- `--des`, `--model_id`: used to label experiments and organize outputs.

## Dataset
- Example trajectories and a detailed schema explanation live in `fcz_data/README.md`.
- The loaders in `data_provider/data_loader.py` support both the sanitized flight corpus and generic ETT-style benchmarks; select via `--data` and `--data_path`.
- To plug in new flights, keep the same naming template (`Variflight_<FLIGHT_ID>_YYYYMMDD.csv`) and add the folder under `fcz_data/PEK_NOT_ON_GROUND/` with `train/val/test` splits.

## Results & Logs
- Intermediate artifacts (metrics, tables, qualitative analyses) are written to `results/`.
- Console outputs plus per-epoch statistics are mirrored to `logs/LongForecasting/` for later inspection.
