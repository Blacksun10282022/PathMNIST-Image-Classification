
# Small-Image Classification Pipeline (Keras & scikit-learn)

**What**: 9-class classification on small RGB images (28×28×3, ~40k samples).  
**How**: RandomForest baseline + lightweight CNN (Keras), with one-command train/eval/export and auto-saved reports.  
**Why**: Reproducible, engineering-ready pipeline for coursework-to-production transition.

## Quick Start
```bash
# 1) create env (conda or pip)
# conda create -n sip python=3.11 -y && conda activate sip
pip install -r requirements.txt

# 2) run a fast smoke test on sample data
python -m scripts.train --model rf --use-sample
python -m scripts.train --model cnn --use-sample

# 3) run on full data (place .npy files into data/ as described below)
# python -m scripts.train --model cnn
# python -m scripts.eval  --run outputs/2025-10-12_12-00-00
```

## Data
This repo **does not** include the full dataset. To run on full data, put these files in `data/`:
- `X_train.npy` `(N_train, 28, 28, 3)`
- `y_train.npy` `(N_train,)`
- `X_test.npy`  `(N_test, 28, 28, 3)`
- `y_test.npy`  `(N_test,)`

For CI and quick verification we include a tiny `data/samples/` subset.

## Results (fill after first full run)
- RF Baseline: `acc = __%`, `macro-F1 = __`
- CNN (no tune): `acc = __%`, `macro-F1 = __`

## Repo Layout
- `src/`: modular data loading, models, metrics, utils, inference
- `scripts/`: CLI entry points for training/evaluation/export
- `configs/`: YAML configs for training and CNN model
- `data/`: local data folder (ignored by git). Includes a tiny `samples/` subset.
- `outputs/`: generated artifacts (ignored by git)

## License
MIT
