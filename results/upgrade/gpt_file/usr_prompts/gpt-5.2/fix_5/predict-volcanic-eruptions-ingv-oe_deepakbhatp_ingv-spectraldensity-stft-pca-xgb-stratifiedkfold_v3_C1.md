# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Given readings from several seismic sensors around a volcano, estimate how long it will be until the next eruption.

## Metric
Mean absolute error (MAE) between the predicted loss and the actual loss.

## Submission Format
For every id in the test set, you should predict the time until the next eruption. The file should contain a header and have the following format:

```
segment_id,time_to_eruption
1,1
2,2
3,3
etc.
```

## Data
### Dataset Description

#### Files
**train.csv** Metadata for the train files.

- `segment_id`: ID code for the data segment. Matches the name of the associated data file.
- `time_to_eruption`: The target value, the time until the next eruption.

**[train|test]/*.csv**: the data files. Each file contains ten minutes of logs from ten different sensors arrayed around a volcano. The readings have been normalized within each segment, in part to ensure that the readings fall within the range of int16 values. If you are using the Pandas library you may find that you still need to load the data as float32 due to the presence of some nulls.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
sklearn-pandas==2.2.0
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (70 lines)
            sample_submission.csv (445 lines)
            sample_submission.csv.zip (2.8 kB)
            test.zip (514.3 MB)
            train.csv (3988 lines)
            train.csv.zip (39.1 kB)
            train.zip (4.6 GB)
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
            test/
                1003520023.csv (60002 lines)
                1004346803.csv (60002 lines)
                ... and 442 other files
                test/
            train/
                1000015382.csv (60002 lines)
                1000554676.csv (60002 lines)
                ... and 3985 other files
                train/
        input/
            description.md (70 lines)
            sample_submission.csv (445 lines)
            sample_submission.csv.zip (2.8 kB)
            test.zip (514.3 MB)
            train.csv (3988 lines)
            train.csv.zip (39.1 kB)
            train.zip (4.6 GB)
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
            test/
                1003520023.csv (60002 lines)
                1004346803.csv (60002 lines)
                ... and 442 other files
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
            train/
                1000015382.csv (60002 lines)
                1000554676.csv (60002 lines)
                ... and 3985 other files
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
        working/
            predict-volcanic-eruptions-ingv-oe/
                description.md (70 lines)
                sample_submission.csv (445 lines)
                ... and 5 other files
                predict-volcanic-eruptions-ingv-oe/
                test/
                    1003520023.csv (60002 lines)
                    1004346803.csv (60002 lines)
                    ... and 442 other files
                    test/
                train/
                    1000015382.csv (60002 lines)
                    1000554676.csv (60002 lines)
                    ... and 3985 other files
                    train/
```

-> data/predict-volcanic-eruptions-ingv-oe/sample_submission.csv has 444 rows and 2 columns.
The columns are: segment_id, time_to_eruption

-> data/predict-volcanic-eruptions-ingv-oe/test/1003520023.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1004346803.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1007996426.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1009749143.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1016956864.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1024522044.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> data/predict-volcanic-eruptions-ingv-oe/test/1028325789.csv has 60001 rows and 10 columns.
The columns are: sensor_1, sensor_2, sensor_3, sensor_4, sensor_5, sensor_6, sensor_7, sensor_8, sensor_9, sensor_10

-> (stopped after 10 files for performance)

# 5. Target score

5140185.634133191

# 6. Current score

4020988.0

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 24402200.0) has done: 'The timeout is dominated by per-segment feature extraction: reading ~4k train + 444 test CSVs and running 10× SciPy STFT per file (tens of thousands of STFTs) is too slow. I keep the exact same features and model training logic, but make extraction faster by (1) using a much faster CSV reader path (pyarrow when available, otherwise a tight pandas path), (2) avoiding expensive `np.quantile` by using equivalent `np.partition` for 5/50/95% quantiles, (3) parallelizing feature extraction across CPU cores with ordered results to preserve determinism, and (4) removing unnecessary DataFrame copies/indexing overhead in the fold loop. These changes are computation-order/caching/parallelism optimizations only; they preserve feature definitions, fold semantics, and model settings (including early stopping) so accuracy stays the same up to negligible floating-point differences.'
- What this solution (achieved 4020988.0) has done: 'The timeout is dominated by feature extraction: each segment loads a 60k×10 CSV and then runs 10 separate `scipy.signal.stft` calls plus repeated `np.partition` work for quantiles, across ~4k train + 444 test segments. To keep identical features/logic but cut constant factors, the refactor (a) reads CSVs with explicit `dtype`/`na_values` to avoid extra conversions and `.replace/.fillna`, (b) computes all requested quantiles in a single `np.quantile(..., method="linear")` call (equivalent to the current linear interpolation semantics), and (c) computes STFT for all 10 channels in one vectorized `signal.stft(..., axis=0)` call, then derives per-sensor stats from the same result. Additionally, it avoids costly per-element Python loops by using precomputed column names and building the feature dict with bulk updates, while keeping the same multiprocessing structure and the same XGBoost training/early-stopping behavior.'

# 9. Code solution

## === cell 0
import os
import gc
import glob
import numpy as np
import pandas as pd

import scipy
from scipy import signal
import random

from sklearn.decomposition import PCA
import xgboost

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BASE_PATH = "/kaggle/data/predict-volcanic-eruptions-ingv-oe"
TRAIN_META_PATH = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")

assert os.path.exists(TRAIN_META_PATH), f"Missing: {TRAIN_META_PATH}"
assert os.path.exists(SAMPLE_SUB_PATH), f"Missing: {SAMPLE_SUB_PATH}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"




## === cell 1
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]
BASE_STATS = ("mean", "std", "min", "max", "p05", "p50", "p95")


def _safe_float32_df(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            path,
            engine="pyarrow",
            dtype={c: np.float32 for c in SENSOR_COLS},
            na_values=[
                "",
                "nan",
                "NaN",
                "NULL",
                "null",
                "inf",
                "-inf",
                "Infinity",
                "-Infinity",
            ],
        )
    except Exception:
        df = pd.read_csv(
            path,
            dtype={c: np.float32 for c in SENSOR_COLS},
            na_values=[
                "",
                "nan",
                "NaN",
                "NULL",
                "null",
                "inf",
                "-inf",
                "Infinity",
                "-Infinity",
            ],
        )
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0.0)
    return df


def _quantiles_linear(X: np.ndarray, probs=(0.05, 0.50, 0.95)) -> np.ndarray:
    return np.quantile(X, probs, axis=0, method="linear").astype(np.float32, copy=False)


def compute_segment_features(seg_path: str) -> dict:
    df = _safe_float32_df(seg_path)  # shape ~ (60001, 10)
    X = df.to_numpy(dtype=np.float32, copy=False)  # (T, C)
    feats = {}

    mu = X.mean(axis=0, dtype=np.float32)
    sd = X.std(axis=0, dtype=np.float32)
    mn = X.min(axis=0)
    mx = X.max(axis=0)
    qs = _quantiles_linear(X, (0.05, 0.50, 0.95))
    p05, p50, p95 = qs[0], qs[1], qs[2]

    for i in range(X.shape[1]):
        s = i + 1
        feats[f"sensor_{s}_mean"] = float(mu[i])
        feats[f"sensor_{s}_std"] = float(sd[i])
        feats[f"sensor_{s}_min"] = float(mn[i])
        feats[f"sensor_{s}_max"] = float(mx[i])
        feats[f"sensor_{s}_p05"] = float(p05[i])
        feats[f"sensor_{s}_p50"] = float(p50[i])
        feats[f"sensor_{s}_p95"] = float(p95[i])

    f, t, Zxx = signal.stft(
        X, nperseg=256, noverlap=128, boundary=None, padded=False, axis=0
    )  # Zxx: (freq, time, C)
    mag = np.abs(Zxx).astype(np.float32, copy=False)
    logmag = np.log1p(mag, dtype=np.float32)

    lm_mean = logmag.mean(axis=(0, 1), dtype=np.float32)
    lm_std = logmag.std(axis=(0, 1), dtype=np.float32)

    nb = 4
    freq_idx = np.arange(logmag.shape[0])
    bands = np.array_split(freq_idx, nb)

    for i in range(X.shape[1]):
        feats[f"sensor_{i+1}_stft_logmag_mean"] = float(lm_mean[i])
        feats[f"sensor_{i+1}_stft_logmag_std"] = float(lm_std[i])
        for b, idx in enumerate(bands):
            feats[f"sensor_{i+1}_stft_band{b}_mean"] = float(
                logmag[idx, :, i].mean(dtype=np.float32)
            )

    return feats


def _compute_one(sid_and_folder):
    sid, folder = sid_and_folder
    path = os.path.join(folder, f"{sid}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Segment file not found: {path}")
    feats = compute_segment_features(path)
    feats["segment_id"] = int(sid)
    return feats


def build_feature_table(segment_ids, folder: str) -> pd.DataFrame:
    items = [(sid, folder) for sid in segment_ids]
    max_workers = min(8, (os.cpu_count() or 2))
    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else None
    if ctx is None:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            rows = list(ex.map(_compute_one, items, chunksize=16))
    else:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            rows = list(ex.map(_compute_one, items, chunksize=16))
    return pd.DataFrame(rows)




## === cell 2
train_meta = pd.read_csv(TRAIN_META_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_meta["segment_id"] = train_meta["segment_id"].astype(np.int64)
sample_sub["segment_id"] = sample_sub["segment_id"].astype(np.int64)

time = train_meta["time_to_eruption"].astype(np.float32).reset_index(drop=True)

train_segment_ids = train_meta["segment_id"].tolist()
test_segment_ids = sample_sub["segment_id"].tolist()

train_feats = build_feature_table(train_segment_ids, TRAIN_DIR)
test_feats = build_feature_table(test_segment_ids, TEST_DIR)

train_feats = train_feats.sort_values("segment_id").reset_index(drop=True)
train_meta_sorted = train_meta.sort_values("segment_id").reset_index(drop=True)
time = train_meta_sorted["time_to_eruption"].astype(np.float32)

test_feats = test_feats.sort_values("segment_id").reset_index(drop=True)
sample_sub_sorted = sample_sub.sort_values("segment_id").reset_index(drop=True)

train_df_stft = train_feats.drop(["segment_id"], axis=1)
test_df_stft = test_feats.drop(["segment_id"], axis=1)

sd_cols = [c for c in train_df_stft.columns if ("_std" in c) or ("_stft_" in c)]
train_df_sd = train_df_stft[sd_cols]
test_df_sd = test_df_stft[sd_cols]

gc.collect()




## === cell 3
pca = PCA(n_components=0.999, svd_solver="full", random_state=RANDOM_SEED)
train_pca = pca.fit_transform(train_df_sd.to_numpy(copy=False))
test_pca = pca.transform(test_df_sd.to_numpy(copy=False))
train_pca = pd.DataFrame(train_pca, index=train_df_stft.index)
test_pca = pd.DataFrame(test_pca, index=test_df_stft.index)

train_df = pd.concat([train_df_stft, train_pca], axis=1)
test_df = pd.concat([test_df_stft, test_pca], axis=1)

assert len(train_df) == len(time), "Train features and target length mismatch"
assert len(test_df) == len(
    sample_sub_sorted
), "Test features and submission length mismatch"




## === cell 4
input_df = train_meta_sorted[["segment_id", "time_to_eruption"]].copy()
input_df = input_df.sort_values("time_to_eruption").reset_index(drop=True)

k = 5
folds = np.tile(np.arange(1, k + 1), int(np.ceil(len(input_df) / k)))[: len(input_df)]
rng = np.random.default_rng(RANDOM_SEED)
rng.shuffle(folds)
input_df["fold"] = folds

input_df["orig_row"] = input_df.index
segid_to_row = pd.Series(
    train_meta_sorted.index.values, index=train_meta_sorted["segment_id"]
).to_dict()
input_df["row_in_train_df"] = input_df["segment_id"].map(segid_to_row).astype(int)




## === cell 5
predictions = np.zeros(len(test_df), dtype=np.float32)

X_all = train_df.to_numpy(copy=False)
y_all = time.to_numpy(copy=False)
X_test = test_df.to_numpy(copy=False)


def _get_tree_method() -> str:
    try:
        X_small = np.asarray([[0.0], [1.0]], dtype=np.float32)
        y_small = np.asarray([0.0, 1.0], dtype=np.float32)
        m = xgboost.XGBRegressor(
            n_estimators=1,
            max_depth=1,
            learning_rate=1.0,
            tree_method="gpu_hist",
            random_state=RANDOM_SEED,
            verbosity=0,
        )
        m.fit(X_small, y_small, verbose=False)
        return "gpu_hist"
    except Exception:
        return "hist"


tree_method = _get_tree_method()
n_jobs = max(1, (os.cpu_count() or 2))

for fold in range(1, 6):
    train_rows = input_df.loc[input_df["fold"] != fold, "row_in_train_df"].to_numpy()
    val_rows = input_df.loc[input_df["fold"] == fold, "row_in_train_df"].to_numpy()

    X_train = X_all[train_rows]
    y_train = y_all[train_rows]
    X_val = X_all[val_rows]
    y_val = y_all[val_rows]

    model = xgboost.XGBRegressor(
        n_estimators=100000,
        tree_method=tree_method,
        max_depth=8,
        learning_rate=0.05,
        reg_alpha=0.1,
        subsample=0.6,
        colsample_bytree=0.6,
        random_state=RANDOM_SEED,
        n_jobs=n_jobs,
    )

    eval_set = [(X_val, y_val)]
    model.fit(
        X_train,
        y_train,
        early_stopping_rounds=10,
        eval_metric="mae",
        eval_set=eval_set,
        verbose=False,
    )

    try:
        print(
            f"Fold {fold} last 5 MAE:", model.evals_result()["validation_0"]["mae"][-5:]
        )
    except Exception:
        pass

    predictions += model.predict(X_test).astype(np.float32)

predictions /= 5.0




## === cell 6
sub = sample_sub_sorted.copy()
sub["time_to_eruption"] = predictions

sub = sub.set_index("segment_id").loc[sample_sub["segment_id"]].reset_index()

assert list(sub.columns) == ["segment_id", "time_to_eruption"]
assert len(sub) == len(sample_sub)

out_path = "xgb_5fldst_ft_sdpca_stft.csv"
sub.to_csv(out_path, index=False)
print("Wrote submission:", out_path, "shape:", sub.shape)
print(sub.head())
