# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
import xgboost

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/predict-volcanic-eruptions-ingv-oe",
    "/kaggle/data/predict-volcanic-eruptions-ingv-oe",
    "/kaggle/data/input/predict-volcanic-eruptions-ingv-oe",
    "../input/predict-volcanic-eruptions-ingv-oe",
]


def find_data_root():
    for p in DATA_ROOT_CANDIDATES:
        if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
            os.path.join(p, "train")
        ):
            return p
    for base in ["/kaggle/input", "/kaggle/data", "/kaggle/data/input", "../input"]:
        cand = os.path.join(base, "predict-volcanic-eruptions-ingv-oe")
        if os.path.exists(os.path.join(cand, "train.csv")) and os.path.exists(
            os.path.join(cand, "train")
        ):
            return cand
    raise FileNotFoundError(
        "Could not locate competition data root containing train.csv"
    )


DATA_ROOT = find_data_root()
TRAIN_META_PATH = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

print("Using DATA_ROOT:", DATA_ROOT)
print(
    "Train dir exists:",
    os.path.isdir(TRAIN_DIR),
    "Test dir exists:",
    os.path.isdir(TEST_DIR),
)



## === cell 1
SENSORS = [f"sensor_{i}" for i in range(1, 11)]
SENSOR_TO_IDX = {c: i for i, c in enumerate(SENSORS)}


def _interp_limit_both_fill0(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32, copy=False)
    n = x.size
    if n == 0:
        return x
    mask = np.isfinite(x)
    if not mask.any():
        return np.zeros_like(x, dtype=np.float32)
    idx = np.arange(n, dtype=np.int64)
    out = np.interp(
        idx.astype(np.float64), idx[mask].astype(np.float64), x[mask].astype(np.float64)
    ).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def _safe_stats_and_diffs(x: np.ndarray):
    x = x.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    if finite.any():
        xf = x[finite].astype(
            np.float64, copy=False
        )  # use float64 for stable reductions like numpy does internally
        m = float(xf.mean())
        s = float(xf.std())
        mn = float(xf.min())
        mx = float(xf.max())

        xs = np.sort(
            xf
        )  # exact quantiles via same "linear" interpolation as default nanquantile
        n = xs.size

        def q(p):
            if n == 1:
                return float(xs[0])
            pos = (n - 1) * p
            lo = int(np.floor(pos))
            hi = int(np.ceil(pos))
            if lo == hi:
                return float(xs[lo])
            w = pos - lo
            return float(xs[lo] * (1.0 - w) + xs[hi] * w)

        q05 = q(0.05)
        q25 = q(0.25)
        q50 = q(0.50)
        q75 = q(0.75)
        q95 = q(0.95)
    else:
        m = s = mn = mx = q05 = q25 = q50 = q75 = q95 = np.nan

    s_filled = _interp_limit_both_fill0(x)
    dif = np.diff(s_filled)
    if dif.size:
        dif64 = dif.astype(np.float64, copy=False)
        d_mean = float(dif64.mean())
        d_std = float(dif64.std())
        abs_d_mean = float(np.abs(dif64).mean())
    else:
        d_mean = d_std = abs_d_mean = 0.0

    return (m, s, mn, mx, q05, q25, q50, q75, q95, d_mean, d_std, abs_d_mean)


_READCSV_KW = dict(
    usecols=SENSORS,
    dtype={c: np.float32 for c in SENSORS},
    engine="c",
)


def featurize_segment(seg_path: str) -> dict:
    df = pd.read_csv(seg_path, **_READCSV_KW)
    data = df.to_numpy(dtype=np.float32, copy=False)  # shape (rows, 10)
    feats = {}
    keys = ["mean", "std", "min", "max", "q05", "q25", "q50", "q75", "q95"]
    for j, col in enumerate(SENSORS):
        arr = data[:, j]
        (m, s, mn, mx, q05, q25, q50, q75, q95, d_mean, d_std, abs_d_mean) = (
            _safe_stats_and_diffs(arr)
        )
        for k, v in zip(keys, [m, s, mn, mx, q05, q25, q50, q75, q95]):
            feats[f"{col}_{k}"] = v
        feats[f"{col}_diff_mean"] = d_mean
        feats[f"{col}_diff_std"] = d_std
        feats[f"{col}_abs_diff_mean"] = abs_d_mean
    return feats


def build_feature_df(segment_ids, folder: str) -> pd.DataFrame:
    rows = []
    for sid in segment_ids:
        seg_path = os.path.join(folder, f"{sid}.csv")
        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"Missing segment file: {seg_path}")
        feats = featurize_segment(seg_path)
        feats["segment_id"] = int(sid)
        rows.append(feats)
    out = pd.DataFrame(rows)
    return out




## === cell 2
train_meta = pd.read_csv(TRAIN_META_PATH)
train_meta["segment_id"] = train_meta["segment_id"].astype(np.int64)
train_meta = train_meta.sort_values("segment_id").reset_index(drop=True)

time = train_meta["time_to_eruption"].astype(np.float32).copy()

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["segment_id"] = sample_sub["segment_id"].astype(np.int64)
sample_sub = sample_sub.sort_values("segment_id").reset_index(drop=True)

train_ids = train_meta["segment_id"].tolist()
test_ids = sample_sub["segment_id"].tolist()

train_feat = (
    build_feature_df(train_ids, TRAIN_DIR)
    .sort_values("segment_id")
    .reset_index(drop=True)
)
test_feat = (
    build_feature_df(test_ids, TEST_DIR)
    .sort_values("segment_id")
    .reset_index(drop=True)
)

train_X = train_feat.drop(columns=["segment_id"])
test_X = test_feat.drop(columns=["segment_id"])

feat_cols = train_X.columns.tolist()
test_X = test_X[feat_cols]

print("Train features shape:", train_X.shape, "Test features shape:", test_X.shape)



## === cell 3
pca = PCA(n_components=0.999, svd_solver="full", random_state=RANDOM_SEED)
train_pca = pca.fit_transform(train_X.to_numpy(dtype=np.float64, copy=False))
test_pca = pca.transform(test_X.to_numpy(dtype=np.float64, copy=False))

train_pca = pd.DataFrame(train_pca, index=train_X.index)
test_pca = pd.DataFrame(test_pca, index=test_X.index)

train_df = pd.concat([train_X, train_pca], axis=1)
test_df = pd.concat([test_X, test_pca], axis=1)

print("Final train_df shape:", train_df.shape, "Final test_df shape:", test_df.shape)



## === cell 4
input_df = train_meta[["segment_id", "time_to_eruption"]].copy()
input_df = input_df.sort_values("time_to_eruption").reset_index(drop=True)

fold_list = [1, 2, 3, 4, 5]
folds = []
n = input_df.shape[0]
for _ in range(n // 5):
    random.shuffle(fold_list)
    folds.extend(fold_list)
remainder = n - len(folds)
if remainder > 0:
    random.shuffle(fold_list)
    folds.extend(fold_list[:remainder])

assert len(folds) == n
input_df["fold"] = folds

seg_to_row = {sid: i for i, sid in enumerate(train_meta["segment_id"].tolist())}
input_df["row_idx"] = input_df["segment_id"].map(seg_to_row)
input_df = input_df.sort_values("row_idx").reset_index(drop=True)

print("Fold distribution:\n", input_df["fold"].value_counts().sort_index())



## === cell 5
X_all = train_df.to_numpy(dtype=np.float32, copy=False)
X_test = test_df.to_numpy(dtype=np.float32, copy=False)

predictions = np.zeros(len(test_df), dtype=np.float64)

for fold in range(1, 6):
    train_index_list = input_df[input_df["fold"] != fold]["row_idx"].to_numpy()
    val_index_list = input_df[input_df["fold"] == fold]["row_idx"].to_numpy()

    X_train = X_all[train_index_list]
    y_train = time.iloc[train_index_list]
    X_val = X_all[val_index_list]
    y_val = time.iloc[val_index_list]

    model = xgboost.XGBRegressor(
        n_estimators=100000,
        tree_method="hist",
        max_depth=6,
        learning_rate=0.05,
        reg_alpha=0.1,
        subsample=0.6,
        objective="reg:absoluteerror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    eval_set = [(X_val, y_val)]
    model.fit(
        X_train,
        y_train,
        early_stopping_rounds=5,
        eval_metric="mae",
        eval_set=eval_set,
        verbose=False,
    )

    mae_hist = model.evals_result()["validation_0"]["mae"]
    print(
        f"Fold {fold} best_iteration={model.best_iteration} last5_mae={mae_hist[-5:]}"
    )

    predictions += model.predict(X_test)

predictions /= 5.0

submission = sample_sub.copy()
submission = submission.sort_values("segment_id").reset_index(drop=True)
submission["time_to_eruption"] = predictions.astype(np.float64)

out_path = "xgb_5fldst_ft_sdpca_stft.csv"
submission.to_csv(out_path, index=False)

print("Wrote submission:", out_path)
print(submission.head())
