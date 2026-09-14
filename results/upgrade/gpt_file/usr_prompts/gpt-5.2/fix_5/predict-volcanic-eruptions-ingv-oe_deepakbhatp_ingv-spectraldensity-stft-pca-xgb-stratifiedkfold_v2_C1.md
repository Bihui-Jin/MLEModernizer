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

5381042.07761754

# 6. Current score

4594411.0

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 4594411.0) has done: 'I fix the NaN issue that crashes PCA by imputing missing feature values (created by all-NaN sensor columns/segments) using a median strategy computed on the training set, which is score-stable and keeps the same model/feature logic. Then I ensure `train_df`/`test_df` are always created so downstream training doesn’t fail with `NameError`. I also make XGBoost early stopping compatible with xgboost==2.0.3 by switching to the callback-based early stopping API (same semantics: MAE on the validation fold), so training runs end-to-end. Finally, I keep the submission format aligned to `sample_submission.csv` and write a `.csv` file.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
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

_READCSV_KW = dict(
    usecols=SENSORS,
    dtype={c: np.float32 for c in SENSORS},
    engine="c",
)


def _quantile_linear_via_partition(x_sortedless: np.ndarray, p: float) -> float:
    n = x_sortedless.size
    if n == 0:
        return np.nan
    if n == 1:
        return float(x_sortedless[0])

    pos = (n - 1) * p
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        kth = np.partition(x_sortedless, lo)[lo]
        return float(kth)
    part = np.partition(x_sortedless, (lo, hi))
    xlo = part[lo]
    xhi = part[hi]
    w = pos - lo
    return float(xlo * (1.0 - w) + xhi * w)


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
        xf = x[finite].astype(np.float64, copy=False)
        m = float(xf.mean())
        s = float(xf.std())
        mn = float(xf.min())
        mx = float(xf.max())

        q05 = _quantile_linear_via_partition(xf, 0.05)
        q25 = _quantile_linear_via_partition(xf, 0.25)
        q50 = _quantile_linear_via_partition(xf, 0.50)
        q75 = _quantile_linear_via_partition(xf, 0.75)
        q95 = _quantile_linear_via_partition(xf, 0.95)
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


def featurize_segment(seg_path: str) -> np.ndarray:
    df = pd.read_csv(seg_path, **_READCSV_KW)
    data = df.to_numpy(dtype=np.float32, copy=False)  # (rows, 10)
    out = np.empty((len(SENSORS), 12), dtype=np.float64)
    for j in range(len(SENSORS)):
        arr = data[:, j]
        out[j, :] = _safe_stats_and_diffs(arr)
    return out.reshape(-1)


def build_feature_df(segment_ids, folder: str, n_jobs: int = None) -> pd.DataFrame:
    segment_ids = list(segment_ids)
    paths = [os.path.join(folder, f"{sid}.csv") for sid in segment_ids]
    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing segment file: {p}")

    keys = [
        "mean",
        "std",
        "min",
        "max",
        "q05",
        "q25",
        "q50",
        "q75",
        "q95",
        "diff_mean",
        "diff_std",
        "abs_diff_mean",
    ]
    feat_names = []
    for col in SENSORS:
        for k in keys:
            feat_names.append(f"{col}_{k}")

    if n_jobs is None:
        cpu = os.cpu_count() or 2
        n_jobs = max(1, min(8, cpu))

    if n_jobs == 1:
        feats_mat = np.vstack([featurize_segment(p) for p in paths])
    else:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            feats_list = list(ex.map(featurize_segment, paths, chunksize=32))
        feats_mat = np.vstack(feats_list)

    out = pd.DataFrame(feats_mat, columns=feat_names)
    out.insert(0, "segment_id", np.asarray(segment_ids, dtype=np.int64))
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
    build_feature_df(train_ids, TRAIN_DIR, n_jobs=None)
    .sort_values("segment_id")
    .reset_index(drop=True)
)
test_feat = (
    build_feature_df(test_ids, TEST_DIR, n_jobs=None)
    .sort_values("segment_id")
    .reset_index(drop=True)
)

train_X = train_feat.drop(columns=["segment_id"])
test_X = test_feat.drop(columns=["segment_id"])

feat_cols = train_X.columns.tolist()
test_X = test_X[feat_cols]

print("Train features shape:", train_X.shape, "Test features shape:", test_X.shape)



## === cell 3
imputer = SimpleImputer(strategy="median")

train_X_np = np.ascontiguousarray(train_X.to_numpy(dtype=np.float32, copy=False))
test_X_np = np.ascontiguousarray(test_X.to_numpy(dtype=np.float32, copy=False))

train_X_imp = imputer.fit_transform(train_X_np).astype(np.float32, copy=False)
test_X_imp = imputer.transform(test_X_np).astype(np.float32, copy=False)

pca = PCA(n_components=0.999, svd_solver="full", random_state=RANDOM_SEED)

train_pca = pca.fit_transform(train_X_imp)
test_pca = pca.transform(test_X_imp)

train_pca = pd.DataFrame(train_pca, index=train_X.index)
test_pca = pd.DataFrame(test_pca, index=test_X.index)

train_df = pd.concat(
    [train_X.reset_index(drop=True), train_pca.reset_index(drop=True)], axis=1
)
test_df = pd.concat(
    [test_X.reset_index(drop=True), test_pca.reset_index(drop=True)], axis=1
)

print("Final train_df shape:", train_df.shape, "Final test_df shape:", test_df.shape)
print(
    "Any NaN in train_df:",
    np.isnan(train_df.to_numpy(dtype=np.float32, copy=False)).any(),
)



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
X_all = np.ascontiguousarray(train_df.to_numpy(dtype=np.float32, copy=False))
X_test = np.ascontiguousarray(test_df.to_numpy(dtype=np.float32, copy=False))

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
        eval_set=eval_set,
        eval_metric="mae",
        callbacks=[xgboost.callback.EarlyStopping(rounds=5, save_best=True)],
        verbose=False,
    )

    mae_hist = model.evals_result()["validation_0"]["mae"]
    best_it = getattr(model, "best_iteration", None)
    print(f"Fold {fold} best_iteration={best_it} last5_mae={mae_hist[-5:]}")

    predictions += model.predict(X_test)

predictions /= 5.0

submission = sample_sub.copy()
submission = submission.sort_values("segment_id").reset_index(drop=True)
submission["time_to_eruption"] = predictions.astype(np.float64)

out_path = "xgb_5fldst_ft_sdpca_stft.csv"
submission.to_csv(out_path, index=False)

print("Wrote submission:", out_path, "shape:", submission.shape)
print(submission.head())
