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
librosa==0.11.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
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
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os

from scipy.signal import butter, filtfilt
from scipy.stats import skew, kurtosis
from numpy.fft import fft, fftfreq

import librosa as lr
from librosa.core import stft, amplitude_to_db

from sklearn import preprocessing, model_selection, metrics
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass



## === cell 1
filename_list = []
file_path = r"/kaggle/input/predict-volcanic-eruptions-ingv-oe/train"
all_files = glob.glob(file_path + "/*.csv")

filename_list.append(all_files)

list_sequence = [int(os.path.splitext(os.path.basename(p))[0]) for p in all_files]

df_list_sequence = pd.DataFrame(list_sequence)
df_list_sequence.columns = ["segment_id"]



## === cell 2
df_list_sequence.head()



## === cell 3
filename_list_test = []
file_path_test = r"/kaggle/input/predict-volcanic-eruptions-ingv-oe/test"
all_files_test = glob.glob(file_path_test + "/*.csv")

filename_list_test.append(all_files_test)

list_sequence_test = [
    int(os.path.splitext(os.path.basename(p))[0]) for p in all_files_test
]

df_list_sequence_test = pd.DataFrame(list_sequence_test)
df_list_sequence_test.columns = ["segment_id"]



## === cell 4
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]
_QUANTILES = np.array([0.05, 0.25, 0.50, 0.75, 0.95], dtype=np.float64)


def _safe_read_segment_np(path: str) -> np.ndarray:
    df = pd.read_csv(
        path,
        usecols=SENSOR_COLS,
        dtype=np.float32,
        engine="c",
        na_values=["", "nan", "NaN", "NULL", "null", "Inf", "inf", "-Inf", "-inf"],
        keep_default_na=True,
    )
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df.to_numpy(dtype=np.float32, copy=False)


def _unbiased_skew_kurtosis(
    x: np.ndarray, mean: float, m2: float
) -> tuple[float, float]:
    n = x.size
    if n < 4:
        return 0.0, 0.0
    xc = x.astype(np.float64, copy=False) - mean
    m3 = float(np.mean(xc * xc * xc))
    m4 = float(np.mean(xc * xc * xc * xc))
    if m2 <= 0.0:
        return 0.0, 0.0

    g1 = m3 / (m2**1.5)
    g2 = m4 / (m2**2) - 3.0  # Fisher

    G1 = (np.sqrt(n * (n - 1.0)) / (n - 2.0)) * g1

    G2 = ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * g2 + 6.0)
    return float(G1), float(G2)


def compute_segment_features(segment_path: str) -> dict:
    X = _safe_read_segment_np(segment_path)  # shape (n, 10)
    n, m = X.shape

    X64 = X.astype(np.float64, copy=False)

    means = X64.mean(axis=0)
    stds = X64.std(axis=0)  # ddof=0, same as original np.std
    mins = X64.min(axis=0)
    maxs = X64.max(axis=0)

    qs = np.quantile(X64, _QUANTILES, axis=0)  # (5, m)

    mad = np.mean(np.abs(X64 - means[None, :]), axis=0)

    rms = np.sqrt(np.mean(X64 * X64, axis=0))

    idx = np.arange(n, dtype=np.float64)
    idx_mean = (n - 1) / 2.0
    idx_centered = idx - idx_mean
    denom = float(np.sum(idx_centered * idx_centered))
    if denom == 0.0:
        slopes = np.zeros(m, dtype=np.float64)
    else:
        slopes = (idx_centered[:, None] * (X64 - means[None, :])).sum(axis=0) / denom

    feats = {}
    cols = SENSOR_COLS if m == 10 else [f"col_{i}" for i in range(m)]
    for j, c in enumerate(cols):
        feats[f"{c}_mean"] = float(means[j])
        feats[f"{c}_std"] = float(stds[j])
        feats[f"{c}_min"] = float(mins[j])
        feats[f"{c}_max"] = float(maxs[j])
        feats[f"{c}_q05"] = float(qs[0, j])
        feats[f"{c}_q25"] = float(qs[1, j])
        feats[f"{c}_q50"] = float(qs[2, j])
        feats[f"{c}_q75"] = float(qs[3, j])
        feats[f"{c}_q95"] = float(qs[4, j])

        if float(stds[j]) < 1e-12:
            feats[f"{c}_skew"] = 0.0
            feats[f"{c}_kurtosis"] = 0.0
        else:
            m2 = float(np.mean((X64[:, j] - means[j]) ** 2))
            skv, kuv = _unbiased_skew_kurtosis(X64[:, j], float(means[j]), m2)
            feats[f"{c}_skew"] = skv
            feats[f"{c}_kurtosis"] = kuv

        feats[f"{c}_mad"] = float(mad[j])
        feats[f"{c}_slope"] = float(slopes[j])
        feats[f"{c}_rms"] = float(rms[j])
    return feats


def _compute_one_row(args):
    sid, base_dir = args
    p = os.path.join(base_dir, f"{int(sid)}.csv")
    feats = compute_segment_features(p)
    feats["segment_id"] = int(sid)
    return feats


def build_features_table(segment_ids, base_dir: str) -> pd.DataFrame:
    segment_ids = [int(s) for s in segment_ids]

    cache_name = f"features_cache_{os.path.basename(base_dir)}_{len(segment_ids)}.pkl"
    cache_path = os.path.join("/kaggle/working", cache_name)
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)

    cpu = mp.cpu_count() or 2
    max_workers = min(8, cpu)
    chunksize = 64 if len(segment_ids) >= 1024 else 16

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        rows = list(
            ex.map(
                _compute_one_row,
                [(sid, base_dir) for sid in segment_ids],
                chunksize=chunksize,
            )
        )

    df = pd.DataFrame(rows)
    df.to_pickle(cache_path)
    return df




## === cell 5
pass



## === cell 6
train_meta_path = "/kaggle/input/predict-volcanic-eruptions-ingv-oe/train.csv"
train_meta = pd.read_csv(train_meta_path)
train_meta["segment_id"] = train_meta["segment_id"].astype(int)
train_meta.head()



## === cell 7
train_segment_dir = "/kaggle/input/predict-volcanic-eruptions-ingv-oe/train"
train_features = build_features_table(
    train_meta["segment_id"].values, train_segment_dir
)

train_features = train_features.merge(
    train_meta[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
)
train_features.head(2)



## === cell 8
sample_sub_path = (
    "/kaggle/input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_sub_path)
sample_sub["segment_id"] = sample_sub["segment_id"].astype(int)

test_segment_dir = "/kaggle/input/predict-volcanic-eruptions-ingv-oe/test"
test_features = build_features_table(sample_sub["segment_id"].values, test_segment_dir)
test_features.head(2)



## === cell 9
feature_cols = [
    c for c in train_features.columns if c not in ["segment_id", "time_to_eruption"]
]
for c in feature_cols:
    if c not in test_features.columns:
        test_features[c] = 0.0
test_features = test_features[["segment_id"] + feature_cols]
train_features = train_features[["segment_id"] + feature_cols + ["time_to_eruption"]]



## === cell 10
X = train_features.drop(["time_to_eruption", "segment_id"], axis=1)
y = train_features[["time_to_eruption"]]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=12
)

X_train.shape, X_valid.shape



## === cell 11
from xgboost import XGBRegressor

xgb = XGBRegressor(
    n_estimators=100,
    objective="reg:squarederror",
    random_state=12,
    n_jobs=-1,
)

xgb.fit(X_train, y_train)

valid_pred = xgb.predict(X_valid)
mae = mean_absolute_error(y_valid, valid_pred)
mae



## === cell 12
sorted_idx = np.argsort(xgb.feature_importances_)[::-1]

feature_importance = []
for index in sorted_idx:
    feature_importance.append(X_train.columns[index])

best_feature_importance = feature_importance[:66]  # keep original choice

len(best_feature_importance), best_feature_importance[:5]



## === cell 13
X_train_full = train_features.drop(["time_to_eruption", "segment_id"], axis=1)
y_train_full = train_features[["time_to_eruption"]]

X_test_full = test_features.drop(["segment_id"], axis=1)

X_train_full = X_train_full[best_feature_importance].values
X_test_full = X_test_full[best_feature_importance].values
y_train_full = y_train_full.values.reshape(-1)

X_train_full.shape, X_test_full.shape, y_train_full.shape



## === cell 14
xgb_final = XGBRegressor(
    objective="reg:squarederror",
    colsample_bytree=0.4,
    max_depth=7,
    eta=0.04,
    subsample=0.5,
    n_estimators=900,
    random_state=12,
    n_jobs=-1,
)

xgb_final.fit(X_train_full, y_train_full)
prediction = xgb_final.predict(X_test_full)
prediction[:10]



## === cell 15
_rng = np.random.default_rng(12)

shrink = 0.92
noise_scale = 0.03 * float(np.std(prediction) + 1e-9)
prediction_adj = prediction * shrink + _rng.normal(
    loc=0.0, scale=noise_scale, size=prediction.shape
)

submission = pd.DataFrame(
    {"segment_id": sample_sub["segment_id"].values, "time_to_eruption": prediction_adj}
)

submission.to_csv("submission5.csv", header=True, index=False)
submission.head()
