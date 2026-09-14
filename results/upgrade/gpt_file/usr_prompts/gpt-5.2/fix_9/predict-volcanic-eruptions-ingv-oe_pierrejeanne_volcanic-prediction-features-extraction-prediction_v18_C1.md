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

from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(0)



## === cell 1
TRAIN_DIR = "/kaggle/input/predict-volcanic-eruptions-ingv-oe/train"
TEST_DIR = "/kaggle/input/predict-volcanic-eruptions-ingv-oe/test"
TRAIN_META_PATH = "/kaggle/input/predict-volcanic-eruptions-ingv-oe/train.csv"
SAMPLE_SUB_PATH = (
    "/kaggle/input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv"
)

train_meta = pd.read_csv(TRAIN_META_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

train_meta.head()



## === cell 2
all_train_files = glob.glob(os.path.join(TRAIN_DIR, "*.csv"))
all_test_files = glob.glob(os.path.join(TEST_DIR, "*.csv"))

train_segment_ids = [
    int(os.path.splitext(os.path.basename(p))[0]) for p in all_train_files
]
test_segment_ids = [
    int(os.path.splitext(os.path.basename(p))[0]) for p in all_test_files
]

df_list_sequence = pd.DataFrame({"segment_id": train_segment_ids})
df_list_sequence_test = pd.DataFrame({"segment_id": test_segment_ids})

df_list_sequence.head(), df_list_sequence_test.head()



## === cell 3
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]
_PCTS = np.array([1, 5, 50, 95, 99], dtype=np.float32)

_STAT_NAMES = (
    "mean",
    "std",
    "min",
    "max",
    "q01",
    "q05",
    "q50",
    "q95",
    "q99",
    "skew",
    "kurt",
    "mad",
    "energy",
    "abs_mean",
    "abs_max",
)

_FEATURE_NAMES = []
for j in range(10):
    col = f"sensor_{j+1}"
    for k in _STAT_NAMES:
        _FEATURE_NAMES.append(f"{col}_{k}")
    for k in _STAT_NAMES:
        _FEATURE_NAMES.append(f"{col}_diff_{k}")
_FEATURE_NAMES = tuple(_FEATURE_NAMES)
_N_FEATURES = len(_FEATURE_NAMES)


def _skew_kurt_bias_false_from_centered(
    xc: np.ndarray, m2: np.ndarray, m3: np.ndarray, m4: np.ndarray, n: np.ndarray
):
    """
    Vectorized equivalent of:
      skew(x, bias=False) and kurtosis(x, fisher=True, bias=False)
    for each column, with NaNs ignored upstream.

    Inputs are central moments (with denominator n): m2=mean(xc^2), m3=mean(xc^3), m4=mean(xc^4)
    """
    skew_out = np.zeros_like(m2, dtype=np.float32)
    kurt_out = np.zeros_like(m2, dtype=np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        g1 = m3 / np.power(m2, 1.5)
        g2 = m4 / (m2 * m2) - 3.0

        mask1 = n > 2
        if np.any(mask1):
            nn = n[mask1].astype(np.float32)
            skew_out[mask1] = (np.sqrt(nn * (nn - 1.0)) / (nn - 2.0)) * g1[mask1]

        mask2 = n > 3
        if np.any(mask2):
            nn = n[mask2].astype(np.float32)
            kurt_out[mask2] = ((nn - 1.0) / ((nn - 2.0) * (nn - 3.0))) * (
                (nn + 1.0) * g2[mask2] + 6.0
            )

        skew_out = np.nan_to_num(skew_out, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )
        kurt_out = np.nan_to_num(kurt_out, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )

    return skew_out, kurt_out


def _safe_stats_from_mat(mat: np.ndarray) -> dict:
    """
    Compute stats for all columns at once.
    NaNs/infs ignored (treated as missing).
    """
    x = mat.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    n = finite.sum(axis=0).astype(np.int32)

    x0 = np.where(finite, x, 0.0).astype(np.float32, copy=False)
    n_safe = np.maximum(n, 1).astype(np.float32)

    mean = x0.sum(axis=0) / n_safe
    ex2 = (x0 * x0).sum(axis=0) / n_safe
    var = np.maximum(ex2 - mean * mean, 0.0)
    std = np.sqrt(var, dtype=np.float32)

    x_min = np.where(finite, x, np.inf).min(axis=0)
    x_max = np.where(finite, x, -np.inf).max(axis=0)
    x_min = np.where(np.isfinite(x_min), x_min, 0.0).astype(np.float32)
    x_max = np.where(np.isfinite(x_max), x_max, 0.0).astype(np.float32)

    x_nan = np.where(finite, x, np.nan).astype(np.float32, copy=False)
    try:
        q = np.nanpercentile(x_nan, _PCTS, axis=0, method="linear").astype(np.float32)
    except TypeError:
        q = np.nanpercentile(x_nan, _PCTS, axis=0, interpolation="linear").astype(
            np.float32
        )
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    xc = np.where(finite, x - mean[None, :], 0.0).astype(np.float32, copy=False)
    m2 = (xc * xc).sum(axis=0) / n_safe
    m3 = (xc * xc * xc).sum(axis=0) / n_safe
    m4 = (xc * xc * xc * xc).sum(axis=0) / n_safe
    skew_v, kurt_v = _skew_kurt_bias_false_from_centered(
        xc, m2, m3, m4, n.astype(np.int32)
    )

    mad = np.abs(xc).sum(axis=0) / n_safe
    energy = ex2  # mean(x^2)
    abs_mean = np.abs(np.where(finite, x, 0.0)).sum(axis=0) / n_safe
    abs_max = np.where(finite, np.abs(x), 0.0).max(axis=0).astype(np.float32)

    out = {
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "min": x_min,
        "max": x_max,
        "q01": q[0],
        "q05": q[1],
        "q50": q[2],
        "q95": q[3],
        "q99": q[4],
        "skew": skew_v.astype(np.float32),
        "kurt": kurt_v.astype(np.float32),
        "mad": mad.astype(np.float32),
        "energy": energy.astype(np.float32),
        "abs_mean": abs_mean.astype(np.float32),
        "abs_max": abs_max.astype(np.float32),
    }
    return out


_DTYPE_MAP = {c: np.float32 for c in SENSOR_COLS}


def _load_segment_matrix_fast(csv_path: str) -> np.ndarray:
    df = pd.read_csv(
        csv_path,
        usecols=SENSOR_COLS,
        dtype=_DTYPE_MAP,
        engine="c",
        low_memory=False,
        memory_map=True,
    )
    return df.to_numpy(dtype=np.float32, copy=False)


def extract_features_for_segment_vec(args):
    csv_path, segment_id, out_index = args
    mat = _load_segment_matrix_fast(csv_path)  # shape (n, 10)

    stats = _safe_stats_from_mat(mat)
    dmat = np.diff(mat, axis=0)
    dstats = _safe_stats_from_mat(dmat)

    out = np.empty(_N_FEATURES, dtype=np.float32)
    p = 0
    for j in range(10):
        for k in _STAT_NAMES:
            out[p] = stats[k][j]
            p += 1
        for k in _STAT_NAMES:
            out[p] = dstats[k][j]
            p += 1

    return out_index, int(segment_id), out


import multiprocessing as mp


def _worker_init():
    np.random.seed(0)


def build_feature_table(segment_ids, directory, max_rows=None):
    if max_rows is not None:
        segment_ids = segment_ids[:max_rows]

    segment_ids = list(segment_ids)
    n = len(segment_ids)
    paths = [os.path.join(directory, f"{sid}.csv") for sid in segment_ids]
    tasks = [(paths[i], segment_ids[i], i) for i in range(n)]

    cpu = os.cpu_count() or 2
    n_workers = min(8, cpu)
    chunksize = 256 if n >= 1024 else 64

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    feats_mat = np.empty((n, _N_FEATURES), dtype=np.float32)
    seg_arr = np.empty(n, dtype=np.int64)

    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        for out_index, sid, vec in pool.imap_unordered(
            extract_features_for_segment_vec, tasks, chunksize=chunksize
        ):
            seg_arr[out_index] = sid
            feats_mat[out_index, :] = vec

    df = pd.DataFrame(feats_mat, columns=_FEATURE_NAMES)
    df.insert(0, "segment_id", seg_arr.astype(np.int64, copy=False))
    return df




## === cell 4
train_features_only = build_feature_table(train_meta["segment_id"].tolist(), TRAIN_DIR)
train_features = pd.concat(
    [
        train_meta[["segment_id", "time_to_eruption"]].reset_index(drop=True),
        train_features_only.drop(columns=["segment_id"]).reset_index(drop=True),
    ],
    axis=1,
)

test_features_only = build_feature_table(sample_sub["segment_id"].tolist(), TEST_DIR)
test_features = pd.concat(
    [
        sample_sub[["segment_id"]].reset_index(drop=True),
        test_features_only.drop(columns=["segment_id"]).reset_index(drop=True),
    ],
    axis=1,
)

train_features.head(2), test_features.head(2)



## === cell 5
X_train = train_features.drop(["time_to_eruption", "segment_id"], axis=1)
y_train = train_features[["time_to_eruption"]]
X_test = test_features.drop(["segment_id"], axis=1)

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)
y_train = y_train.replace([np.inf, -np.inf], np.nan).fillna(y_train.median())

X_train.shape, X_test.shape, y_train.shape



## === cell 6
scalerx = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
X_train_scaled = scalerx.fit_transform(X_train)
X_test_scaled = scalerx.transform(X_test)

scalery = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
y_train_scaled = scalery.fit_transform(y_train)

X_cols = X_train.columns.to_list()



## === cell 7
y = y_train_scaled.reshape(-1).astype(np.float64, copy=False)
X = X_train_scaled.astype(np.float64, copy=False)

y0 = y - y.mean()
X0 = X - X.mean(axis=0)
den = np.sqrt((X0 * X0).sum(axis=0)) * np.sqrt((y0 * y0).sum())
corr = (X0.T @ y0) / den
corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)

df_cor = pd.DataFrame(
    {"time_to_eruption": corr.astype(np.float32)}, index=X_cols
).sort_values("time_to_eruption", ascending=False)

selected = df_cor[df_cor["time_to_eruption"] < -0.24]
if selected.shape[0] == 0:
    selected = df_cor.reindex(
        df_cor["time_to_eruption"].abs().sort_values(ascending=False).head(100).index
    )

X_names = selected.index.tolist()
len(X_names), X_names[:10]



## === cell 8
name_to_idx = {n: i for i, n in enumerate(X_cols)}
sel_idx = np.fromiter((name_to_idx[n] for n in X_names), dtype=np.int64)

X_train_sel = X_train_scaled[:, sel_idx].astype(np.float64, copy=False)
X_test_sel = X_test_scaled[:, sel_idx].astype(np.float64, copy=False)
y_train_sel = y_train_scaled.reshape(-1).astype(np.float64, copy=False)

X_train_sel.shape, X_test_sel.shape, y_train_sel.shape



## === cell 9
DO_PLOTS = False
if DO_PLOTS:
    df_X_train = pd.DataFrame(X_train_sel, columns=X_names)
    df_y_train = pd.DataFrame(y_train_sel, columns=["time_to_eruption"])
    Xy_train_plot = pd.concat([df_X_train, df_y_train], axis=1, sort=False)
    Xy_train_plot.plot(
        x="time_to_eruption",
        y=X_names[0:],
        kind="line",
        legend=False,
        subplots=True,
        sharex=True,
        figsize=(20, 30),
        ls="none",
        marker="o",
        layout=(10, 9),
    )
    plt.show()



## === cell 10
from sklearn.svm import SVR

svr_rbf = SVR(kernel="rbf", C=3.0, gamma=0.03, degree=2, epsilon=0.2, coef0=0)
svr_rbf.fit(X_train_sel, y_train_sel)

pred_scaled = svr_rbf.predict(X_test_sel).reshape(-1, 1)
predicted = scalery.inverse_transform(pred_scaled).ravel()

predicted[:5], predicted.shape



## === cell 11
submission = sample_sub.copy()
submission["time_to_eruption"] = predicted

submission["segment_id"] = submission["segment_id"].astype(int)
submission["time_to_eruption"] = submission["time_to_eruption"].astype(float)

submission.head()



## === cell 12
SUB_PATH = "submission.csv"
submission.to_csv(SUB_PATH, index=False)
print(
    f"Wrote {SUB_PATH} with shape {submission.shape} and columns {submission.columns.tolist()}"
)



## === cell 13
pass
