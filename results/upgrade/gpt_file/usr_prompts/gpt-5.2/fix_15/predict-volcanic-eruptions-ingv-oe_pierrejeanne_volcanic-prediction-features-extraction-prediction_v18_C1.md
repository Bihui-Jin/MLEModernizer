# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

11742913.23097345

# 6. Current score

24402200.0

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 5574561.0) has done: 'I remove the dependency on the missing `../input/features-from-version-16/*.csv` files by computing the same kind of tabular features directly from the provided per-segment sensor CSVs in `/kaggle/input/predict-volcanic-eruptions-ingv-oe/train` and `/kaggle/input/predict-volcanic-eruptions-ingv-oe/test`. I keep the core modeling approach (SVR with RBF kernel) and keep the same general scaling workflow, but fix a critical bug where the test data scaler was incorrectly refit on test instead of using the train-fitted scaler (this is score-improving and still semantically consistent). I also ensure `segment_id` ordering matches `sample_submission.csv` so predictions align with the required output rows, and I write a valid `submission.csv`. Finally, I remove/disable the unfinished XGBoost randomized search section that currently cannot run due to undefined variables and missing imports.'
- What this solution (achieved 24402200.0) has done: 'The timeout is dominated by per-file pandas CSV parsing and heavy per-sensor statistics (especially `skew`, `kurtosis`, and repeated `quantile`) across ~4k train + 444 test segments. To keep identical features and model logic while cutting runtime, I (1) switch feature extraction to fast, deterministic NumPy loading (`np.loadtxt`) with precomputed column indices, (2) compute quantiles once per sensor using `np.nanpercentile` and reuse shared masks to avoid repeated `isfinite` passes, and (3) parallelize per-segment feature extraction with a process pool while preserving determinism (fixed ordering of segment_ids). Everything downstream (scaling, correlation-based selection, SVR fit/predict, submission) remains unchanged in semantics.'
- What this solution (achieved 24402200.0) has done: 'The timeout is dominated by per-segment CSV parsing plus expensive per-column sorting for percentiles (done twice per segment: raw and diff), multiplied by ~4k train + 444 test files. To keep identical feature semantics, I (1) switch percentile computation to NumPy’s `np.nanpercentile(..., method="linear")` which matches pandas’ default linear interpolation and avoids full-column sorts, and (2) speed up I/O by using `pyarrow.csv` if available, otherwise `pandas.read_csv` with `float32` and memory mapping, while also avoiding repeated DataFrame concat/index resets. I also reduce multiprocessing overhead by using `imap` with a larger chunksize and avoiding unnecessary data copies, while keeping the same worker paradigm and determinism. Model/scaling/training remain unchanged.'
- What this solution (achieved 24402200.0) has done: 'The timeout is dominated by per-segment CSV parsing and feature extraction over ~4000 train + 444 test files; the model fit is relatively small after feature selection. I keep the exact same features and SVR training, but make the extraction faster by (1) removing heavy unused imports, (2) using a much faster CSV reader path when available (pyarrow), (3) avoiding extra array copies inside stats computation, and (4) cutting multiprocessing overhead by using a larger chunksize and fewer worker processes tuned for I/O-bound workloads. I also make cached feature loading avoid an unnecessary full copy by building DataFrames directly from memmaps. All changes preserve identical feature definitions and downstream training/prediction semantics (only negligible floating-point order-of-ops differences).'

# 9. Code solution

## === cell 0
import glob
import os
import numpy as np
import pandas as pd

from sklearn import preprocessing

np.random.seed(0)
os.environ.setdefault("PYTHONHASHSEED", "0")




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
_PCTS = np.array([1, 5, 50, 95, 99], dtype=np.float64)

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
    m2: np.ndarray, m3: np.ndarray, m4: np.ndarray, n: np.ndarray
):
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
            np.float32, copy=False
        )
        kurt_out = np.nan_to_num(kurt_out, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        )

    return skew_out, kurt_out


def _nanpercentiles_linear(mat: np.ndarray, pcts=_PCTS) -> np.ndarray:
    q = np.nanpercentile(mat, pcts, axis=0, method="linear")
    return q.astype(np.float32, copy=False)


def _safe_stats_from_mat(mat: np.ndarray) -> dict:
    """
    Compute stats for all columns at once.
    NaNs/infs ignored (treated as missing).
    """
    x = np.asarray(mat, dtype=np.float32, order="C")
    finite = np.isfinite(x)
    n = finite.sum(axis=0).astype(np.int32)
    n_safe = np.maximum(n, 1).astype(np.float32)

    x0 = x.copy()
    x0[~finite] = 0.0

    sum1 = x0.sum(axis=0)
    mean = sum1 / n_safe

    sum2 = (x0 * x0).sum(axis=0)
    ex2 = sum2 / n_safe
    var = np.maximum(ex2 - mean * mean, 0.0)
    std = np.sqrt(var, dtype=np.float32)

    x_min = np.where(finite, x, np.inf).min(axis=0)
    x_max = np.where(finite, x, -np.inf).max(axis=0)
    x_min = np.where(np.isfinite(x_min), x_min, 0.0).astype(np.float32, copy=False)
    x_max = np.where(np.isfinite(x_max), x_max, 0.0).astype(np.float32, copy=False)

    x_nan = x.copy()
    x_nan[~finite] = np.nan
    q = _nanpercentiles_linear(x_nan, pcts=_PCTS)

    xc = x0 - mean[None, :]
    xc[~finite] = 0.0
    xc2 = xc * xc
    m2 = xc2.sum(axis=0) / n_safe
    m3 = (xc2 * xc).sum(axis=0) / n_safe
    m4 = (xc2 * xc2).sum(axis=0) / n_safe
    skew_v, kurt_v = _skew_kurt_bias_false_from_centered(m2, m3, m4, n)

    mad = np.abs(xc).sum(axis=0) / n_safe
    energy = ex2  # mean(x^2)

    abs_x0 = np.abs(x0)
    abs_mean = abs_x0.sum(axis=0) / n_safe

    abs_max = abs_x0.max(axis=0).astype(np.float32, copy=False)

    out = {
        "mean": mean.astype(np.float32, copy=False),
        "std": std.astype(np.float32, copy=False),
        "min": x_min,
        "max": x_max,
        "q01": np.nan_to_num(q[0], nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        ),
        "q05": np.nan_to_num(q[1], nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        ),
        "q50": np.nan_to_num(q[2], nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        ),
        "q95": np.nan_to_num(q[3], nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        ),
        "q99": np.nan_to_num(q[4], nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        ),
        "skew": skew_v.astype(np.float32, copy=False),
        "kurt": kurt_v.astype(np.float32, copy=False),
        "mad": mad.astype(np.float32, copy=False),
        "energy": energy.astype(np.float32, copy=False),
        "abs_mean": abs_mean.astype(np.float32, copy=False),
        "abs_max": abs_max,
    }
    return out


_DTYPE_MAP = {c: np.float32 for c in SENSOR_COLS}

try:
    import pyarrow.csv as pacsv  # type: ignore
    import pyarrow as pa  # type: ignore

    _HAVE_PYARROW = True
except Exception:
    _HAVE_PYARROW = False


def _load_segment_matrix_fast(csv_path: str) -> np.ndarray:
    if _HAVE_PYARROW:
        tbl = pacsv.read_csv(
            csv_path,
            read_options=pacsv.ReadOptions(use_threads=True),
            parse_options=pacsv.ParseOptions(delimiter=","),
            convert_options=pacsv.ConvertOptions(
                include_columns=SENSOR_COLS,
                column_types={c: pa.float32() for c in SENSOR_COLS},
            ),
        )
        arr = tbl.to_numpy(zero_copy_only=False)
        return np.asarray(arr, dtype=np.float32, order="C")

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
    mat = _load_segment_matrix_fast(csv_path)  # (n, 10)

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
    os.environ.setdefault("PYTHONHASHSEED", "0")


_CACHE_DIR = "/kaggle/working/feature_cache_volcano"
os.makedirs(_CACHE_DIR, exist_ok=True)


def _cache_paths(prefix: str):
    return (
        os.path.join(_CACHE_DIR, f"{prefix}_segment_id.npy"),
        os.path.join(_CACHE_DIR, f"{prefix}_features.npy"),
    )


def build_feature_table(segment_ids, directory, max_rows=None, cache_prefix=None):
    if max_rows is not None:
        segment_ids = segment_ids[:max_rows]

    segment_ids = list(segment_ids)
    n = len(segment_ids)

    if cache_prefix is not None:
        sid_path, feat_path = _cache_paths(cache_prefix)
        if os.path.exists(sid_path) and os.path.exists(feat_path):
            seg_arr = np.load(sid_path, mmap_mode="r")
            feats_mat = np.load(feat_path, mmap_mode="r")
            df = pd.DataFrame(feats_mat, columns=_FEATURE_NAMES)
            df.insert(0, "segment_id", seg_arr.astype(np.int64, copy=False))
            return df

    paths = [os.path.join(directory, f"{sid}.csv") for sid in segment_ids]
    tasks = [(paths[i], segment_ids[i], i) for i in range(n)]

    cpu = os.cpu_count() or 2
    n_workers = min(8, cpu)

    if n >= 4096:
        chunksize = 1024
    elif n >= 2048:
        chunksize = 512
    elif n >= 1024:
        chunksize = 256
    else:
        chunksize = 128

    try:
        ctx = mp.get_context("fork")
    except ValueError:
        ctx = mp.get_context("spawn")

    feats_mat = np.empty((n, _N_FEATURES), dtype=np.float32)
    seg_arr = np.empty(n, dtype=np.int64)

    with ctx.Pool(
        processes=n_workers, initializer=_worker_init, maxtasksperchild=2000
    ) as pool:
        for out_index, sid, vec in pool.imap_unordered(
            extract_features_for_segment_vec, tasks, chunksize=chunksize
        ):
            seg_arr[out_index] = sid
            feats_mat[out_index, :] = vec

    if cache_prefix is not None:
        sid_path, feat_path = _cache_paths(cache_prefix)
        np.save(sid_path, seg_arr, allow_pickle=False)
        np.save(feat_path, feats_mat, allow_pickle=False)

    df = pd.DataFrame(feats_mat, columns=_FEATURE_NAMES)
    df.insert(0, "segment_id", seg_arr.astype(np.int64, copy=False))
    return df




## === cell 4
train_features_only = build_feature_table(
    train_meta["segment_id"].tolist(), TRAIN_DIR, cache_prefix="train"
)
train_features = train_meta[["segment_id", "time_to_eruption"]].merge(
    train_features_only, on="segment_id", how="left", sort=False, copy=False
)

test_features_only = build_feature_table(
    sample_sub["segment_id"].tolist(), TEST_DIR, cache_prefix="test"
)
test_features = sample_sub[["segment_id"]].merge(
    test_features_only, on="segment_id", how="left", sort=False, copy=False
)

train_features.head(2), test_features.head(2)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RemoteTraceback                           Traceback (most recent call last)
RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
                    ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 48, in mapstar
    return list(map(*args))
           ^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/1738086668.py", line 192, in extract_features_for_segment_vec
    mat = _load_segment_matrix_fast(csv_path)  # (n, 10)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/1738086668.py", line 176, in _load_segment_matrix_fast
    arr = tbl.to_numpy(zero_copy_only=False)
          ^^^^^^^^^^^^
AttributeError: 'pyarrow.lib.Table' object has no attribute 'to_numpy'
"""

The above exception was the direct cause of the following exception:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3048744130.py in <cell line: 0>()
----> 1 train_features_only = build_feature_table(
      2     train_meta["segment_id"].tolist(), TRAIN_DIR, cache_prefix="train"
      3 )
      4 train_features = train_meta[["segment_id", "time_to_eruption"]].merge(
      5     train_features_only, on="segment_id", how="left", sort=False, copy=False

/tmp/ipykernel_11/1738086668.py in build_feature_table(segment_ids, directory, max_rows, cache_prefix)
    273         processes=n_workers, initializer=_worker_init, maxtasksperchild=2000
    274     ) as pool:
--> 275         for out_index, sid, vec in pool.imap_unordered(
    276             extract_features_for_segment_vec, tasks, chunksize=chunksize
    277         ):

/usr/lib/python3.11/multiprocessing/pool.py in <genexpr>(.0)
    449                     result._set_length
    450                 ))
--> 451             return (item for chunk in result for item in chunk)
    452 
    453     def apply_async(self, func, args=(), kwds={}, callback=None,

/usr/lib/python3.11/multiprocessing/pool.py in next(self, timeout)
    871         if success:
    872             return value
--> 873         raise value
    874 
    875     __next__ = next                    # XXX

AttributeError: 'pyarrow.lib.Table' object has no attribute 'to_numpy'

## === cell 5
X_train = train_features.drop(["time_to_eruption", "segment_id"], axis=1)
y_train = train_features[["time_to_eruption"]]
X_test = test_features.drop(["segment_id"], axis=1)

X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)
y_train = y_train.replace([np.inf, -np.inf], np.nan).fillna(y_train.median())

X_train.shape, X_test.shape, y_train.shape




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3740348229.py in <cell line: 0>()
      1 # Speed: keep cleaning vectorized; unchanged semantics.
----> 2 X_train = train_features.drop(["time_to_eruption", "segment_id"], axis=1)
      3 y_train = train_features[["time_to_eruption"]]
      4 X_test = test_features.drop(["segment_id"], axis=1)
      5 

NameError: name 'train_features' is not defined

## === cell 6
scalerx = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
X_train_scaled = scalerx.fit_transform(X_train)
X_test_scaled = scalerx.transform(X_test)

scalery = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
y_train_scaled = scalery.fit_transform(y_train)

X_cols = X_train.columns.to_list()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2687777993.py in <cell line: 0>()
      1 scalerx = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
----> 2 X_train_scaled = scalerx.fit_transform(X_train)
      3 X_test_scaled = scalerx.transform(X_test)
      4 
      5 scalery = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))

NameError: name 'X_train' is not defined

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




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3608423869.py in <cell line: 0>()
----> 1 y = y_train_scaled.reshape(-1).astype(np.float64, copy=False)
      2 X = X_train_scaled.astype(np.float64, copy=False)
      3 
      4 y0 = y - y.mean()
      5 X0 = X - X.mean(axis=0)

NameError: name 'y_train_scaled' is not defined

## === cell 8
name_to_idx = {n: i for i, n in enumerate(X_cols)}
sel_idx = np.fromiter((name_to_idx[n] for n in X_names), dtype=np.int64)

X_train_sel = X_train_scaled[:, sel_idx].astype(np.float64, copy=False)
X_test_sel = X_test_scaled[:, sel_idx].astype(np.float64, copy=False)
y_train_sel = y_train_scaled.reshape(-1).astype(np.float64, copy=False)

X_train_sel.shape, X_test_sel.shape, y_train_sel.shape




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3404857591.py in <cell line: 0>()
----> 1 name_to_idx = {n: i for i, n in enumerate(X_cols)}
      2 sel_idx = np.fromiter((name_to_idx[n] for n in X_names), dtype=np.int64)
      3 
      4 X_train_sel = X_train_scaled[:, sel_idx].astype(np.float64, copy=False)
      5 X_test_sel = X_test_scaled[:, sel_idx].astype(np.float64, copy=False)

NameError: name 'X_cols' is not defined

## === cell 9
DO_PLOTS = False
if DO_PLOTS:
    import matplotlib.pyplot as plt

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




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/666513535.py in <cell line: 0>()
      2 
      3 svr_rbf = SVR(kernel="rbf", C=3.0, gamma=0.03, degree=2, epsilon=0.2, coef0=0)
----> 4 svr_rbf.fit(X_train_sel, y_train_sel)
      5 
      6 pred_scaled = svr_rbf.predict(X_test_sel).reshape(-1, 1)

NameError: name 'X_train_sel' is not defined

## === cell 11
submission = sample_sub.copy()
submission["time_to_eruption"] = predicted

submission["segment_id"] = submission["segment_id"].astype(int)
submission["time_to_eruption"] = submission["time_to_eruption"].astype(float)

submission.head()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3226371802.py in <cell line: 0>()
      1 submission = sample_sub.copy()
----> 2 submission["time_to_eruption"] = predicted
      3 
      4 submission["segment_id"] = submission["segment_id"].astype(int)
      5 submission["time_to_eruption"] = submission["time_to_eruption"].astype(float)

NameError: name 'predicted' is not defined

## === cell 12
SUB_PATH = "submission.csv"
submission.to_csv(SUB_PATH, index=False)
print(
    f"Wrote {SUB_PATH} with shape {submission.shape} and columns {submission.columns.tolist()}"
)




## === cell 13
pass
