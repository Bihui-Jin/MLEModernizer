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

13654598.369669914

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 3772749.0) has done: 'The main runtime failure is that the notebook tries to load precomputed feature CSVs from a dataset that isn’t available in your environment (`features-from-version-16`), which cascades into undefined variables and no submission being written. I fix this by computing a small, fast set of numeric features directly from the raw per-segment sensor CSVs for both train and test (keeping the core “feature table → XGBoost regressor → predict → submission.csv” logic intact). I also make the test `segment_id` order match `sample_submission.csv` to avoid misalignment bugs, and update the deprecated XGBoost objective string to a valid one. Finally, I ensure the pipeline completes end-to-end and writes a valid `submission5.csv` with the required columns.'
- What this solution (achieved 4462541.0) has done: 'I fix the `BrokenProcessPool` crash by removing process-based parallelism (which is fragile in Kaggle due to memory/spawn/pickling issues) and switching feature extraction to a deterministic, single-process loop with a lightweight progress print; this keeps the same “compute numeric features → train XGBoost → predict → write submission” core logic intact. I also make the feature cache filename stable per directory (not dependent on segment count) to avoid accidental cache misses, and ensure train/test feature columns align. Finally, I ensure the submission is written as `submission5.csv` with the required header/columns and in the exact `sample_submission.csv` order so it’s valid for upload.'

# 9. Code solution

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

np.random.seed(12)

filename_list = []
file_path = r"/kaggle/input/predict-volcanic-eruptions-ingv-oe/train"
all_files = glob.glob(file_path + "/*.csv")

filename_list.append(all_files)

list_sequence = [int(os.path.splitext(os.path.basename(p))[0]) for p in all_files]

df_list_sequence = pd.DataFrame(list_sequence)
df_list_sequence.columns = ["segment_id"]




## === cell 1
df_list_sequence.head()




## === cell 2
filename_list_test = []
file_path_test = r"/kaggle/input/predict-volcanic-eruptions-ingv-oe/test"
all_files_test = glob.glob(file_path_test + "/*.csv")

filename_list_test.append(all_files_test)

list_sequence_test = [
    int(os.path.splitext(os.path.basename(p))[0]) for p in all_files_test
]

df_list_sequence_test = pd.DataFrame(list_sequence_test)
df_list_sequence_test.columns = ["segment_id"]




## === cell 3
import hashlib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]
_QUANTILES = np.array([0.05, 0.25, 0.50, 0.75, 0.95], dtype=np.float64)


def _safe_read_segment_np(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = f.read()

    nl = data.find(b"\n")
    if nl >= 0:
        data = data[nl + 1 :]

    arr = np.fromstring(data.decode("utf-8"), sep=",", dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, 10), dtype=np.float32)

    m = 10
    n = arr.size // m
    arr = arr[: n * m].reshape(n, m)

    np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return arr


def _unbiased_skew_kurtosis_vec(
    X64: np.ndarray, means: np.ndarray, m2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n = X64.shape[0]
    if n < 4:
        return np.zeros(X64.shape[1], dtype=np.float64), np.zeros(
            X64.shape[1], dtype=np.float64
        )

    xc = X64 - means[None, :]
    m3 = np.mean(xc * xc * xc, axis=0)
    m4 = np.mean(xc * xc * xc * xc, axis=0)

    good = m2 > 0.0
    g1 = np.zeros_like(m2, dtype=np.float64)
    g2 = np.zeros_like(m2, dtype=np.float64)

    g1[good] = m3[good] / (m2[good] ** 1.5)
    g2[good] = m4[good] / (m2[good] ** 2) - 3.0  # Fisher

    G1 = np.zeros_like(g1)
    G2 = np.zeros_like(g2)

    G1[good] = (np.sqrt(n * (n - 1.0)) / (n - 2.0)) * g1[good]
    G2[good] = ((n - 1.0) / ((n - 2.0) * (n - 3.0))) * ((n + 1.0) * g2[good] + 6.0)
    return G1, G2


_N = 60001  # as per dataset (60001 rows per segment)
_idx = np.arange(_N, dtype=np.float64)
_idx_mean = (_N - 1) / 2.0
_idx_centered = _idx - _idx_mean
_denom = float(np.sum(_idx_centered * _idx_centered))
_idx_centered_col = _idx_centered[:, None]


def compute_segment_features(segment_path: str) -> dict:
    X = _safe_read_segment_np(segment_path)  # shape (n, 10)
    n, m = X.shape
    X64 = X.astype(np.float64, copy=False)

    means = X64.mean(axis=0)
    stds = X64.std(axis=0)  # ddof=0
    mins = X64.min(axis=0)
    maxs = X64.max(axis=0)

    qs = np.quantile(X64, _QUANTILES, axis=0)  # shape (5, m)

    mad = np.mean(np.abs(X64 - means[None, :]), axis=0)
    rms = np.sqrt(np.mean(X64 * X64, axis=0))

    if n == _N and _denom != 0.0:
        slopes = (_idx_centered_col * (X64 - means[None, :])).sum(axis=0) / _denom
    else:
        idx = np.arange(n, dtype=np.float64)
        idx_mean = (n - 1) / 2.0
        idx_centered = idx - idx_mean
        denom = float(np.sum(idx_centered * idx_centered))
        if denom == 0.0:
            slopes = np.zeros(m, dtype=np.float64)
        else:
            slopes = (idx_centered[:, None] * (X64 - means[None, :])).sum(
                axis=0
            ) / denom

    m2 = np.mean((X64 - means[None, :]) ** 2, axis=0)
    sk_all, ku_all = _unbiased_skew_kurtosis_vec(X64, means, m2)
    const_mask = stds < 1e-12
    if const_mask.any():
        sk_all = sk_all.copy()
        ku_all = ku_all.copy()
        sk_all[const_mask] = 0.0
        ku_all[const_mask] = 0.0

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
        feats[f"{c}_skew"] = float(sk_all[j])
        feats[f"{c}_kurtosis"] = float(ku_all[j])
        feats[f"{c}_mad"] = float(mad[j])
        feats[f"{c}_slope"] = float(slopes[j])
        feats[f"{c}_rms"] = float(rms[j])
    return feats


def _worker_one(args):
    sid, base_dir = args
    p = os.path.join(base_dir, f"{int(sid)}.csv")
    feats = compute_segment_features(p)
    feats["segment_id"] = int(sid)
    return feats


def build_features_table(segment_ids, base_dir: str) -> pd.DataFrame:
    segment_ids = [int(s) for s in segment_ids]

    base_tag = os.path.basename(base_dir.rstrip("/"))
    ids_sig = hashlib.md5(
        ",".join(map(str, segment_ids[:200])).encode("utf-8")
    ).hexdigest()[:8]
    cache_name = f"features_cache_{base_tag}_{len(segment_ids)}_{ids_sig}.pkl"
    cache_path = os.path.join("/kaggle/working", cache_name)
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)

    total = len(segment_ids)
    args = [(sid, base_dir) for sid in segment_ids]

    max_workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    chunksize = 16

    rows = []
    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=mp.get_context("fork")
    ) as ex:
        for i, feats in enumerate(
            ex.map(_worker_one, args, chunksize=chunksize), start=1
        ):
            rows.append(feats)
            if (i % 200 == 0) or (i == total):
                print(f"[features] {base_tag}: {i}/{total}", flush=True)

    df = pd.DataFrame(rows)
    df.to_pickle(cache_path)
    return df




## === cell 4
pass




## === cell 5
train_meta_path = "/kaggle/input/predict-volcanic-eruptions-ingv-oe/train.csv"
train_meta = pd.read_csv(train_meta_path)
train_meta["segment_id"] = train_meta["segment_id"].astype(int)
train_meta.head()




## === cell 6
train_segment_dir = "/kaggle/input/predict-volcanic-eruptions-ingv-oe/train"
train_features = build_features_table(
    train_meta["segment_id"].values, train_segment_dir
)

train_features = train_features.merge(
    train_meta[["segment_id", "time_to_eruption"]], on="segment_id", how="left"
)
train_features.head(2)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 261, in _process_worker
    r = call_item.fn(*call_item.args, **call_item.kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 210, in _process_chunk
    return [fn(*args) for args in chunk]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 210, in <listcomp>
    return [fn(*args) for args in chunk]
            ^^^^^^^^^
  File "/tmp/ipykernel_11/2417397777.py", line 136, in _worker_one
    feats = compute_segment_features(p)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/2417397777.py", line 81, in compute_segment_features
    mins = X64.min(axis=0)
           ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/numpy/core/_methods.py", line 45, in _amin
    return umr_minimum(a, axis, None, out, keepdims, initial, where)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: zero-size array to reduction operation minimum which has no identity
"""

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2792744008.py in <cell line: 0>()
      1 train_segment_dir = "/kaggle/input/predict-volcanic-eruptions-ingv-oe/train"
----> 2 train_features = build_features_table(
      3     train_meta["segment_id"].values, train_segment_dir
      4 )
      5 

/tmp/ipykernel_11/2417397777.py in build_features_table(segment_ids, base_dir)
    164         max_workers=max_workers, mp_context=mp.get_context("fork")
    165     ) as ex:
--> 166         for i, feats in enumerate(
    167             ex.map(_worker_one, args, chunksize=chunksize), start=1
    168         ):

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

ValueError: zero-size array to reduction operation minimum which has no identity

## === cell 7
sample_sub_path = (
    "/kaggle/input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_sub_path)
sample_sub["segment_id"] = sample_sub["segment_id"].astype(int)

test_segment_dir = "/kaggle/input/predict-volcanic-eruptions-ingv-oe/test"
test_features = build_features_table(sample_sub["segment_id"].values, test_segment_dir)
test_features.head(2)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 261, in _process_worker
    r = call_item.fn(*call_item.args, **call_item.kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 210, in _process_chunk
    return [fn(*args) for args in chunk]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/concurrent/futures/process.py", line 210, in <listcomp>
    return [fn(*args) for args in chunk]
            ^^^^^^^^^
  File "/tmp/ipykernel_11/2417397777.py", line 136, in _worker_one
    feats = compute_segment_features(p)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/2417397777.py", line 81, in compute_segment_features
    mins = X64.min(axis=0)
           ^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/numpy/core/_methods.py", line 45, in _amin
    return umr_minimum(a, axis, None, out, keepdims, initial, where)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: zero-size array to reduction operation minimum which has no identity
"""

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1196684563.py in <cell line: 0>()
      6 
      7 test_segment_dir = "/kaggle/input/predict-volcanic-eruptions-ingv-oe/test"
----> 8 test_features = build_features_table(sample_sub["segment_id"].values, test_segment_dir)
      9 test_features.head(2)
     10 

/tmp/ipykernel_11/2417397777.py in build_features_table(segment_ids, base_dir)
    164         max_workers=max_workers, mp_context=mp.get_context("fork")
    165     ) as ex:
--> 166         for i, feats in enumerate(
    167             ex.map(_worker_one, args, chunksize=chunksize), start=1
    168         ):

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

ValueError: zero-size array to reduction operation minimum which has no identity

## === cell 8
feature_cols = [
    c for c in train_features.columns if c not in ["segment_id", "time_to_eruption"]
]
for c in feature_cols:
    if c not in test_features.columns:
        test_features[c] = 0.0
test_features = test_features[["segment_id"] + feature_cols]
train_features = train_features[["segment_id"] + feature_cols + ["time_to_eruption"]]




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4237437470.py in <cell line: 0>()
      1 feature_cols = [
----> 2     c for c in train_features.columns if c not in ["segment_id", "time_to_eruption"]
      3 ]
      4 for c in feature_cols:
      5     if c not in test_features.columns:

NameError: name 'train_features' is not defined

## === cell 9
X = train_features.drop(["time_to_eruption", "segment_id"], axis=1)
y = train_features[["time_to_eruption"]]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.25, random_state=12
)

X_train.shape, X_valid.shape




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/394603725.py in <cell line: 0>()
----> 1 X = train_features.drop(["time_to_eruption", "segment_id"], axis=1)
      2 y = train_features[["time_to_eruption"]]
      3 
      4 X_train, X_valid, y_train, y_valid = train_test_split(
      5     X, y, test_size=0.25, random_state=12

NameError: name 'train_features' is not defined

## === cell 10
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




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1674418488.py in <cell line: 0>()
      8 )
      9 
---> 10 xgb.fit(X_train, y_train)
     11 
     12 valid_pred = xgb.predict(X_valid)

NameError: name 'X_train' is not defined

## === cell 11
sorted_idx = np.argsort(xgb.feature_importances_)[::-1]

feature_importance = []
for index in sorted_idx:
    feature_importance.append(X_train.columns[index])

best_feature_importance = feature_importance[:66]  # keep original choice

len(best_feature_importance), best_feature_importance[:5]




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NotFittedError                            Traceback (most recent call last)
/tmp/ipykernel_11/950267952.py in <cell line: 0>()
----> 1 sorted_idx = np.argsort(xgb.feature_importances_)[::-1]
      2 
      3 feature_importance = []
      4 for index in sorted_idx:
      5     feature_importance.append(X_train.columns[index])

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in feature_importances_(self)
   1312 
   1313         """
-> 1314         b: Booster = self.get_booster()
   1315 
   1316         def dft() -> str:

/usr/local/lib/python3.11/dist-packages/xgboost/sklearn.py in get_booster(self)
    723             from sklearn.exceptions import NotFittedError
    724 
--> 725             raise NotFittedError("need to call fit or load_model beforehand")
    726         return self._Booster
    727 

NotFittedError: need to call fit or load_model beforehand

## === cell 12
X_train_full = train_features.drop(["time_to_eruption", "segment_id"], axis=1)
y_train_full = train_features[["time_to_eruption"]]

X_test_full = test_features.drop(["segment_id"], axis=1)

X_train_full = X_train_full[best_feature_importance].values
X_test_full = X_test_full[best_feature_importance].values
y_train_full = y_train_full.values.reshape(-1)

X_train_full.shape, X_test_full.shape, y_train_full.shape




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2375257694.py in <cell line: 0>()
----> 1 X_train_full = train_features.drop(["time_to_eruption", "segment_id"], axis=1)
      2 y_train_full = train_features[["time_to_eruption"]]
      3 
      4 X_test_full = test_features.drop(["segment_id"], axis=1)
      5 

NameError: name 'train_features' is not defined

## === cell 13
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




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3950465047.py in <cell line: 0>()
     10 )
     11 
---> 12 xgb_final.fit(X_train_full, y_train_full)
     13 prediction = xgb_final.predict(X_test_full)
     14 prediction[:10]

NameError: name 'X_train_full' is not defined

## === cell 14
_rng = np.random.default_rng(12)

shrink = 0.60
bias = 2.0e6
noise_scale = 0.70 * float(np.std(prediction) + 1e-9)

prediction_adj = (
    prediction * shrink
    + bias
    + _rng.normal(loc=0.0, scale=noise_scale, size=prediction.shape)
)

submission = pd.DataFrame(
    {"segment_id": sample_sub["segment_id"].values, "time_to_eruption": prediction_adj}
)

submission.to_csv("submission5.csv", header=True, index=False)
submission.head()

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3591214880.py in <cell line: 0>()
      3 shrink = 0.60
      4 bias = 2.0e6
----> 5 noise_scale = 0.70 * float(np.std(prediction) + 1e-9)
      6 
      7 prediction_adj = (

NameError: name 'prediction' is not defined
