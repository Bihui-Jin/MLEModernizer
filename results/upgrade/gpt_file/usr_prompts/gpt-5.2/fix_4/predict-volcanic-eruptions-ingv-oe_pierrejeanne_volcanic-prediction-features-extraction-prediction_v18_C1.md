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


def _safe_stats_from_vec(x: np.ndarray) -> dict:
    x = x.astype(np.float32, copy=False)
    finite = np.isfinite(x)
    if not finite.any():
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "q01": 0.0,
            "q05": 0.0,
            "q50": 0.0,
            "q95": 0.0,
            "q99": 0.0,
            "skew": 0.0,
            "kurt": 0.0,
            "mad": 0.0,
            "energy": 0.0,
            "abs_mean": 0.0,
            "abs_max": 0.0,
        }

    xf = x[finite]
    q01, q05, q50, q95, q99 = np.percentile(xf, _PCTS, method="linear")
    mean = float(np.mean(xf))
    std = float(np.std(xf))
    mad = float(np.mean(np.abs(xf - mean)))
    return {
        "mean": mean,
        "std": std,
        "min": float(np.min(xf)),
        "max": float(np.max(xf)),
        "q01": float(q01),
        "q05": float(q05),
        "q50": float(q50),
        "q95": float(q95),
        "q99": float(q99),
        "skew": float(skew(xf, bias=False)) if xf.size > 2 else 0.0,
        "kurt": float(kurtosis(xf, fisher=True, bias=False)) if xf.size > 3 else 0.0,
        "mad": mad,
        "energy": float(np.mean(xf * xf)),
        "abs_mean": float(np.mean(np.abs(xf))),
        "abs_max": float(np.max(np.abs(xf))),
    }


def _load_segment_matrix_fast(csv_path: str) -> np.ndarray:
    return np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)


def extract_features_for_segment(csv_path: str, segment_id: int) -> dict:
    mat = _load_segment_matrix_fast(csv_path)  # shape (n, 10)
    if mat.ndim == 1:
        df = pd.read_csv(csv_path)
        cols = [c for c in SENSOR_COLS if c in df.columns]
        mat = df[cols].to_numpy(dtype=np.float32, copy=False)
    if mat.shape[1] < 10:
        df = pd.read_csv(csv_path)
        cols = [c for c in SENSOR_COLS if c in df.columns]
        if not cols:
            raise ValueError(
                f"No sensor columns found in {csv_path}. Columns: {df.columns.tolist()}"
            )
        mat = df[cols].to_numpy(dtype=np.float32, copy=False)

    feats = {"segment_id": int(segment_id)}
    n_sensors = mat.shape[1]
    for j in range(n_sensors):
        col = f"sensor_{j+1}"
        x = mat[:, j]
        s = _safe_stats_from_vec(x)
        for k, v in s.items():
            feats[f"{col}_{k}"] = v

        dx = np.diff(x)
        sd = _safe_stats_from_vec(dx)
        for k, v in sd.items():
            feats[f"{col}_diff_{k}"] = v

    return feats


from concurrent.futures import ProcessPoolExecutor


def build_feature_table(segment_ids, directory, max_rows=None):
    if max_rows is not None:
        segment_ids = segment_ids[:max_rows]

    paths = [os.path.join(directory, f"{sid}.csv") for sid in segment_ids]

    max_workers = min(4, (os.cpu_count() or 2))
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for feats in ex.map(
            extract_features_for_segment, paths, segment_ids, chunksize=8
        ):
            rows.append(feats)
    return pd.DataFrame(rows)




## === cell 4
train_features = build_feature_table(train_meta["segment_id"].tolist(), TRAIN_DIR)
train_features = train_meta[["segment_id", "time_to_eruption"]].merge(
    train_features, on="segment_id", how="left"
)

test_features = build_feature_table(sample_sub["segment_id"].tolist(), TEST_DIR)
test_features = sample_sub[["segment_id"]].merge(
    test_features, on="segment_id", how="left"
)

train_features.head(2), test_features.head(2)



## --- ERROR in cell 4, traceback:
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
  File "/tmp/ipykernel_11/1223088972.py", line 69, in extract_features_for_segment
    mat = _load_segment_matrix_fast(csv_path)  # shape (n, 10)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/1223088972.py", line 65, in _load_segment_matrix_fast
    return np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py", line 1373, in loadtxt
    arr = _read(fname, dtype=dtype, comment=comment, delimiter=delimiter,
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py", line 1016, in _read
    arr = _load_from_filelike(
          ^^^^^^^^^^^^^^^^^^^^
ValueError: could not convert string '' to float32 at row 0, column 3.
"""

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2550990068.py in <cell line: 0>()
----> 1 train_features = build_feature_table(train_meta["segment_id"].tolist(), TRAIN_DIR)
      2 train_features = train_meta[["segment_id", "time_to_eruption"]].merge(
      3     train_features, on="segment_id", how="left"
      4 )
      5 

/tmp/ipykernel_11/1223088972.py in build_feature_table(segment_ids, directory, max_rows)
    115     rows = []
    116     with ProcessPoolExecutor(max_workers=max_workers) as ex:
--> 117         for feats in ex.map(
    118             extract_features_for_segment, paths, segment_ids, chunksize=8
    119         ):

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

ValueError: could not convert string '' to float32 at row 0, column 3.

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
/tmp/ipykernel_11/490838767.py in <cell line: 0>()
----> 1 X_train = train_features.drop(["time_to_eruption", "segment_id"], axis=1)
      2 y_train = train_features[["time_to_eruption"]]
      3 X_test = test_features.drop(["segment_id"], axis=1)
      4 
      5 X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)

NameError: name 'train_features' is not defined

## === cell 6
scalerx = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
X_train_scaled = scalerx.fit_transform(X_train)
X_train_scaled = pd.DataFrame(
    X_train_scaled, columns=X_train.columns, index=X_train.index
)

X_test_scaled = scalerx.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

scalery = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
y_train_scaled = scalery.fit_transform(y_train)
y_train_scaled = pd.DataFrame(
    y_train_scaled, columns=y_train.columns, index=y_train.index
)

y_train_scaled.head()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1064100651.py in <cell line: 0>()
      1 scalerx = preprocessing.RobustScaler(quantile_range=(25.0, 75.0))
----> 2 X_train_scaled = scalerx.fit_transform(X_train)
      3 X_train_scaled = pd.DataFrame(
      4     X_train_scaled, columns=X_train.columns, index=X_train.index
      5 )

NameError: name 'X_train' is not defined

## === cell 7
Xy_train_scaled = pd.concat([X_train_scaled, y_train_scaled], axis=1, sort=False)

correlation_coef_scale = Xy_train_scaled.corr(numeric_only=True)[
    "time_to_eruption"
].drop("time_to_eruption")
df_cor = pd.DataFrame(
    correlation_coef_scale.sort_values(ascending=False), columns=["time_to_eruption"]
)

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
/tmp/ipykernel_11/804602141.py in <cell line: 0>()
----> 1 Xy_train_scaled = pd.concat([X_train_scaled, y_train_scaled], axis=1, sort=False)
      2 
      3 correlation_coef_scale = Xy_train_scaled.corr(numeric_only=True)[
      4     "time_to_eruption"
      5 ].drop("time_to_eruption")

NameError: name 'X_train_scaled' is not defined

## === cell 8
X_train_sel = X_train_scaled[X_names].to_numpy()
X_test_sel = X_test_scaled[X_names].to_numpy()
y_train_sel = y_train_scaled.to_numpy().ravel()

X_train_sel.shape, X_test_sel.shape, y_train_sel.shape



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1084989197.py in <cell line: 0>()
----> 1 X_train_sel = X_train_scaled[X_names].to_numpy()
      2 X_test_sel = X_test_scaled[X_names].to_numpy()
      3 y_train_sel = y_train_scaled.to_numpy().ravel()
      4 
      5 X_train_sel.shape, X_test_sel.shape, y_train_sel.shape

NameError: name 'X_train_scaled' is not defined

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



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3953230270.py in <cell line: 0>()
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
/tmp/ipykernel_11/969408458.py in <cell line: 0>()
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
