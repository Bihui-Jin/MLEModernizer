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
h2o==3.46.0.8
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

7791545.688837831

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 5211002.0) has done: 'Your notebook doesn’t yield a Kaggle score because it writes `submission_recent.csv` instead of the required `submission.csv`, and it can also silently misalign predictions if the feature-building multiprocessing returns rows out of order. I make two minimal fixes: (1) preserve segment order deterministically by returning `(segment_id, feats)` from workers and rebuilding the DataFrame in the original id order, and (2) write the submission to `submission.csv` with the exact required columns and order matching `sample_submission.csv`. These changes keep your feature set and H2OAutoML training logic intact while ensuring the submission is valid and correctly aligned (which also avoids a potentially very large MAE from mismatched rows).'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pathlib import Path



## === cell 1
import h2o

print(h2o.__version__)
from h2o.automl import H2OAutoML

h2o.init(max_mem_size="16G")



## === cell 2
train = pd.read_csv("/kaggle/input/predict-volcanic-eruptions-ingv-oe/train.csv")
test = pd.read_csv(
    "/kaggle/input/predict-volcanic-eruptions-ingv-oe/sample_submission.csv"
)

train_dir = Path("/kaggle/input/predict-volcanic-eruptions-ingv-oe/train")
test_dir = Path("/kaggle/input/predict-volcanic-eruptions-ingv-oe/test")

assert train_dir.exists(), f"Train directory not found: {train_dir}"
assert test_dir.exists(), f"Test directory not found: {test_dir}"



## === cell 3
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]


def _nan_skew_kurt(arr_2d: np.ndarray):
    """
    Compute per-column skewness and kurtosis (Fisher=False, i.e., normal -> 3),
    with NaNs skipped, matching the previous implementation.
    """
    x = arr_2d.astype(np.float64, copy=False)
    n = np.sum(~np.isnan(x), axis=0).astype(np.float64)
    mu = np.nanmean(x, axis=0)
    xc = x - mu
    m2 = np.nanmean(xc * xc, axis=0)
    m3 = np.nanmean(xc * xc * xc, axis=0)
    m4 = np.nanmean(xc * xc * xc * xc, axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        g1 = m3 / np.power(m2, 1.5)
        g2 = m4 / (m2 * m2)

        skew = np.where(n > 2, (np.sqrt(n * (n - 1)) / (n - 2)) * g1, np.nan)

        kurt = np.where(
            n > 3,
            ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 - 3 * (n - 1)) + 3.0,
            np.nan,
        )
    return skew.astype(np.float32), kurt.astype(np.float32)


def _load_segment_array_fast(csv_path: Path) -> np.ndarray:
    """
    SPEEDUP (correctness-preserving): Use NumPy to parse the numeric CSV with a header.
    This avoids pandas overhead for 60k-row files and returns float32 with NaNs supported.
    """
    return np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)


def _segment_features_from_file(csv_path: Path) -> dict:
    arr32 = _load_segment_array_fast(csv_path)  # shape (60001, 10)
    x = arr32.astype(np.float64, copy=False)

    means = np.nanmean(x, axis=0)
    stds = np.nanstd(x, axis=0)
    mins = np.nanmin(x, axis=0)
    maxs = np.nanmax(x, axis=0)
    medians = np.nanmedian(x, axis=0)

    q25 = np.quantile(x, 0.25, axis=0, method="linear")
    q75 = np.quantile(x, 0.75, axis=0, method="linear")

    skew_s, kurt_s = _nan_skew_kurt(arr32)

    feats = {}
    for i, c in enumerate(SENSOR_COLS):
        feats[c + "_mean"] = float(means[i])
        feats[c + "_std"] = float(stds[i])
        feats[c + "_min"] = float(mins[i])
        feats[c + "_max"] = float(maxs[i])
        feats[c + "_median"] = float(medians[i])
        feats[c + "_q25"] = float(q25[i])
        feats[c + "_q75"] = float(q75[i])
        feats[c + "_skew_approx"] = float(skew_s[i])
        feats[c + "_kurt_approx"] = float(kurt_s[i])

    feats["nan_count"] = int(np.isnan(x).sum())
    return feats


def _features_for_one_sid(args):
    sid, base_dir = args
    p = Path(base_dir) / f"{sid}.csv"
    if not p.exists():
        return sid, {"segment_id": sid}, True
    feats = _segment_features_from_file(p)
    feats["segment_id"] = sid
    return sid, feats, False


def build_feature_df(segment_ids, base_dir: Path) -> pd.DataFrame:
    segs = list(segment_ids)
    tasks = [(sid, str(base_dir)) for sid in segs]

    cpu = os.cpu_count() or 2
    max_workers = min(16, max(4, cpu))
    chunksize = 64

    feat_map = {}
    missing = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for sid, feats, was_missing in ex.map(
            _features_for_one_sid, tasks, chunksize=chunksize
        ):
            feat_map[sid] = feats
            if was_missing:
                missing += 1

    if missing:
        print(f"Warning: {missing} segment files missing under {base_dir}")

    rows = [feat_map[sid] for sid in segs]
    feat_df = pd.DataFrame.from_records(rows)
    return feat_df


scaled_feature_df = build_feature_df(train["segment_id"].values, train_dir)
scaled_test_df = build_feature_df(test["segment_id"].values, test_dir)

feature_cols = [c for c in scaled_feature_df.columns if c != "segment_id"]
for c in feature_cols:
    if c not in scaled_test_df.columns:
        scaled_test_df[c] = np.nan
extra_test_cols = [
    c for c in scaled_test_df.columns if c not in feature_cols + ["segment_id"]
]
if extra_test_cols:
    scaled_test_df = scaled_test_df.drop(columns=extra_test_cols)

scaled_feature_df = scaled_feature_df[["segment_id"] + feature_cols]
scaled_test_df = scaled_test_df[["segment_id"] + feature_cols]

medians = scaled_feature_df[feature_cols].median(numeric_only=True)
scaled_feature_df[feature_cols] = scaled_feature_df[feature_cols].fillna(medians)
scaled_test_df[feature_cols] = scaled_test_df[feature_cols].fillna(medians)

X_train_df = scaled_feature_df.drop(columns=["segment_id"])
X_test_df = scaled_test_df.drop(columns=["segment_id"])



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1718507769.py in <cell line: 0>()
    119 
    120 
--> 121 scaled_feature_df = build_feature_df(train["segment_id"].values, train_dir)
    122 scaled_test_df = build_feature_df(test["segment_id"].values, test_dir)
    123 

/tmp/ipykernel_11/1718507769.py in build_feature_df(segment_ids, base_dir)
    104     with ThreadPoolExecutor(max_workers=max_workers) as ex:
    105         # executor.map preserves input order -> deterministic row order
--> 106         for sid, feats, was_missing in ex.map(
    107             _features_for_one_sid, tasks, chunksize=chunksize
    108         ):

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
    447                     raise CancelledError()
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 
    451                 self._condition.wait(timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_11/1718507769.py in _features_for_one_sid(args)
     85     if not p.exists():
     86         return sid, {"segment_id": sid}, True
---> 87     feats = _segment_features_from_file(p)
     88     feats["segment_id"] = sid
     89     return sid, feats, False

/tmp/ipykernel_11/1718507769.py in _segment_features_from_file(csv_path)
     48 
     49 def _segment_features_from_file(csv_path: Path) -> dict:
---> 50     arr32 = _load_segment_array_fast(csv_path)  # shape (60001, 10)
     51     x = arr32.astype(np.float64, copy=False)
     52 

/tmp/ipykernel_11/1718507769.py in _load_segment_array_fast(csv_path)
     44     # These competition files are simple: 10 numeric columns + header row.
     45     # np.loadtxt is fast in CPython and keeps NaNs if present.
---> 46     return np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32)
     47 
     48 

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in loadtxt(fname, dtype, comments, delimiter, converters, skiprows, usecols, unpack, ndmin, encoding, max_rows, quotechar, like)
   1371         delimiter = delimiter.decode('latin1')
   1372 
-> 1373     arr = _read(fname, dtype=dtype, comment=comment, delimiter=delimiter,
   1374                 converters=converters, skiplines=skiprows, usecols=usecols,
   1375                 unpack=unpack, ndmin=ndmin, encoding=encoding,

/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py in _read(fname, delimiter, comment, quote, imaginary_unit, usecols, skiplines, max_rows, converters, ndmin, unpack, dtype, encoding)
   1014 
   1015         if read_dtype_via_object_chunks is None:
-> 1016             arr = _load_from_filelike(
   1017                 data, delimiter=delimiter, comment=comment, quote=quote,
   1018                 imaginary_unit=imaginary_unit,

ValueError: could not convert string '' to float32 at row 0, column 3.

## === cell 4
train_h2o = h2o.H2OFrame(X_train_df)
train_label_h2o = h2o.H2OFrame(train[["time_to_eruption"]])
train_h2o["time_to_eruption"] = train_label_h2o["time_to_eruption"]

test_feature_h2o = h2o.H2OFrame(X_test_df)

print(train_h2o.shape)
print(test_feature_h2o.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4165478557.py in <cell line: 0>()
----> 1 train_h2o = h2o.H2OFrame(X_train_df)
      2 train_label_h2o = h2o.H2OFrame(train[["time_to_eruption"]])
      3 train_h2o["time_to_eruption"] = train_label_h2o["time_to_eruption"]
      4 
      5 test_feature_h2o = h2o.H2OFrame(X_test_df)

NameError: name 'X_train_df' is not defined

## === cell 5
x = test_feature_h2o.columns
y = "time_to_eruption"



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2734472956.py in <cell line: 0>()
----> 1 x = test_feature_h2o.columns
      2 y = "time_to_eruption"
      3 

NameError: name 'test_feature_h2o' is not defined

## === cell 6
aml = H2OAutoML(max_models=1000, seed=121, max_runtime_secs=1 * 60)
aml.train(x=x, y=y, training_frame=train_h2o)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2058796779.py in <cell line: 0>()
      1 aml = H2OAutoML(max_models=1000, seed=121, max_runtime_secs=1 * 60)
----> 2 aml.train(x=x, y=y, training_frame=train_h2o)
      3 

NameError: name 'x' is not defined

## === cell 7
lb = aml.leaderboard
lb.head(rows=lb.nrows)



## === cell 8
aml.leader



## === cell 9
preds = aml.predict(test_feature_h2o)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3313764495.py in <cell line: 0>()
----> 1 preds = aml.predict(test_feature_h2o)
      2 

NameError: name 'test_feature_h2o' is not defined

## === cell 10
submission = test[["segment_id"]].copy()
pred_arr = preds.as_data_frame(use_pandas=True).iloc[:, 0].to_numpy()
submission["time_to_eruption"] = pred_arr.astype(float)

submission.to_csv("submission.csv", header=True, index=False)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/259374751.py in <cell line: 0>()
      1 submission = test[["segment_id"]].copy()
----> 2 pred_arr = preds.as_data_frame(use_pandas=True).iloc[:, 0].to_numpy()
      3 submission["time_to_eruption"] = pred_arr.astype(float)
      4 
      5 submission.to_csv("submission.csv", header=True, index=False)

NameError: name 'preds' is not defined

## === cell 11
submission
