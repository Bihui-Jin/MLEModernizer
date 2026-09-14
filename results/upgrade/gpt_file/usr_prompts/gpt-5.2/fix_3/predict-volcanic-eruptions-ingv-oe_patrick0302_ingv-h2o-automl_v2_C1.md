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

# 5. Code solution

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

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def _segment_features_from_file(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path, dtype=np.float32, engine="c")
    if not all(np.issubdtype(t, np.number) for t in df.dtypes):
        df = df.apply(pd.to_numeric, errors="coerce").astype("float32")

    cols = df.columns.tolist()
    arr = df.to_numpy(dtype=np.float32, copy=False)

    feats = {}
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0)
    mins = np.nanmin(arr, axis=0)
    maxs = np.nanmax(arr, axis=0)
    medians = np.nanmedian(arr, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)

    skew_s = df.skew(skipna=True)
    kurt_s = df.kurt(skipna=True)

    for i, c in enumerate(cols):
        feats[f"{c}_mean"] = float(means[i])
        feats[f"{c}_std"] = float(stds[i])
        feats[f"{c}_min"] = float(mins[i])
        feats[f"{c}_max"] = float(maxs[i])
        feats[f"{c}_median"] = float(medians[i])
        feats[f"{c}_q25"] = float(q25[i])
        feats[f"{c}_q75"] = float(q75[i])
        feats[f"{c}_skew_approx"] = float(skew_s.iloc[i])
        feats[f"{c}_kurt_approx"] = float(kurt_s.iloc[i])

    feats["nan_count"] = int(np.isnan(arr).sum())
    return feats


def _features_for_one_sid(args):
    sid, base_dir = args
    p = Path(base_dir) / f"{sid}.csv"
    if not p.exists():
        return {"segment_id": sid}, True
    feats = _segment_features_from_file(p)
    feats["segment_id"] = sid
    return feats, False


def build_feature_df(segment_ids, base_dir: Path) -> pd.DataFrame:
    segs = list(segment_ids)
    tasks = [(sid, str(base_dir)) for sid in segs]

    rows = [None] * len(segs)
    missing = 0

    max_workers = min(4, (os.cpu_count() or 2))
    ctx = mp.get_context("fork") if hasattr(mp, "get_context") else None

    Executor = ProcessPoolExecutor
    with Executor(max_workers=max_workers, mp_context=ctx) as ex:
        futures = [ex.submit(_features_for_one_sid, t) for t in tasks]
        for i, fut in enumerate(futures):
            feats, was_missing = fut.result()
            rows[i] = feats
            if was_missing:
                missing += 1

    feat_df = pd.DataFrame(rows)
    if missing:
        print(f"Warning: {missing} segment files missing under {base_dir}")
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



## === cell 4
train_h2o = h2o.H2OFrame(X_train_df)
train_label_h2o = h2o.H2OFrame(train[["time_to_eruption"]])
train_h2o["time_to_eruption"] = train_label_h2o["time_to_eruption"]

test_feature_h2o = h2o.H2OFrame(X_test_df)

print(train_h2o.shape)
print(test_feature_h2o.shape)



## === cell 5
x = test_feature_h2o.columns
y = "time_to_eruption"



## === cell 6
aml = H2OAutoML(max_models=1000, seed=121, max_runtime_secs=1 * 60)
aml.train(x=x, y=y, training_frame=train_h2o)



## === cell 7
lb = aml.leaderboard
lb.head(rows=lb.nrows)



## === cell 8
aml.leader



## === cell 9
preds = aml.predict(test_feature_h2o)



## === cell 10
submission = pd.DataFrame()
submission["segment_id"] = test["segment_id"].astype(np.int64)
pred_arr = preds.as_data_frame(use_pandas=True).iloc[:, 0].to_numpy()
submission["time_to_eruption"] = pred_arr.astype(float)

submission.to_csv("submission_recent.csv", header=True, index=False)



## === cell 11
submission
