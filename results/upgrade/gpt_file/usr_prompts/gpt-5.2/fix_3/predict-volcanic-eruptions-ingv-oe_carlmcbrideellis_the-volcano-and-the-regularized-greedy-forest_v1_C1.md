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
rgf-python==3.12.0
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

7173796.816445133

# 6. Current score

5496425.0

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 5496425.0) has done: 'The timeout is dominated by per-file `pd.read_csv` overhead across ~4,400 segments; the core model fit is comparatively cheap once features are built. I keep the exact same features (mean/std/min/max per sensor) and the same RGFRegressor settings, but speed up feature extraction by (1) reading CSVs with `dtype=np.float32` and `engine="c"` to avoid extra conversions, (2) computing all stats vectorized in NumPy (single pass per file) instead of looping columns with Pandas Series ops, and (3) parallelizing feature extraction across CPU cores with deterministic ordering. I also avoid expensive `apply(pd.to_numeric)` on already-numeric data and ensure column alignment without changing semantics.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd
import numpy as np

BASE_PATH = "/kaggle/input/predict-volcanic-eruptions-ingv-oe"

train_meta_path = os.path.join(BASE_PATH, "train.csv")
sample_path = os.path.join(BASE_PATH, "sample_submission.csv")
train_dir = os.path.join(BASE_PATH, "train")
test_dir = os.path.join(BASE_PATH, "test")

if not os.path.exists(train_meta_path):
    raise FileNotFoundError(f"train.csv not found at: {train_meta_path}")
if not os.path.exists(sample_path):
    raise FileNotFoundError(f"sample_submission.csv not found at: {sample_path}")
if not os.path.isdir(train_dir):
    raise FileNotFoundError(f"train directory not found at: {train_dir}")
if not os.path.isdir(test_dir):
    raise FileNotFoundError(f"test directory not found at: {test_dir}")

train_meta = pd.read_csv(train_meta_path)
sample = pd.read_csv(sample_path)

train_meta["segment_id"] = train_meta["segment_id"].astype(str)
sample["segment_id"] = sample["segment_id"].astype(str)



## === cell 1
import os
from concurrent.futures import ProcessPoolExecutor

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 11)]
READ_KWARGS = dict(
    dtype=np.float32, engine="c"
)  # exact values preserved; just faster parsing


def extract_features_for_segment(csv_path: str) -> dict:
    seg_id = os.path.splitext(os.path.basename(csv_path))[0]

    df = pd.read_csv(csv_path, usecols=SENSOR_COLS, **READ_KWARGS)
    x = df.to_numpy(dtype=np.float32, copy=False)  # shape (n_rows, 10)

    means = np.nanmean(x, axis=0)
    stds = np.nanstd(x, axis=0, ddof=1)  # pandas Series.std default ddof=1
    mins = np.nanmin(x, axis=0)
    maxs = np.nanmax(x, axis=0)

    feats = {"segment_id": seg_id}
    for i, col in enumerate(SENSOR_COLS):
        feats[f"{col}_mean"] = float(means[i])
        feats[f"{col}_std"] = float(stds[i])
        feats[f"{col}_min"] = float(mins[i])
        feats[f"{col}_max"] = float(maxs[i])
    return feats


def build_feature_df(file_paths):
    file_paths = list(file_paths)
    if not file_paths:
        return pd.DataFrame()

    max_workers = min(8, (os.cpu_count() or 2))
    if max_workers <= 1:
        rows = [extract_features_for_segment(p) for p in file_paths]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            rows = list(ex.map(extract_features_for_segment, file_paths, chunksize=16))
    return pd.DataFrame(rows)


train_files = sorted(glob.glob(os.path.join(train_dir, "*.csv")))
test_files = sorted(glob.glob(os.path.join(test_dir, "*.csv")))

if len(train_files) == 0 or len(test_files) == 0:
    raise RuntimeError(
        f"No segment files found. train_files={len(train_files)}, test_files={len(test_files)}"
    )

train_feat = build_feature_df(train_files)
test_feat = build_feature_df(test_files)

train = train_feat.merge(
    train_meta[["segment_id", "time_to_eruption"]], on="segment_id", how="inner"
)
if train.shape[0] != train_meta.shape[0]:
    missing = set(train_meta["segment_id"]) - set(train["segment_id"])
    raise RuntimeError(
        f"Mismatch after merge: got {train.shape[0]} rows, expected {train_meta.shape[0]}. Missing={len(missing)}"
    )

test = test_feat.copy()



## === cell 2
X_train = train.drop(["segment_id", "time_to_eruption"], axis=1)
y_train = train["time_to_eruption"].astype(np.float64)
X_test = test.drop(["segment_id"], axis=1)

X_train = X_train.fillna(0.0)
X_test = X_test.fillna(0.0)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)



## === cell 3
from rgf.sklearn import RGFRegressor

regressor = RGFRegressor(
    max_leaf=2000, algorithm="RGF_Sib", test_interval=100, loss="LS", verbose=False
)

regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)



## === cell 4
pred_df = pd.DataFrame(
    {
        "segment_id": test["segment_id"].astype(str).values,
        "time_to_eruption": predictions,
    }
)

submission = sample[["segment_id"]].merge(pred_df, on="segment_id", how="left")

if submission["time_to_eruption"].isna().any():
    n_missing = int(submission["time_to_eruption"].isna().sum())
    raise RuntimeError(
        f"Submission has {n_missing} missing predictions after merge; check segment_id alignment."
    )

submission.to_csv("submission.csv", index=False)
print(submission.head())
print(
    f"Wrote submission.csv with shape={submission.shape} and columns={list(submission.columns)}"
)
