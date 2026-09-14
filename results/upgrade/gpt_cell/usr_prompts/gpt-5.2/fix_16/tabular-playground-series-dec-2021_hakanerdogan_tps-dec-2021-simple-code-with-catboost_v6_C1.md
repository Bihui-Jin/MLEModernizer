# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict the class of a given image from a synthetic dataset.

## MetricMulti-class classification accuracy.

## Submission FormatFor each `Id` in the test set, you must predict the `Cover_Type` class. The file should contain a header and have the following format:
```
Id,Cover_Type
4000000,2
4000001,1
4000001,3
etc.
```

## Dataset 
- train.csv - the training data with the target `Cover_Type` column
- test.csv - the test set; you will be predicting the `Cover_Type` for each row in this file (the target integer class)
- sample_submission.csv - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        input/
            description.md (59 lines)
            sample_submission.csv (400001 lines)
            sample_submission.csv.zip (1.6 MB)
            test.csv (400001 lines)
            test.csv.zip (10.7 MB)
            train.csv (3600001 lines)
            train.csv.zip (97.9 MB)
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
        working/
            tabular-playground-series-dec-2021/
                description.md (59 lines)
                sample_submission.csv (400001 lines)
                ... and 5 other files
                tabular-playground-series-dec-2021/
```

-> data/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> data/tabular-playground-series-dec-2021/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/tabular-playground-series-dec-2021/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> data/test.csv has 400000 rows and 55 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 40 more columns

-> data/train.csv has 3600000 rows and 56 columns.
The columns are: Id, Elevation, Aspect, Slope, Horizontal_Distance_To_Hydrology, Vertical_Distance_To_Hydrology, Horizontal_Distance_To_Roadways, Hillshade_9am, Hillshade_Noon, Hillshade_3pm, Horizontal_Distance_To_Fire_Points, Wilderness_Area1, Wilderness_Area2, Wilderness_Area3, Wilderness_Area4... and 41 more columns

-> input/sample_submission.csv has 400000 rows and 2 columns.
The columns are: Id, Cover_Type

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from pathlib import Path
import gc

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)



## === cell 1
TRAIN_PATH = "../input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "../input/tabular-playground-series-dec-2021/test.csv"
SUB_PATH = "../input/tabular-playground-series-dec-2021/sample_submission.csv"

cache_dir = Path("../working")
cache_dir.mkdir(parents=True, exist_ok=True)

cols_cache = cache_dir / "tps_dec2021_cols.npy"
if cols_cache.exists():
    train_cols = np.load(cols_cache, allow_pickle=True).tolist()
else:
    train_cols = pd.read_csv(TRAIN_PATH, nrows=1).columns.tolist()
    np.save(cols_cache, np.array(train_cols, dtype=object), allow_pickle=True)

feature_cols = [c for c in train_cols if c not in ("Id", "Cover_Type")]
usecols_train = ["Id"] + feature_cols + ["Cover_Type"]
usecols_test = ["Id"] + feature_cols

sample_submission = pd.read_csv(SUB_PATH)



## === cell 2
from catboost import CatBoostClassifier, Pool

n_threads = os.cpu_count() or 1
n_threads = max(1, min(n_threads, 8))

model_path = cache_dir / "catboost_model.cbm"
snapshot_file = str(cache_dir / "catboost_snapshot")

try:
    import pyarrow  # noqa: F401

    _CSV_ENGINE = "pyarrow"
except Exception:
    _CSV_ENGINE = "c"

dtype_train = {c: np.int32 for c in feature_cols}
dtype_train["Id"] = np.int32
dtype_train["Cover_Type"] = np.int32

dtype_test = {c: np.int32 for c in feature_cols}
dtype_test["Id"] = np.int32

_common_read_kwargs_train = dict(
    usecols=usecols_train,
    dtype=dtype_train,
    engine=_CSV_ENGINE,
)
_common_read_kwargs_test = dict(
    usecols=usecols_test,
    dtype=dtype_test,
    engine=_CSV_ENGINE,
)
if _CSV_ENGINE != "pyarrow":
    _common_read_kwargs_train["low_memory"] = False
    _common_read_kwargs_test["low_memory"] = False

X_train_cache = cache_dir / "X_train_float32.npy"
y_train_cache = cache_dir / "y_train_int32.npy"
X_test_cache = cache_dir / "X_test_float32.npy"

if X_train_cache.exists() and y_train_cache.exists() and X_test_cache.exists():
    X_train = np.load(X_train_cache, mmap_mode="r")
    y_train = np.load(y_train_cache, mmap_mode="r")
    X_test = np.load(X_test_cache, mmap_mode="r")
else:
    train_df = pd.read_csv(TRAIN_PATH, **_common_read_kwargs_train)
    test_df = pd.read_csv(TEST_PATH, **_common_read_kwargs_test)

    y_train = train_df["Cover_Type"].to_numpy(dtype=np.int32, copy=False)
    X_train = np.ascontiguousarray(
        train_df[feature_cols].to_numpy(copy=False), dtype=np.float32
    )
    X_test = np.ascontiguousarray(
        test_df[feature_cols].to_numpy(copy=False), dtype=np.float32
    )

    np.save(X_train_cache, X_train, allow_pickle=False)
    np.save(y_train_cache, y_train, allow_pickle=False)
    np.save(X_test_cache, X_test, allow_pickle=False)

    del train_df, test_df
    gc.collect()

train_pool = Pool(X_train, label=y_train)
test_pool = Pool(X_test)

del X_train, y_train
gc.collect()

clf_CatBoostClassifier = CatBoostClassifier(
    verbose=0,
    task_type="CPU",
    thread_count=n_threads,
    random_seed=0,
    allow_writing_files=True,  # needed for snapshot; does not change training semantics
    snapshot_file=snapshot_file,
    snapshot_interval=60,
    iterations=1000,
)

if model_path.exists():
    clf_CatBoostClassifier.load_model(str(model_path))
else:
    clf_CatBoostClassifier.fit(train_pool)
    clf_CatBoostClassifier.save_model(str(model_path))

pred = clf_CatBoostClassifier.predict(test_pool)

sample_submission.iloc[:, 1] = np.asarray(pred, dtype=np.int32).ravel()
sample_submission.to_csv("submission_CatBoostClassifier.csv", index=False)
sample_submission.head()



## === cell 3
sample_submission.head()
