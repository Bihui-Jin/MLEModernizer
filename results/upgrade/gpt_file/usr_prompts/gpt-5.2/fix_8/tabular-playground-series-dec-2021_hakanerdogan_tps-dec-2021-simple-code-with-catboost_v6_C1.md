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
import os
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42)



## === cell 1
BASE_1 = "/kaggle/input/tabular-playground-series-dec-2021"
BASE_2 = "/kaggle/input"  # files also appear directly under /kaggle/input


def _pick_path(rel_name: str) -> str:
    p1 = os.path.join(BASE_1, rel_name)
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(BASE_2, rel_name)
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"Could not find {rel_name} in {BASE_1} or {BASE_2}")


train_path = _pick_path("train.csv")
test_path = _pick_path("test.csv")
sub_path = _pick_path("sample_submission.csv")

target_col = "Cover_Type"
id_col = "Id"

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
feature_cols = [c for c in train_cols if c not in (id_col, target_col)]

binary_cols = [
    c
    for c in feature_cols
    if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
]
num_cols = [c for c in feature_cols if c not in set(binary_cols)]

dtype_train = {c: np.int32 for c in num_cols}
dtype_train.update({c: np.uint8 for c in binary_cols})
dtype_train[id_col] = np.int32
dtype_train[target_col] = np.int8  # labels 1..7 in raw file

dtype_test = {c: np.int32 for c in num_cols}
dtype_test.update({c: np.uint8 for c in binary_cols})
dtype_test[id_col] = np.int32

_read_csv_kwargs = {}
_use_pyarrow = False
try:
    import pyarrow  # noqa: F401

    _use_pyarrow = True
except Exception:
    _use_pyarrow = False

if _use_pyarrow:
    _read_csv_kwargs["engine"] = "pyarrow"
else:
    _read_csv_kwargs["low_memory"] = False

train = pd.read_csv(
    train_path,
    usecols=[target_col] + feature_cols,  # Id unused for training
    dtype={**{k: v for k, v in dtype_train.items() if k != id_col}},
    **_read_csv_kwargs,
)
test = pd.read_csv(
    test_path,
    usecols=feature_cols,  # submission Id comes from sample_submission
    dtype={k: v for k, v in dtype_test.items() if k != id_col},
    **_read_csv_kwargs,
)
sample_submission = pd.read_csv(
    sub_path,
    usecols=[id_col, target_col],
    dtype={id_col: np.int32, target_col: np.int32},
    **_read_csv_kwargs,
)



## === cell 2
y_train = train[target_col].to_numpy(dtype=np.int32, copy=False)
y_train = y_train - 1  # 0..6

X_train = train[feature_cols].to_numpy(dtype=np.float32, copy=False)
if not X_train.flags["C_CONTIGUOUS"]:
    X_train = np.ascontiguousarray(X_train)

X_test = test[feature_cols].to_numpy(dtype=np.float32, copy=False)
if not X_test.flags["C_CONTIGUOUS"]:
    X_test = np.ascontiguousarray(X_test)

del train, test



## === cell 3
from catboost import CatBoostClassifier, Pool

train_pool = Pool(X_train, label=y_train)
test_pool = Pool(X_test)

clf_CatBoostClassifier = CatBoostClassifier(
    verbose=200,
    task_type="CPU",
    random_seed=42,
    thread_count=-1,
    allow_writing_files=False,
    loss_function="MultiClass",
    eval_metric="Accuracy",
    grow_policy="Lossguide",
    boosting_type="Plain",
    one_hot_max_size=2,
    iterations=4000,
    od_type="Iter",
    od_wait=200,
)

clf_CatBoostClassifier.fit(train_pool)

pred = clf_CatBoostClassifier.predict(test_pool, prediction_type="Class").reshape(-1)

pred = pred.astype(np.int32, copy=False) + 1
pred = np.clip(pred, 1, 7).astype(np.int32, copy=False)

submission = sample_submission.copy()
submission[target_col] = pred
submission.to_csv("submission_CatBoostClassifier.csv", index=False)

submission.head()



## === cell 4
pass


## === cell 5
pass


## === cell 6
pass


## === cell 7
pass


## === cell 8
pass


## === cell 9
pass


## === cell 10
pass


## === cell 11
pass


## === cell 12
pass
