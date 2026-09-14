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

category_encoders==2.7.0
geopandas==0.14.4
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
seaborn==0.12.2
sklearn-pandas==2.2.0
xgboost==2.0.3

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
import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

np.random.seed(2021)

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

_CANDIDATE_DATA_DIRS = [
    "../input/tabular-playground-series-dec-2021",
    "/kaggle/input/tabular-playground-series-dec-2021",
]
DATA_DIR = None
for _d in _CANDIDATE_DATA_DIRS:
    if os.path.exists(os.path.join(_d, "train.csv")) and os.path.exists(
        os.path.join(_d, "test.csv")
    ):
        DATA_DIR = _d
        break
if DATA_DIR is None:
    DATA_DIR = "../input/tabular-playground-series-dec-2021"

train_path = f"{DATA_DIR}/train.csv"
test_path = f"{DATA_DIR}/test.csv"

_train_head = pd.read_csv(train_path, nrows=1)
_test_head = pd.read_csv(test_path, nrows=1)
train_cols = _train_head.columns.tolist()
test_cols = _test_head.columns.tolist()
del _train_head, _test_head

dtype_map = {c: np.int32 for c in train_cols if c not in ("Id", "Cover_Type")}
dtype_map["Id"] = np.int32
dtype_map["Cover_Type"] = np.int32

test_dtype_map = {c: np.int32 for c in test_cols if c != "Id"}
test_dtype_map["Id"] = np.int32

_read_engine = "pyarrow"
try:
    import pyarrow  # noqa: F401
except Exception:
    _read_engine = "c"

_read_csv_kwargs = dict(
    dtype=dtype_map,
    engine=_read_engine,
    usecols=train_cols,
)
_read_csv_kwargs_test = dict(
    dtype=test_dtype_map,
    engine=_read_engine,
    usecols=test_cols,
)
if _read_engine != "pyarrow":
    _read_csv_kwargs["low_memory"] = False
    _read_csv_kwargs_test["low_memory"] = False
    _read_csv_kwargs["memory_map"] = True
    _read_csv_kwargs_test["memory_map"] = True

train = pd.read_csv(train_path, **_read_csv_kwargs)
test = pd.read_csv(test_path, **_read_csv_kwargs_test)



## === cell 1
_ = train.shape, test.shape



## === cell 2
train_y = train["Cover_Type"].astype(np.int32) - 1
train_X = train.drop(["Id", "Cover_Type"], axis=1)



## === cell 3
X_train = train_X
y_train = train_y

_valid_n = min(200_000, len(X_train))
y_valid = y_train.iloc[:_valid_n]



## === cell 4
del train, train_X, train_y



## === cell 5
dtypes = X_train.dtypes
nums_cols = dtypes.index[
    dtypes.isin([np.dtype("float16"), np.dtype("float32"), np.dtype("float64")])
].tolist()
catgo_cols = [c for c in X_train.columns if c not in nums_cols]



## === cell 6
test_ids = test["Id"].to_numpy(copy=False)
test = test.drop("Id", axis=1)
test = test.loc[:, X_train.columns]



## === cell 7
if len(catgo_cols) > 0:
    cat_dtype_map = {c: "category" for c in catgo_cols}
    X_train = X_train.astype(cat_dtype_map)
    test = test.astype(cat_dtype_map)

d_test = test



## === cell 8
del test



## === cell 9
X_train_np = X_train.to_numpy(copy=False)
d_test_np = d_test.to_numpy(copy=False)

use_pandas_for_xgb = len(catgo_cols) > 0

if not use_pandas_for_xgb:
    X_train_np = np.asarray(X_train_np, dtype=np.float32, order="C")
    d_test_np = np.asarray(d_test_np, dtype=np.float32, order="C")

    mean_ = X_train_np.mean(axis=0, dtype=np.float64)
    var_ = X_train_np.var(axis=0, dtype=np.float64)
    scale_ = np.sqrt(var_, dtype=np.float64)
    scale_[scale_ == 0.0] = 1.0
    mean32 = mean_.astype(np.float32, copy=False)
    scale32 = scale_.astype(np.float32, copy=False)

    train_X = np.empty_like(X_train_np, dtype=np.float32, order="C")
    test = np.empty_like(d_test_np, dtype=np.float32, order="C")

    np.subtract(X_train_np, mean32, out=train_X)
    np.divide(train_X, scale32, out=train_X)

    np.subtract(d_test_np, mean32, out=test)
    np.divide(test, scale32, out=test)

    valid_X = train_X[:_valid_n]
    y_valid = y_valid.to_numpy(dtype=np.int32, copy=False)

    del mean_, var_, scale_, mean32, scale32
else:
    train_X = X_train
    valid_X = X_train.iloc[:_valid_n]
    test = d_test
    y_valid = y_valid.to_numpy(dtype=np.int32, copy=False)



## === cell 10
del X_train, d_test, X_train_np, d_test_np



## === cell 11
y_train = y_train.to_numpy(dtype=np.int32, copy=False)



## === cell 12
import xgboost as xgb

num_class = int(np.max(y_train) + 1)

params = {
    "objective": "multi:softmax",
    "num_class": num_class,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
    "booster": "gbtree",
    "gamma": 0.75,
    "max_depth": 7,
    "alpha": 10,
    "learning_rate": 0.007,
    "seed": 2021,
    "nthread": 4,
    "verbosity": 1,
    "cache_opt": 1,
    "max_bin": 256,
}

_missing = np.nan

if len(catgo_cols) > 0:
    params["enable_categorical"] = True
    try:
        dtrain = xgb.QuantileDMatrix(
            train_X, label=y_train, enable_categorical=True, missing=_missing
        )
    except Exception:
        dtrain = xgb.DMatrix(
            train_X, label=y_train, enable_categorical=True, missing=_missing
        )
    try:
        dvalid = xgb.QuantileDMatrix(
            valid_X,
            label=y_valid,
            ref=dtrain,
            enable_categorical=True,
            missing=_missing,
        )
    except Exception:
        dvalid = xgb.DMatrix(
            valid_X, label=y_valid, enable_categorical=True, missing=_missing
        )
else:
    try:
        dtrain = xgb.QuantileDMatrix(train_X, label=y_train, missing=_missing)
    except Exception:
        dtrain = xgb.DMatrix(train_X, label=y_train, missing=_missing)
    try:
        dvalid = xgb.QuantileDMatrix(
            valid_X, label=y_valid, ref=dtrain, missing=_missing
        )
    except Exception:
        dvalid = xgb.DMatrix(valid_X, label=y_valid, missing=_missing)

del train_X, valid_X

xgb_model = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=2000,
    evals=[(dvalid, "validation")],
    verbose_eval=200,
)



## === cell 13
preds_valid = (xgb_model.predict(dvalid).astype(np.int32) + 1).astype("int")
acc = accuracy_score((y_valid + 1).astype(np.int32), preds_valid)
print("accuracy score:", acc)



## === cell 14
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

if "Id" in sub.columns and len(test_ids) == len(sub):
    sub_ids = sub["Id"].to_numpy(copy=False)
    if not np.array_equal(sub_ids, test_ids):
        sub = sub.drop(columns=["Cover_Type"], errors="ignore")
        sub["Id"] = test_ids

if len(catgo_cols) > 0:
    try:
        dtest = xgb.QuantileDMatrix(
            test, ref=dtrain, enable_categorical=True, missing=_missing
        )
    except Exception:
        dtest = xgb.DMatrix(test, enable_categorical=True, missing=_missing)
else:
    try:
        dtest = xgb.QuantileDMatrix(test, ref=dtrain, missing=_missing)
    except Exception:
        dtest = xgb.DMatrix(test, missing=_missing)

del test

sub["Cover_Type"] = (xgb_model.predict(dtest).astype(np.int32) + 1).astype("int")
sub["Cover_Type"] = sub["Cover_Type"].clip(1, num_class).astype(np.int32)

sub.to_csv("submission.csv", index=False)
sub.head()
