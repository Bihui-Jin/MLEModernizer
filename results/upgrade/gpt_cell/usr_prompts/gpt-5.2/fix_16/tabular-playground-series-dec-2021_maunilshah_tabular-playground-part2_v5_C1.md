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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import gc
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

np.random.seed(42)

TRAIN_PATH = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

train_cols = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
test_cols = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()


def _dtype_for_col(c: str):
    if c == "Id":
        return np.int32
    if c == "Cover_Type":
        return np.uint8
    if c.startswith("Wilderness_Area") or c.startswith("Soil_Type"):
        return np.uint8
    return np.int32


train_dtypes = {c: _dtype_for_col(c) for c in train_cols}
test_dtypes = {c: _dtype_for_col(c) for c in test_cols}

train_df = pd.read_csv(
    TRAIN_PATH,
    dtype=train_dtypes,
    usecols=train_cols,
    engine="c",
    low_memory=False,
)
test_df = pd.read_csv(
    TEST_PATH,
    dtype=test_dtypes,
    usecols=test_cols,
    engine="c",
    low_memory=False,
)



## === cell 1
feature_cols = [c for c in train_df.columns if c not in ("Cover_Type", "Id")]

X_np = np.ascontiguousarray(
    train_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
)
Y_np = train_df["Cover_Type"].to_numpy(copy=False)

del train_df
gc.collect()

X_np.shape, Y_np.shape



## === cell 2
from sklearn.model_selection import train_test_split

VAL_SIZE = 20000  # keep identical

_class_counts = np.bincount(Y_np.astype(np.int64, copy=False))
_use_stratify = _class_counts.min() >= 2

x_train, x_val, y_train, y_val = train_test_split(
    X_np,
    Y_np,
    test_size=VAL_SIZE,
    random_state=42,
    stratify=Y_np if _use_stratify else None,
)

del X_np, Y_np
gc.collect()

x_train.shape, x_val.shape, y_train.shape, y_val.shape



## === cell 3
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_val_enc = le.transform(y_val)

test_id = test_df["Id"].to_numpy(copy=False)

test_np = np.ascontiguousarray(
    test_df[feature_cols].to_numpy(dtype=np.float32, copy=False)
)
del test_df
gc.collect()


def _make_qdmatrix_train(X, y, nthread=4):
    try:
        return xgb.QuantileDMatrix(X, label=y, nthread=nthread)
    except Exception:
        return xgb.DMatrix(X, label=y, nthread=nthread, enable_categorical=False)


def _make_qdmatrix_ref(X, y, ref, nthread=4):
    try:
        return xgb.QuantileDMatrix(X, label=y, nthread=nthread, ref=ref)
    except Exception:
        return xgb.DMatrix(X, label=y, nthread=nthread, enable_categorical=False)


dtrain = _make_qdmatrix_train(x_train, y_train_enc, nthread=4)
dval = _make_qdmatrix_ref(x_val, y_val_enc, ref=dtrain, nthread=4)
dtest = _make_qdmatrix_ref(test_np, None, ref=dtrain, nthread=4)

num_class = int(le.classes_.shape[0])
params = {
    "objective": "multi:softmax",  # direct class prediction
    "num_class": num_class,
    "eta": 0.01,
    "random_state": 42,
    "seed": 42,
    "nthread": 4,
    "eval_metric": "mlogloss",
}

EVAL_EVERY = 50

try:
    params_gpu = params | {
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "gpu_id": 0,
    }
    booster = xgb.train(
        params=params_gpu,
        dtrain=dtrain,
        num_boost_round=20000,
        evals=[(dval, "validation")],
        early_stopping_rounds=5,
        verbose_eval=EVAL_EVERY,
    )
except Exception:
    params_cpu = params | {"tree_method": "hist", "predictor": "auto"}
    booster = xgb.train(
        params=params_cpu,
        dtrain=dtrain,
        num_boost_round=20000,
        evals=[(dval, "validation")],
        early_stopping_rounds=5,
        verbose_eval=EVAL_EVERY,
    )

try:
    _pred_enc = booster.inplace_predict(test_np).astype(np.int64, copy=False)
except Exception:
    _pred_enc = booster.predict(dtest).astype(np.int64, copy=False)

y_predict_xgbc = le.inverse_transform(_pred_enc)

del (
    x_train,
    x_val,
    y_train,
    y_val,
    y_train_enc,
    y_val_enc,
    test_np,
    _pred_enc,
    dtrain,
    dval,
    dtest,
    booster,
)
gc.collect()



## === cell 4
result = pd.DataFrame(
    {
        "Id": test_id,
        "Cover_Type": y_predict_xgbc,
    }
)
result.head()



## === cell 5
result.shape



## === cell 6
result.to_csv("/kaggle/working/submission.csv", index=False)
print("Done")
