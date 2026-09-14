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

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)



## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier



## === cell 2
import warnings

warnings.filterwarnings("ignore")



## === cell 3
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

TRAIN_PATH = "../input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "../input/tabular-playground-series-dec-2021/test.csv"

_train_head = pd.read_csv(TRAIN_PATH, nrows=0)
_all_cols = _train_head.columns.tolist()
_target_col = "Cover_Type"
_id_col = "Id"
_feature_cols = [c for c in _all_cols if c not in (_id_col, _target_col)]

dtype_map_train = {_id_col: np.int32, _target_col: np.int8}
dtype_map_train.update({c: np.float32 for c in _feature_cols})
dtype_map_test = {_id_col: np.int32}
dtype_map_test.update({c: np.float32 for c in _feature_cols})

try:
    train = pd.read_csv(TRAIN_PATH, dtype=dtype_map_train, engine="pyarrow")
    test = pd.read_csv(TEST_PATH, dtype=dtype_map_test, engine="pyarrow")
except Exception:
    train = pd.read_csv(TRAIN_PATH, dtype=dtype_map_train)
    test = pd.read_csv(TEST_PATH, dtype=dtype_map_test)

x_data = train[_feature_cols].copy()
x_test = test[_feature_cols].copy()
y_data = train[_target_col].copy()




## === cell 4
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    arr = df.to_numpy(dtype=np.float32, copy=False)
    n_rows, n_cols = arr.shape

    s = arr.sum(axis=1, dtype=np.float64)
    ss = np.square(arr, dtype=np.float32).sum(axis=1, dtype=np.float64)
    mx = arr.max(axis=1)
    mn = arr.min(axis=1)

    mean = (s / n_cols).astype(np.float32)

    if n_cols > 1:
        mean64 = s / n_cols
        var = (ss - n_cols * (mean64 * mean64)) / (n_cols - 1)
        var = np.maximum(var, 0.0)
        std = np.sqrt(var, dtype=np.float64).astype(np.float32)
    else:
        std = np.zeros(n_rows, dtype=np.float32)

    if n_cols >= 3:
        denom = arr[:, 2] + np.float32(1.0)
        f1 = (arr[:, 0] * arr[:, 1]) / denom
    else:
        f1 = np.zeros(n_rows, dtype=np.float32)

    def _sanitize(a: np.ndarray) -> np.ndarray:
        a = a.astype(np.float32, copy=False)
        a = np.where(np.isfinite(a), a, np.float32(0.0))
        np.clip(a, np.float32(-1e9), np.float32(1e9), out=a)
        return a

    df["mean"] = _sanitize(mean)
    df["std"] = _sanitize(std)
    df["max"] = _sanitize(mx)
    df["min"] = _sanitize(mn)
    df["f1"] = _sanitize(f1)
    return df


x_data = add_features(x_data)
x_test = add_features(x_test)



## === cell 5
y_data_int = pd.to_numeric(y_data, errors="coerce")

y_data_int = y_data_int.fillna(1).astype(np.int32)
y_data_int = np.clip(y_data_int, 1, 7).astype(np.int32)

y_train_mapped = y_data_int - 1  # now guaranteed in 0..6

try:
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_train_mapped, test_size=0.2, random_state=42, stratify=y_train_mapped
    )
except ValueError:
    x_train, x_val, y_train, y_val = train_test_split(
        x_data, y_train_mapped, test_size=0.2, random_state=42, shuffle=True
    )



## === cell 6
import xgboost as xgb


def make_model():
    params = dict(
        objective="multi:softmax",
        num_class=7,
        random_state=42,
        n_estimators=500,
        learning_rate=0.1,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        eval_metric="mlogloss",
    )

    gpu_params = dict(tree_method="gpu_hist", predictor="gpu_predictor")
    cpu_params = dict(tree_method="hist")

    try:
        return XGBClassifier(**gpu_params, **params)
    except Exception:
        return XGBClassifier(**cpu_params, **params)


def fit_with_fallback(model, X, y, xgb_model=None):
    try:
        model.fit(X, y, xgb_model=xgb_model)
        return model
    except Exception:
        params = model.get_params()
        params.pop("predictor", None)
        params["tree_method"] = "hist"
        m2 = XGBClassifier(**params)
        m2.fit(X, y, xgb_model=xgb_model)
        return m2


Xtr = np.ascontiguousarray(x_train.to_numpy(dtype=np.float32, copy=False))
Xva = np.ascontiguousarray(x_val.to_numpy(dtype=np.float32, copy=False))
ytr = (
    y_train.to_numpy(dtype=np.int32, copy=False)
    if hasattr(y_train, "to_numpy")
    else np.asarray(y_train, dtype=np.int32)
)
yva = (
    y_val.to_numpy(dtype=np.int32, copy=False)
    if hasattr(y_val, "to_numpy")
    else np.asarray(y_val, dtype=np.int32)
)

model = make_model()
model = fit_with_fallback(model, Xtr, ytr)
y_pred = model.predict(Xva)

acc = accuracy_score(yva, y_pred)
print(f"validation acc {acc:.6f}")



## === cell 7
Xfull = np.ascontiguousarray(x_data.to_numpy(dtype=np.float32, copy=False))
yfull = y_train_mapped.to_numpy(dtype=np.int32, copy=False)
Xtest = np.ascontiguousarray(x_test.to_numpy(dtype=np.float32, copy=False))

booster = None
try:
    booster = model.get_booster()
except Exception:
    booster = None

model_final = make_model()
model_final = fit_with_fallback(model_final, Xfull, yfull, xgb_model=booster)
y_test_mapped = model_final.predict(Xtest).astype(np.int32)

y_test = y_test_mapped + 1  # map back to 1..7

submission = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/sample_submission.csv"
)
if len(submission) != len(test):
    submission = pd.DataFrame({_id_col: test[_id_col].to_numpy()})
submission["Cover_Type"] = y_test.astype(np.int32)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv")
