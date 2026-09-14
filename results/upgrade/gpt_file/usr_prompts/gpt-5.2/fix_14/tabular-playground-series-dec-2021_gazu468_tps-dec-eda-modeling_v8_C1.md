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
from category_encoders.target_encoder import TargetEncoder

warnings.filterwarnings("ignore")

os.environ.setdefault("PYTHONHASHSEED", "2021")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

SEED = 2021
np.random.seed(SEED)

train_path = "../input/tabular-playground-series-dec-2021/train.csv"
test_path = "../input/tabular-playground-series-dec-2021/test.csv"

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()

feature_cols = [c for c in train_cols if c != "Cover_Type"]

int16_cols = [
    c
    for c in feature_cols
    if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
]

dtype_train = {"Id": "int32", "Cover_Type": "int8"}
dtype_test = {"Id": "int32"}
for c in feature_cols:
    if c in int16_cols:
        dtype_train[c] = "int16"
        dtype_test[c] = "int16"
    elif c == "Id":
        continue
    else:
        dtype_train[c] = "float32"
        dtype_test[c] = "float32"

usecols_train = ["Id", "Cover_Type"] + [c for c in feature_cols if c != "Id"]
usecols_test = ["Id"] + [c for c in test_cols if c != "Id"]

_read_kwargs_train = dict(
    dtype=dtype_train,
    usecols=usecols_train,
    low_memory=False,
)
_read_kwargs_test = dict(
    dtype=dtype_test,
    usecols=usecols_test,
    low_memory=False,
)

try:
    train = pd.read_csv(train_path, engine="pyarrow", **_read_kwargs_train)
    test = pd.read_csv(test_path, engine="pyarrow", **_read_kwargs_test)
except Exception:
    train = pd.read_csv(train_path, engine="c", **_read_kwargs_train)
    test = pd.read_csv(test_path, engine="c", **_read_kwargs_test)



## === cell 1
try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass



## === cell 2
train_y = train["Cover_Type"]
train_X = train.drop(columns=["Cover_Type", "Id"])
test_ids = test["Id"].to_numpy(copy=False)
test_X_full = test.drop(columns=["Id"])



## === cell 3
n = len(train_X)
rng = np.random.RandomState(SEED)
valid_mask = rng.randint(0, 100, size=n) < 22



## === cell 4
nums_cols = train_X.select_dtypes(
    include=[
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ]
).columns.tolist()
catgo_cols = [col for col in train_X.columns if col not in nums_cols]

d_test = test_X_full

if len(catgo_cols) > 0:
    enc = TargetEncoder(cols=catgo_cols)
    train_X_enc = enc.fit_transform(train_X, train_y)
    d_test = enc.transform(d_test)
else:
    train_X_enc = train_X



## === cell 5
del train, test, train_X, test_X_full



## === cell 6
from sklearn.preprocessing import StandardScaler

X_all = np.asarray(train_X_enc.to_numpy(dtype=np.float32, copy=False), order="C")
y_all = train_y.to_numpy(copy=False).astype(np.int32) - 1
X_test_full_np = np.asarray(d_test.to_numpy(dtype=np.float32, copy=False), order="C")

X_valid = X_all[valid_mask]
y_valid = y_all[valid_mask]
X_train = X_all[~valid_mask]
y_train = y_all[~valid_mask]

scaler = StandardScaler(copy=False)
scaler.fit(X_train)

train_X = scaler.transform(X_train)
test_X = scaler.transform(X_valid)
test = scaler.transform(X_test_full_np)

train_X = np.asarray(train_X, dtype=np.float32, order="C")
test_X = np.asarray(test_X, dtype=np.float32, order="C")
test = np.asarray(test, dtype=np.float32, order="C")

y_test = y_valid  # keep original variable naming
del (
    train_X_enc,
    d_test,
    X_all,
    X_test_full_np,
    X_train,
    X_valid,
    y_all,
    y_valid,
    valid_mask,
    train_y,
)



## === cell 7
import xgboost as xgb


class _XGBSoftmaxWrapper:
    """Minimal wrapper to preserve .predict API used below (returns class indices 0..6)."""

    def __init__(self, booster, num_class: int = 7, max_bin: int = 256):
        self.booster = booster
        self.num_class = num_class
        self.max_bin = max_bin

    def predict(self, X):
        try:
            dm = xgb.QuantileDMatrix(X, max_bin=self.max_bin)
        except Exception:
            dm = xgb.DMatrix(X)
        return self.booster.predict(dm).astype(np.int32)


def _make_matrix(X, y=None, max_bin=256):
    """
    Speed-critical: QuantileDMatrix accelerates histogram preprocessing and reduces memory for hist.
    Semantics are consistent for histogram tree methods; max_bin default (256) matches XGBoost default.
    """
    try:
        dm = xgb.QuantileDMatrix(X, label=y, max_bin=max_bin)
    except Exception:
        dm = xgb.DMatrix(X, label=y)
    return dm


def _build_xgb_and_fit(train_X, y_train, test_X, y_test, base_params):
    max_bin = int(base_params.get("max_bin", 256))
    dtrain = _make_matrix(train_X, y_train, max_bin=max_bin)
    dvalid = _make_matrix(test_X, y_test, max_bin=max_bin)

    params = dict(base_params)
    num_boost_round = int(params.pop("n_estimators"))
    evals = [(dvalid, "validation")]
    params.setdefault("verbosity", 0)
    params.setdefault("max_bin", max_bin)

    params.update({"tree_method": "hist", "predictor": "cpu_predictor"})

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=200,
        verbose_eval=False,
    )
    return _XGBSoftmaxWrapper(
        booster, num_class=params["num_class"], max_bin=params["max_bin"]
    )


params = {
    "objective": "multi:softmax",
    "num_class": 7,
    "eval_metric": "merror",
    "booster": "gbtree",
    "gamma": 0.75,
    "max_depth": 7,
    "alpha": 10,
    "learning_rate": 0.007,
    "n_estimators": 2000,
    "random_state": SEED,
    "nthread": 4,
    "max_bin": 256,
}

xgb_model = _build_xgb_and_fit(train_X, y_train, test_X, y_test, params)



## === cell 8
preds_valid = xgb_model.predict(test_X).astype("int32")
acc = accuracy_score(y_test, preds_valid)
print("accuracy score:", acc)

sub = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/sample_submission.csv",
    usecols=["Id", "Cover_Type"],
)

pred_test = xgb_model.predict(test).astype("int32") + 1
sub["Cover_Type"] = pred_test

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("submission.csv written with shape:", sub.shape)
