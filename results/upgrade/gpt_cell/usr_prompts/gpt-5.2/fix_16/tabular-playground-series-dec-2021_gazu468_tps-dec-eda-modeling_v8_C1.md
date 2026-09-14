# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

RANDOM_STATE = 2021
np.random.seed(RANDOM_STATE)



## === cell 1
train_path = "../input/tabular-playground-series-dec-2021/train.csv"
test_path = "../input/tabular-playground-series-dec-2021/test.csv"

_sample = pd.read_csv(train_path, nrows=200)
dtype_map = {}
for c in _sample.columns:
    if c == "Id":
        dtype_map[c] = np.int32
    elif c == "Cover_Type":
        dtype_map[c] = np.int8
    else:
        if pd.api.types.is_integer_dtype(_sample[c].dtype):
            dtype_map[c] = np.int32
        else:
            dtype_map[c] = np.float32

read_kwargs = dict(dtype=dtype_map)
read_kwargs_test = dict(dtype={k: v for k, v in dtype_map.items() if k != "Cover_Type"})

try:
    train = pd.read_csv(train_path, engine="pyarrow", **read_kwargs)
    test = pd.read_csv(test_path, engine="pyarrow", **read_kwargs_test)
except Exception:
    train = pd.read_csv(train_path, **read_kwargs)
    test = pd.read_csv(test_path, **read_kwargs_test)



## === cell 2
_ = train.shape
_ = test.shape



## === cell 3
pass



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



## === cell 13
_ = train.isnull().sum().sum()



## === cell 14
train_X = train.drop("Cover_Type", axis=1)
train_y = train["Cover_Type"]



## === cell 15
_ = train_y.iloc[:5].to_list()



## === cell 16
_ = train_X.shape



## === cell 17
from sklearn.model_selection import StratifiedShuffleSplit

y_all = train_y.to_numpy(dtype=np.int32, copy=False)

counts = np.bincount(y_all)
_min_class_count = int(counts[counts > 0].min()) if counts.size else 0
use_strat = _min_class_count >= 2

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.22, random_state=RANDOM_STATE)
if use_strat:
    train_idx, test_idx = next(sss.split(train_X, y_all))
else:
    rng = np.random.RandomState(RANDOM_STATE)
    idx = np.arange(train_X.shape[0])
    rng.shuffle(idx)
    n_test = int(round(train_X.shape[0] * 0.22))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]



## === cell 18
_ = test_idx.shape



## === cell 19
nums_cols = [
    col
    for col in train_X.columns
    if train_X[col].dtype in ["float16", "float32", "float64"]
]
catgo_cols = [
    col
    for col in train_X.columns
    if train_X[col].dtype not in ["float16", "float32", "float64"]
]



## === cell 20
d_test = test
catgo_cols = []



## === cell 21
X_all = np.asarray(train_X, dtype=np.float32, order="C")
X_test = X_all[test_idx]
X_train = X_all[train_idx]

y_test = y_all[test_idx]
y_train = y_all[train_idx]

test = np.asarray(d_test, dtype=np.float32, order="C")



## === cell 22
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(copy=False)
scaler.fit(X_train)

scaler.transform(X_train)
scaler.transform(X_test)
scaler.transform(test)



## === cell 23
_ = X_train.shape



## === cell 24
X_train = np.ascontiguousarray(X_train, dtype=np.float32)
X_test = np.ascontiguousarray(X_test, dtype=np.float32)
test = np.ascontiguousarray(test, dtype=np.float32)

y_train = np.asarray(y_train, dtype=np.int32, order="C")
y_test = np.asarray(y_test, dtype=np.int32, order="C")



## === cell 25
y_train = (y_train - 1).astype(np.int32, copy=False)
y_test = (y_test - 1).astype(np.int32, copy=False)



## === cell 26
import xgboost as xgb
import tempfile

use_gpu = False
try:
    use_gpu = getattr(xgb, "cuda", None) is not None and xgb.cuda.get_device_count() > 0
except Exception:
    use_gpu = False

params = {
    "objective": "multi:softmax",
    "num_class": 7,
    "tree_method": "gpu_hist" if use_gpu else "hist",
    "eval_metric": "mlogloss",
    "booster": "gbtree",
    "gamma": 0.75,
    "max_depth": 7,
    "alpha": 10,
    "learning_rate": 0.007,
    "predictor": "gpu_predictor" if use_gpu else "cpu_predictor",
    "random_state": RANDOM_STATE,
    "seed": RANDOM_STATE,
    "nthread": max(1, (os.cpu_count() or 2) - 1),
    "max_bin": 256,
    "verbosity": 1,
}

n_estimators = 2000
early_stopping_rounds = 200

cache_dir = os.path.join(tempfile.gettempdir(), "xgb_cache_tps_dec2021")
os.makedirs(cache_dir, exist_ok=True)

DMatrixCls = getattr(xgb, "QuantileDMatrix", xgb.DMatrix)

dtrain = DMatrixCls(
    X_train,
    label=y_train,
    nthread=params["nthread"],
    cache_prefix=os.path.join(cache_dir, "dtrain"),
)
dvalid = DMatrixCls(
    X_test,
    label=y_test,
    nthread=params["nthread"],
    cache_prefix=os.path.join(cache_dir, "dvalid"),
)

evals = [(dvalid, "valid")]

booster = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=n_estimators,
    evals=evals,
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=100,
)



## --- ERROR in cell 26, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/489823906.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     37[0m [0mDMatrixCls[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mxgb[0m[0;34m,[0m [0;34m"QuantileDMatrix"[0m[0;34m,[0m [0mxgb[0m[0;34m.[0m[0mDMatrix[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     38[0m [0;34m[0m[0m
[0;32m---> 39[0;31m dtrain = DMatrixCls(
[0m[1;32m     40[0m     [0mX_train[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     41[0m     [0mlabel[0m[0;34m=[0m[0my_train[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36minner_f[0;34m(*args, **kwargs)[0m
[1;32m    728[0m             [0;32mfor[0m [0mk[0m[0;34m,[0m [0marg[0m [0;32min[0m [0mzip[0m[0;34m([0m[0msig[0m[0;34m.[0m[0mparameters[0m[0;34m,[0m [0margs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    729[0m                 [0mkwargs[0m[0;34m[[0m[0mk[0m[0;34m][0m [0;34m=[0m [0marg[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 730[0;31m             [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    731[0m [0;34m[0m[0m
[1;32m    732[0m         [0;32mreturn[0m [0minner_f[0m[0;34m[0m[0;34m[0m[0m

[0;31mTypeError[0m: QuantileDMatrix.__init__() got an unexpected keyword argument 'cache_prefix'

## === cell 27
from sklearn.metrics import accuracy_score

preds_valid = booster.predict(dvalid).astype(np.int32) + 1
acc = accuracy_score((y_test + 1).astype(np.int32), preds_valid)
print("accuracy score:", acc)
