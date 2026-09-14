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
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from sklearn.metrics import roc_auc_score

np.random.seed(42)

try:
    from sklearnex import patch_sklearn  # type: ignore

    patch_sklearn()
except Exception:
    pass



## === cell 1
train_path = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
test_path = "/kaggle/input/tabular-playground-series-may-2022/test.csv"

float_feats = [f"f_{i:02d}" for i in range(31) if i != 7 and i != 27]
dtypes_train = {"id": np.int32, "target": np.int8, "f_07": np.int16, "f_27": "string"}
dtypes_train.update({c: np.float32 for c in float_feats})

dtypes_test = {"id": np.int32, "f_07": np.int16, "f_27": "string"}
dtypes_test.update({c: np.float32 for c in float_feats})

train = pd.read_csv(train_path, dtype=dtypes_train)
test = pd.read_csv(test_path, dtype=dtypes_test)

_ = train.shape, test.shape




## === cell 2
def check(df):
    col_list = df.columns.values
    rows = []
    for col in col_list:
        tmp = (
            col,
            df[col].dtype,
            df[col].isnull().sum(),
            df[col].count(),
            df[col].nunique(),
        )
        rows.append(tmp)
    df_out = pd.DataFrame(rows, columns=["feature", "dtype", "nan", "count", "nunique"])
    return df_out




## === cell 3
pass




## === cell 4
def color_negative_red(val):
    color = "red" if val < 0 else "black"
    return "color: %s" % color




## === cell 5
pass



## === cell 6
_ = test.shape



## === cell 7
pass



## === cell 8
pass



## === cell 9
_ = train["target"].value_counts()



## === cell 10
_ = train["target"].describe()



## === cell 11
pass



## === cell 12
_ = train["f_27"].value_counts().head()



## === cell 13
_ = test["f_27"].value_counts().head()



## === cell 14
from collections import OrderedDict


def encord(s: str) -> str:
    d = OrderedDict()
    for ch in s:
        d[ch] = d.get(ch, 0) + 1
    parts = []
    for k, v in d.items():
        parts.append(k)
        parts.append(str(v))
    return "".join(parts)




## === cell 15
def _encord_fast(u: str) -> str:
    arr = np.frombuffer(u.encode("utf-8"), dtype=np.uint8)
    uniq, cnt = np.unique(arr, return_counts=True)  # sorted by byte value
    order = np.unique(arr, return_index=True)[1]
    order = np.argsort(order)
    uniq = uniq[order]
    cnt = cnt[order]
    chars = [bytes([c]).decode("utf-8") for c in uniq.tolist()]
    return "".join(ch + str(int(n)) for ch, n in zip(chars, cnt.tolist()))


def _encord_map_for_values(values: np.ndarray) -> dict:
    out = {}
    for u in values:
        out[u] = _encord_fast(str(u))
    return out


f27_train = train["f_27"]
f27_test = test["f_27"]

uniq_all = pd.unique(pd.concat([f27_train, f27_test], ignore_index=True).dropna())
_enc_map = _encord_map_for_values(uniq_all)

train["f_27_en"] = f27_train.map(_enc_map)
test["f_27_ent"] = f27_test.map(_enc_map)



## === cell 16
label_f27 = LabelEncoder()
train_f27_str = train["f_27"].astype(str).to_numpy(copy=False)
test_f27_str = test["f_27"].astype(str).to_numpy(copy=False)
all_f27 = np.concatenate([train_f27_str, test_f27_str])
label_f27.fit(all_f27)
train["en_27"] = label_f27.transform(train_f27_str)

label_f27_en = LabelEncoder()
train_f27_en_str = train["f_27_en"].astype(str).to_numpy(copy=False)
test_f27_ent_str = test["f_27_ent"].astype(str).to_numpy(copy=False)
all_f27_en = np.concatenate([train_f27_en_str, test_f27_ent_str])
label_f27_en.fit(all_f27_en)

train["f_27_enc"] = label_f27_en.transform(train_f27_en_str)
test["f_27_enc"] = label_f27_en.transform(test_f27_ent_str)

_ = (
    train["en_27"].head(1).tolist(),
    train["f_27_enc"].head(1).tolist(),
    test["f_27_enc"].head(1).tolist(),
)



## === cell 17
_ = train.shape



## === cell 18
_ = test.shape



## === cell 19
pass



## === cell 20
pass



## === cell 21
pass



## === cell 22
pass



## === cell 23
_ = (train.shape, test.shape)



## === cell 24
X = train.drop(["id", "target", "f_27", "en_27", "f_27_en"], axis=1)
y = train["target"]
X_test = test.drop(["id", "f_27", "f_27_ent"], axis=1)

if "f_07" in X.columns:
    X["f_07"] = X["f_07"].astype("category")
    X_test["f_07"] = X_test["f_07"].astype("category")

del train
del test



## === cell 25
params = {
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    "n_estimators": 10000,
    "colsample_bytree": 0.5,
    "subsample": 0.5,
    "learning_rate": 0.02,
    "max_depth": 6,
    "enable_categorical": True,
}



## === cell 26
splits = 5
seed = 42
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

preds = []
scores = []

y_np = y.to_numpy(dtype=np.int8, copy=False)
X_idx = np.arange(len(y_np), dtype=np.int32)

try:
    has_cuda = bool(xgb.core._has_cuda_support())
except Exception:
    has_cuda = False

train_params = {
    "booster": "gbtree",
    "eval_metric": "auc",
    "colsample_bytree": params["colsample_bytree"],
    "subsample": params["subsample"],
    "learning_rate": params["learning_rate"],
    "max_depth": params["max_depth"],
    "enable_categorical": params["enable_categorical"],
    "seed": seed,
    "verbosity": 0,
    "num_parallel_tree": 1,
}

if has_cuda:
    train_params["tree_method"] = "gpu_hist"
    train_params["predictor"] = "gpu_predictor"
else:
    train_params["tree_method"] = "hist"
    train_params["predictor"] = "cpu_predictor"
    train_params["nthread"] = os.cpu_count() or 1

num_boost_round = params["n_estimators"]
feature_names = list(X.columns)

use_quantile = hasattr(xgb, "QuantileDMatrix")

if "f_07" in X.columns:
    X_np = X.copy()
    X_test_np = X_test.copy()
    X_np["f_07"] = X_np["f_07"].cat.codes.astype(np.int16)
    X_test_np["f_07"] = X_test_np["f_07"].cat.codes.astype(np.int16)
else:
    X_np = X
    X_test_np = X_test

X_mat = np.ascontiguousarray(X_np.to_numpy())
X_test_mat = np.ascontiguousarray(X_test_np.to_numpy())

del X_np, X_test_np
del X, X_test  # free RAM early


def _make_dmatrix(data, label=None):
    if use_quantile:
        return xgb.QuantileDMatrix(data, label=label, feature_names=feature_names)
    return xgb.DMatrix(data, label=label, feature_names=feature_names)


dall = _make_dmatrix(X_mat, label=y_np)
dtest = _make_dmatrix(X_test_mat, label=None)

for fold, (idx_train, idx_valid) in enumerate(skf.split(X_idx, y_np)):
    dtrain = dall.slice(idx_train)
    dvalid = dall.slice(idx_valid)

    booster = xgb.train(
        params=train_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    best_end = booster.best_iteration + 1

    pred_valid = booster.predict(dvalid, iteration_range=(0, best_end))
    score = roc_auc_score(y_np[idx_valid], pred_valid)
    scores.append(score)

    test_preds = booster.predict(dtest, iteration_range=(0, best_end))
    preds.append(test_preds)

    print("fold : ", fold, "score : ", score)



## --- ERROR in cell 26, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mXGBoostError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1855300328.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     73[0m [0;34m[0m[0m
[1;32m     74[0m [0;32mfor[0m [0mfold[0m[0;34m,[0m [0;34m([0m[0midx_train[0m[0;34m,[0m [0midx_valid[0m[0;34m)[0m [0;32min[0m [0menumerate[0m[0;34m([0m[0mskf[0m[0;34m.[0m[0msplit[0m[0;34m([0m[0mX_idx[0m[0;34m,[0m [0my_np[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 75[0;31m     [0mdtrain[0m [0;34m=[0m [0mdall[0m[0;34m.[0m[0mslice[0m[0;34m([0m[0midx_train[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     76[0m     [0mdvalid[0m [0;34m=[0m [0mdall[0m[0;34m.[0m[0mslice[0m[0;34m([0m[0midx_valid[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     77[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36mslice[0;34m(self, rindex, allow_groups)[0m
[1;32m   1256[0m         [0mres[0m[0;34m.[0m[0mhandle[0m [0;34m=[0m [0mctypes[0m[0;34m.[0m[0mc_void_p[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1257[0m         [0mrindex[0m [0;34m=[0m [0m_maybe_np_slice[0m[0;34m([0m[0mrindex[0m[0;34m,[0m [0mdtype[0m[0;34m=[0m[0mnp[0m[0;34m.[0m[0mint32[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1258[0;31m         _check_call(
[0m[1;32m   1259[0m             _LIB.XGDMatrixSliceDMatrixEx(
[1;32m   1260[0m                 [0mself[0m[0;34m.[0m[0mhandle[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/xgboost/core.py[0m in [0;36m_check_call[0;34m(ret)[0m
[1;32m    280[0m     """
[1;32m    281[0m     [0;32mif[0m [0mret[0m [0;34m!=[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 282[0;31m         [0;32mraise[0m [0mXGBoostError[0m[0;34m([0m[0mpy_str[0m[0;34m([0m[0m_LIB[0m[0;34m.[0m[0mXGBGetLastError[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    283[0m [0;34m[0m[0m
[1;32m    284[0m [0;34m[0m[0m

[0;31mXGBoostError[0m: [10:31:21] /workspace/src/data/iterative_dmatrix.h:88: Slicing DMatrix is not supported for Quantile DMatrix.
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3effba) [0x7fff83903fba]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3ff7ab) [0x7fff839137ab]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGDMatrixSliceDMatrixEx+0x146) [0x7fff83674206]
  [bt] (3) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (4) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (5) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]
  [bt] (6) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7ffff63bbc8e]
  [bt] (7) /usr/bin/python3(_PyObject_MakeTpCall+0x27c) [0x52f85c]
  [bt] (8) /usr/bin/python3(_PyEval_EvalFrameDefault+0x6bc) [0x53da0c]



## === cell 27
print(scores)
