# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

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

# 5. Target score

0.93489

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import xgboost as xgb

os.environ["PYTHONHASHSEED"] = "42"
np.random.seed(42)

cpu_n = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", str(cpu_n))
os.environ.setdefault("MKL_NUM_THREADS", str(cpu_n))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_n))

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

TRAIN_PATH = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-may-2022/test.csv"

float_cols = [f"f_{i:02d}" for i in range(31) if i != 27]
dtypes_train = {c: "float32" for c in float_cols}
dtypes_train.update({"id": "int32", "target": "int8", "f_27": "string"})
dtypes_test = {c: "float32" for c in float_cols}
dtypes_test.update({"id": "int32", "f_27": "string"})

try:
    train = pd.read_csv(
        TRAIN_PATH, dtype=dtypes_train, low_memory=False, engine="pyarrow"
    )
    test = pd.read_csv(TEST_PATH, dtype=dtypes_test, low_memory=False, engine="pyarrow")
except Exception:
    train = pd.read_csv(TRAIN_PATH, dtype=dtypes_train, low_memory=False)
    test = pd.read_csv(TEST_PATH, dtype=dtypes_test, low_memory=False)



## === cell 1
_ = None



## === cell 2
_ = None



## === cell 3
_ = None



## === cell 4
_ = None



## === cell 5
_ = None



## === cell 6
_ = None



## === cell 7
_ = None



## === cell 8
_ = None



## === cell 9
_ = None



## === cell 10
from functools import lru_cache

_ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
_CHAR_TO_IDX = {c: i for i, c in enumerate(_ALLOWED_CHARS)}


@lru_cache(maxsize=None)
def encord(input_str: str) -> str:
    if not input_str:
        return ""
    counts = [0] * 36
    seen = [False] * 36
    order_idx = []
    for ch in input_str:
        idx = _CHAR_TO_IDX.get(ch, None)
        if idx is None:
            from collections import Counter

            cnt = Counter(input_str)
            seen2 = set()
            order2 = []
            for ch2 in input_str:
                if ch2 not in seen2:
                    seen2.add(ch2)
                    order2.append(ch2)
            return "".join(f"{ch2}{cnt[ch2]}" for ch2 in order2)

        counts[idx] += 1
        if not seen[idx]:
            seen[idx] = True
            order_idx.append(idx)

    parts = []
    for idx in order_idx:
        parts.append(_ALLOWED_CHARS[idx])
        parts.append(str(counts[idx]))
    return "".join(parts)




## === cell 11
train_f27_str = train["f_27"].astype(str, copy=False)
test_f27_str = test["f_27"].astype(str, copy=False)

unique_f27 = pd.unique(
    pd.concat([train_f27_str, test_f27_str], axis=0, ignore_index=True)
)

enc_map = {s: encord(s) for s in unique_f27}

train["f_27_en"] = train_f27_str.map(enc_map)
_ = None



## === cell 12
test["f_27_ent"] = test_f27_str.map(enc_map)
_ = None



## === cell 13
label_f27 = LabelEncoder()
label_f27.fit(unique_f27.astype(object, copy=False))

train["en_27"] = label_f27.transform(train_f27_str)
test["en_27"] = label_f27.transform(test_f27_str)

label_f27_en = LabelEncoder()
unique_f27_en = pd.unique(
    pd.concat(
        [
            train["f_27_en"].astype(str, copy=False),
            test["f_27_ent"].astype(str, copy=False),
        ],
        axis=0,
        ignore_index=True,
    )
)
label_f27_en.fit(unique_f27_en.astype(object, copy=False))

train["f_27_enc"] = label_f27_en.transform(train["f_27_en"].astype(str, copy=False))
test["f_27_enc"] = label_f27_en.transform(test["f_27_ent"].astype(str, copy=False))

_ = None



## === cell 14
_ = None



## === cell 15
_ = None



## === cell 16
_ = None



## === cell 17
_ = None



## === cell 18
_ = None



## === cell 19
_ = None



## === cell 20
_ = None



## === cell 21
_ = None



## === cell 22
X = train.drop(["id", "target", "f_27", "en_27", "f_27_en"], axis=1)
y = train["target"]

X_test = test.drop(["id", "f_27", "en_27", "f_27_ent"], axis=1)
X_test = X_test.reindex(columns=X.columns)

feature_names = list(X.columns)

X_np = np.ascontiguousarray(X.to_numpy(dtype=np.float32, copy=False))
y_np = y.to_numpy(dtype=np.int8, copy=False)
X_test_np = np.ascontiguousarray(X_test.to_numpy(dtype=np.float32, copy=False))

del X, y, X_test
del train, test
gc.collect()



## === cell 23
params = {
    "n_estimators": 10000,
    "colsample_bytree": 0.5,
    "subsample": 0.5,
    "learning_rate": 0.02,
    "max_depth": 6,
}



## === cell 24
splits = 5
seed = 42
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

preds = []
scores = []


def _gpu_visible() -> bool:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    return cvd not in {"", "-1"}


use_gpu = _gpu_visible()

xgb_train_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": params["learning_rate"],
    "max_depth": params["max_depth"],
    "subsample": params["subsample"],
    "colsample_bytree": params["colsample_bytree"],
    "seed": seed,
    "verbosity": 0,
}

if use_gpu:
    xgb_train_params.update(
        {
            "tree_method": "gpu_hist",
            "predictor": "gpu_predictor",
            "gpu_id": 0,
            "device": "cuda",
            "max_bin": 256,
        }
    )
else:
    xgb_train_params.update(
        {
            "tree_method": "hist",
            "predictor": "auto",
            "nthread": cpu_n,
            "max_bin": 256,
        }
    )

_DMat = xgb.QuantileDMatrix if hasattr(xgb, "QuantileDMatrix") else xgb.DMatrix

dall = _DMat(X_np, label=y_np, feature_names=feature_names)
dtest = _DMat(X_test_np, feature_names=feature_names)

fold_indices = list(skf.split(X_np, y_np))
cvfolds = [
    (train_idx.astype(np.int32, copy=False), valid_idx.astype(np.int32, copy=False))
    for train_idx, valid_idx in fold_indices
]

cv_res = xgb.cv(
    params=xgb_train_params,
    dtrain=dall,
    num_boost_round=params["n_estimators"],
    folds=cvfolds,
    early_stopping_rounds=100,
    metrics=("auc",),
    seed=seed,
    verbose_eval=False,
)
best_num_boost_round = int(cv_res.shape[0])

for fold, (idx_train, idx_valid) in enumerate(fold_indices):
    dtrain = dall.slice(idx_train)
    dvalid = dall.slice(idx_valid)

    booster = xgb.train(
        params=xgb_train_params,
        dtrain=dtrain,
        num_boost_round=best_num_boost_round,
        evals=[(dvalid, "valid")],
        verbose_eval=False,
    )

    pred_valid = booster.inplace_predict(
        X_np[idx_valid], iteration_range=(0, best_num_boost_round)
    )
    score = roc_auc_score(y_np[idx_valid], pred_valid)
    scores.append(float(score))

    test_preds_fold = booster.inplace_predict(
        X_test_np, iteration_range=(0, best_num_boost_round)
    )
    preds.append(test_preds_fold)

    print("fold : ", fold, "score : ", score)

    del dtrain, dvalid, booster, pred_valid, test_preds_fold
    gc.collect()



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/2557114030.py in <cell line: 0>()
     64 ]
     65 
---> 66 cv_res = xgb.cv(
     67     params=xgb_train_params,
     68     dtrain=dall,

/usr/local/lib/python3.11/dist-packages/xgboost/training.py in cv(params, dtrain, num_boost_round, nfold, stratified, folds, metrics, obj, feval, maximize, early_stopping_rounds, fpreproc, as_pandas, verbose_eval, show_stdv, seed, callbacks, shuffle, custom_metric)
    541 
    542     results: Dict[str, List[float]] = {}
--> 543     cvfolds = mknfold(
    544         dtrain, nfold, params, seed, metrics, fpreproc, stratified, folds, shuffle
    545     )

/usr/local/lib/python3.11/dist-packages/xgboost/training.py in mknfold(dall, nfold, param, seed, evals, fpreproc, stratified, folds, shuffle)
    396     for k in range(nfold):
    397         # perform the slicing using the indexes determined by the above methods
--> 398         dtrain = dall.slice(in_idset[k])
    399         dtest = dall.slice(out_idset[k])
    400         # run preprocessing on the data set if needed

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in slice(self, rindex, allow_groups)
   1256         res.handle = ctypes.c_void_p()
   1257         rindex = _maybe_np_slice(rindex, dtype=np.int32)
-> 1258         _check_call(
   1259             _LIB.XGDMatrixSliceDMatrixEx(
   1260                 self.handle,

/usr/local/lib/python3.11/dist-packages/xgboost/core.py in _check_call(ret)
    280     """
    281     if ret != 0:
--> 282         raise XGBoostError(py_str(_LIB.XGBGetLastError()))
    283 
    284 

XGBoostError: [21:33:55] /workspace/src/data/iterative_dmatrix.h:88: Slicing DMatrix is not supported for Quantile DMatrix.
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



## === cell 25
print(scores)
print("CV mean AUC:", float(np.mean(scores)), "std:", float(np.std(scores)))



## === cell 26
sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)

if len(preds) == 0:
    raise RuntimeError("No fold predictions were generated; cannot create submission.")

test_pred_mean = np.mean(np.vstack(preds), axis=0)
sub["target"] = test_pred_mean.astype(float)

sub.to_csv("submission.csv", index=False)
_ = sub.head()
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3695207143.py in <cell line: 0>()
      4 
      5 if len(preds) == 0:
----> 6     raise RuntimeError("No fold predictions were generated; cannot create submission.")
      7 
      8 test_pred_mean = np.mean(np.vstack(preds), axis=0)

RuntimeError: No fold predictions were generated; cannot create submission.
