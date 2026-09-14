# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import gc
import numpy as np
import pandas as pd

from collections import OrderedDict, Counter

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

import xgboost as xgb

os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(os.cpu_count() or 1))

np.random.seed(42)

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
_ = None



## === cell 11
_ = None



## === cell 12
from functools import lru_cache


@lru_cache(maxsize=None)
def encord(input_str: str) -> str:
    if not input_str:
        return ""
    counts = {}
    order = []
    for ch in input_str:
        if ch in counts:
            counts[ch] += 1
        else:
            counts[ch] = 1
            order.append(ch)
    parts = []
    for ch in order:
        parts.append(ch)
        parts.append(str(counts[ch]))
    return "".join(parts)




## === cell 13
train_f27_str = train["f_27"].astype(str, copy=False)
test_f27_str = test["f_27"].astype(str, copy=False)

uniq_train = train_f27_str.value_counts(dropna=False).index
uniq_test = test_f27_str.value_counts(dropna=False).index
unique_f27 = uniq_train.union(uniq_test)

enc_map = {s: encord(s) for s in unique_f27}

train["f_27_en"] = train_f27_str.map(enc_map)
_ = None



## === cell 14
test["f_27_ent"] = test_f27_str.map(enc_map)
_ = None



## === cell 15
label_f27 = LabelEncoder()
label_f27.fit(unique_f27.to_numpy(dtype=object, copy=False))

train["en_27"] = label_f27.transform(train_f27_str)
test["en_27"] = label_f27.transform(test_f27_str)

label_f27_en = LabelEncoder()
uniq_train_en = (
    train["f_27_en"].astype(str, copy=False).value_counts(dropna=False).index
)
uniq_test_en = test["f_27_ent"].astype(str, copy=False).value_counts(dropna=False).index
unique_f27_en = uniq_train_en.union(uniq_test_en)

label_f27_en.fit(unique_f27_en.to_numpy(dtype=object, copy=False))

train["f_27_enc"] = label_f27_en.transform(train["f_27_en"].astype(str, copy=False))
test["f_27_enc"] = label_f27_en.transform(test["f_27_ent"].astype(str, copy=False))

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
_ = None



## === cell 23
_ = None



## === cell 24
X = train.drop(["id", "target", "f_27", "en_27", "f_27_en"], axis=1)
y = train["target"]

X_test = test.drop(["id", "f_27", "en_27", "f_27_ent"], axis=1)
X_test = X_test.reindex(columns=X.columns)

feature_names = list(X.columns)

X_np = np.ascontiguousarray(X.to_numpy(dtype=np.float32, copy=False))
y_np = y.to_numpy(dtype=np.int8, copy=False)
X_test_np = np.ascontiguousarray(X_test.to_numpy(dtype=np.float32, copy=False))

del X, y, X_test
del (
    train,
    test,
    train_f27_str,
    test_f27_str,
    uniq_train,
    uniq_test,
    unique_f27,
    enc_map,
    uniq_train_en,
    uniq_test_en,
    unique_f27_en,
)
gc.collect()



## === cell 25
params = {
    "n_estimators": 10000,
    "colsample_bytree": 0.5,
    "subsample": 0.5,
    "learning_rate": 0.02,
    "max_depth": 6,
}



## === cell 26
splits = 5
seed = 42
skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

preds = []
scores = []

cpu_n_jobs = os.cpu_count() or 1


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
            "nthread": cpu_n_jobs,
            "max_bin": 256,
        }
    )

_DMat = xgb.QuantileDMatrix if hasattr(xgb, "QuantileDMatrix") else xgb.DMatrix
dtest = _DMat(X_test_np, feature_names=feature_names)

for fold, (idx_train, idx_valid) in enumerate(skf.split(X_np, y_np)):
    X_train, y_train = X_np[idx_train], y_np[idx_train]
    X_valid, y_valid = X_np[idx_valid], y_np[idx_valid]

    dtrain = _DMat(X_train, label=y_train, feature_names=feature_names)
    dvalid = _DMat(X_valid, label=y_valid, feature_names=feature_names)

    booster = xgb.train(
        params=xgb_train_params,
        dtrain=dtrain,
        num_boost_round=params["n_estimators"],
        evals=[(dvalid, "valid")],
        early_stopping_rounds=100,
        verbose_eval=False,
    )

    best_iter = booster.best_iteration
    if best_iter is None:
        best_iter = params["n_estimators"] - 1
    best_iter = int(best_iter)

    pred_valid = booster.inplace_predict(X_valid, iteration_range=(0, best_iter + 1))
    score = roc_auc_score(y_valid, pred_valid)
    scores.append(float(score))

    test_preds_fold = booster.inplace_predict(
        X_test_np, iteration_range=(0, best_iter + 1)
    )
    preds.append(test_preds_fold)

    print("fold : ", fold, "score : ", score)

    del (
        dtrain,
        dvalid,
        booster,
        X_train,
        y_train,
        X_valid,
        y_valid,
        pred_valid,
        test_preds_fold,
        idx_train,
        idx_valid,
    )
    gc.collect()



## === cell 27
print(scores)
print("CV mean AUC:", float(np.mean(scores)), "std:", float(np.std(scores)))



## === cell 28
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
