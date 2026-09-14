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

# 5. Target score

0.94885

# 6. Current score

0.36817

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.36817) has done: 'I fix the pipeline-breaking errors by making KFold valid (either remove `random_state` or enable `shuffle=True`) and by ensuring the model can train on Kaggle without GPU requirements. I also correct the prediction aggregation and submission writing so the output is a valid `submission.csv` with the exact required columns and aligned `Id`s. Finally, I keep the core modeling approach (5-fold CV with XGBoost) intact while making the minimal necessary changes to avoid invalid averaged class labels and to support early stopping properly.'
- What this solution (achieved 0.36817) has done: 'The timeout is dominated by repeatedly training 5 large XGBoost models on 3.6M rows with a very high `n_estimators=5000`, plus avoidable overhead from materializing/holding multiple full-size arrays and redundant conversions. I keep the exact same model, CV scheme, and early stopping semantics, but speed things up by (1) enabling scikit-learn-intelex to accelerate KFold splitting and metrics, (2) using `xgboost.QuantileDMatrix` (hist-compatible) to reduce CPU/memory pressure and speed training/prediction while preserving results, and (3) reducing peak memory/copies by using `float32` feature matrices (lossless for these integer features) and reusing prebuilt DMatrices. These changes are equivalent in evaluation semantics (same folds, same early stopping criterion, same objective), but reduce constant factors enough to fit 600s more reliably. Paths, core logic, and predictions remain the same aside from negligible floating-point differences.'
- What this solution (achieved 0.36817) has done: 'The timeout is dominated by repeatedly building `QuantileDMatrix` for each fold and re-quantizing the training/validation data, plus extra overhead from pandas → numpy conversions and scikit-learn’s `accuracy_score` call on very large arrays. The main speed fix is to quantize once by constructing a single `QuantileDMatrix` on the full training set and then create fold views via `slice()` (this preserves the exact training setup while removing repeated expensive preprocessing). I also switch accuracy computation to a pure NumPy mean (identical result) and ensure all arrays are contiguous `float32`/`int32` once. These changes keep the model, params, CV, early stopping, and prediction semantics unchanged while cutting major constant-factor overhead.'

# 9. Code solution

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

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import xgboost as xgb

RANDOM_STATE = 2021
np.random.seed(RANDOM_STATE)



## === cell 1
DATA_DIR = r"../input/tabular-playground-series-dec-2021"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

_train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
_test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()

drop_cols = {"Soil_Type7", "Soil_Type15"}

usecols_train = [c for c in _train_cols if c not in drop_cols]
usecols_test = [c for c in _test_cols if c not in drop_cols]

dtype_train = {}
for c in usecols_train:
    if c == "Id":
        dtype_train[c] = np.int32
    elif c == "Cover_Type":
        dtype_train[c] = np.int8
    else:
        dtype_train[c] = np.int16

dtype_test = {}
for c in usecols_test:
    if c == "Id":
        dtype_test[c] = np.int32
    else:
        dtype_test[c] = np.int16

train = pd.read_csv(train_path, usecols=usecols_train, dtype=dtype_train)
test = pd.read_csv(test_path, usecols=usecols_test, dtype=dtype_test)
sample_submission = pd.read_csv(sub_path)

print(f"train set have {train.shape[0]} rows and {train.shape[1]} columns.")
print(f"test set have {test.shape[0]} rows and {test.shape[1]} columns.")
print(
    f"sample_submission set have {sample_submission.shape[0]} rows and {sample_submission.shape[1]} columns."
)



## === cell 2
pass



## === cell 3
pass



## === cell 4
pass



## === cell 5
pass



## === cell 6
from xgboost import QuantileDMatrix

y_raw = train["Cover_Type"].to_numpy(dtype=np.int32, copy=False)
X_df = train.drop("Cover_Type", axis=1)

test_ids = test["Id"].to_numpy(copy=False)
X_df = X_df.drop("Id", axis=1)
X_test_df = test.drop("Id", axis=1)

classes_ = np.unique(y_raw)
classes_.sort()
n_classes = classes_.size
idx_to_class = classes_.copy()

y = np.searchsorted(classes_, y_raw).astype(np.int32, copy=False)

X_np = np.ascontiguousarray(X_df.to_numpy(dtype=np.float32, copy=False))
X_test_np = np.ascontiguousarray(X_test_df.to_numpy(dtype=np.float32, copy=False))

folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof_acc = []
test_proba = np.zeros((X_test_np.shape[0], n_classes), dtype=np.float32)

dtrain_full = QuantileDMatrix(X_np, label=y)
dtest = QuantileDMatrix(X_test_np)

params = {
    "tree_method": "hist",
    "learning_rate": 0.4,
    "objective": "multi:softprob",
    "num_class": int(n_classes),
    "eval_metric": "mlogloss",
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,  # reg_lambda
    "seed": RANDOM_STATE,
    "nthread": -1,
}

num_boost_round = 5000
early_stopping_rounds = 400

for fold, (trn_idx, val_idx) in enumerate(folds.split(X_np)):
    print(f"Fold: {fold}")

    dtrain = dtrain_full.slice(trn_idx)
    dvalid = dtrain_full.slice(val_idx)

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dvalid, "valid")],
        verbose_eval=False,
        early_stopping_rounds=early_stopping_rounds,
    )

    best_it = booster.best_iteration
    if best_it is None:
        best_it = num_boost_round - 1

    val_proba = booster.predict(dvalid, iteration_range=(0, best_it + 1))
    val_pred_idx = np.argmax(val_proba, axis=1).astype(np.int32, copy=False)

    y_valid = y[val_idx]
    acc = float(np.mean(val_pred_idx == y_valid))
    oof_acc.append(acc)

    print(f" accuracy_score: {acc}")
    print("-" * 50)

    test_fold_proba = booster.predict(dtest, iteration_range=(0, best_it + 1))
    test_proba += test_fold_proba.astype(np.float32, copy=False) / folds.n_splits

print(f"Mean CV accuracy: {np.mean(oof_acc):.6f} ± {np.std(oof_acc):.6f}")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
XGBoostError                              Traceback (most recent call last)
/tmp/ipykernel_11/3495612857.py in <cell line: 0>()
     54 
     55     # Slice views of the pre-built QuantileDMatrix (avoids rebuilding / re-quantizing)
---> 56     dtrain = dtrain_full.slice(trn_idx)
     57     dvalid = dtrain_full.slice(val_idx)
     58 

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

XGBoostError: [17:54:05] /workspace/src/data/iterative_dmatrix.h:88: Slicing DMatrix is not supported for Quantile DMatrix.
Stack trace:
  [bt] (0) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3effba) [0x7fff7ada1fba]
  [bt] (1) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(+0x3ff7ab) [0x7fff7adb17ab]
  [bt] (2) /usr/local/lib/python3.11/dist-packages/xgboost/lib/libxgboost.so(XGDMatrixSliceDMatrixEx+0x146) [0x7fff7ab12206]
  [bt] (3) /lib/x86_64-linux-gnu/libffi.so.8(+0x7e2e) [0x7ffff63ace2e]
  [bt] (4) /lib/x86_64-linux-gnu/libffi.so.8(+0x4493) [0x7ffff63a9493]
  [bt] (5) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0xa4d8) [0x7ffff63bc4d8]
  [bt] (6) /usr/lib/python3.11/lib-dynload/_ctypes.cpython-311-x86_64-linux-gnu.so(+0x9c8e) [0x7ffff63bbc8e]
  [bt] (7) /usr/bin/python3(_PyObject_MakeTpCall+0x27c) [0x52f85c]
  [bt] (8) /usr/bin/python3(_PyEval_EvalFrameDefault+0x6bc) [0x53da0c]



## === cell 7
pred_indices = np.argmax(test_proba, axis=1).astype(np.int32, copy=False)
predictions = idx_to_class[pred_indices].astype(int, copy=False)

submission = sample_submission.copy()
if "Id" in submission.columns and len(submission) == len(predictions):
    submission["Cover_Type"] = predictions
else:
    submission = pd.DataFrame({"Id": test_ids, "Cover_Type": predictions})

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)



## === cell 8
submission
