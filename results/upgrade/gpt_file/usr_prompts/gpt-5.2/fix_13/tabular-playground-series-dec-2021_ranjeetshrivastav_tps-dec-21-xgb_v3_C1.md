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
y_raw = train["Cover_Type"].to_numpy(dtype=np.int32, copy=False)

X_df = train.drop(columns=["Cover_Type", "Id"])
test_ids = test["Id"].to_numpy(copy=False)
X_test_df = test.drop(columns=["Id"])

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

QuantileDMatrix = getattr(xgb, "QuantileDMatrix", None)
if QuantileDMatrix is not None:
    dtest = QuantileDMatrix(X_test_np)
else:
    dtest = xgb.DMatrix(X_test_np)

nthread = int(os.environ.get("OMP_NUM_THREADS", "0") or "0")
if nthread <= 0:
    nthread = min(8, os.cpu_count() or 8)

params = {
    "tree_method": "hist",
    "learning_rate": 0.4,
    "objective": "multi:softprob",
    "num_class": int(n_classes),
    "eval_metric": "merror",
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,  # reg_lambda
    "seed": RANDOM_STATE,
    "nthread": nthread,
}

num_boost_round = 5000
early_stopping_rounds = 400

for fold, (trn_idx, val_idx) in enumerate(folds.split(X_np)):
    print(f"Fold: {fold}")

    X_tr, y_tr = X_np[trn_idx], y[trn_idx]
    X_va, y_va = X_np[val_idx], y[val_idx]

    if QuantileDMatrix is not None:
        dtrain = QuantileDMatrix(X_tr, label=y_tr)
        dvalid = QuantileDMatrix(X_va, label=y_va)
    else:
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        dvalid = xgb.DMatrix(X_va, label=y_va)

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

    evals_result = booster.evals_result()
    best_it = int(min(best_it, len(evals_result["valid"]["merror"]) - 1))
    fold_merror = float(evals_result["valid"]["merror"][best_it])
    acc = 1.0 - fold_merror
    oof_acc.append(acc)

    print(f" accuracy_score: {acc}")
    print("-" * 50)

    test_fold_proba = booster.predict(dtest, iteration_range=(0, best_it + 1))
    test_proba += test_fold_proba.astype(np.float32, copy=False) / folds.n_splits

print(f"Mean CV accuracy: {np.mean(oof_acc):.6f} ± {np.std(oof_acc):.6f}")



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
