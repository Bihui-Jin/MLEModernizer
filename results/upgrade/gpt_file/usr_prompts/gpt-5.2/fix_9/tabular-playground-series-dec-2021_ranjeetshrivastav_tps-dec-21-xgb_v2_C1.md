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
import warnings
import gc

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

import xgboost as xgb
from xgboost import XGBClassifier  # kept to preserve original imports/semantics

RANDOM_STATE = 2021
os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))
np.random.seed(RANDOM_STATE)

NTHREAD = os.cpu_count() or 1
os.environ.setdefault("OMP_NUM_THREADS", str(NTHREAD))
os.environ.setdefault("MKL_NUM_THREADS", str(NTHREAD))



## === cell 1
train_path = r"../input/tabular-playground-series-dec-2021/train.csv"
test_path = r"../input/tabular-playground-series-dec-2021/test.csv"
sub_path = r"../input/tabular-playground-series-dec-2021/sample_submission.csv"

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()


def build_dtype_map(cols, has_target: bool):
    dtypes = {}
    for c in cols:
        if c == "Id":
            dtypes[c] = "int32"
        elif has_target and c == "Cover_Type":
            dtypes[c] = "int16"
        else:
            dtypes[c] = "float32"
    return dtypes


drop_cols = {"Id", "Soil_Type7", "Soil_Type15"}

usecols_train = [c for c in train_cols if c not in drop_cols]
usecols_test = [c for c in test_cols if c not in drop_cols]

train = pd.read_csv(
    train_path,
    dtype=build_dtype_map(usecols_train, has_target=True),
    usecols=usecols_train,
    engine="c",
    low_memory=False,
)
test = pd.read_csv(
    test_path,
    dtype=build_dtype_map(usecols_test, has_target=False),
    usecols=usecols_test,
    engine="c",
    low_memory=False,
)
sample_submission = pd.read_csv(
    sub_path, dtype={"Id": "int32", "Cover_Type": "int16"}, engine="c", low_memory=False
)



## === cell 2
print(f"train set have {train.shape[0]} rows and {train.shape[1]} columns.")
print(f"test set have {test.shape[0]} rows and {test.shape[1]} columns.")
print(
    f"sample_submission set have {sample_submission.shape[0]} rows and {sample_submission.shape[1]} columns."
)



## === cell 3
_ = train.head(1)



## === cell 4
pass



## === cell 5
pass



## === cell 6
pass



## === cell 7
y_raw = train["Cover_Type"].astype(np.int32).to_numpy(copy=False)

X_df = train.drop("Cover_Type", axis=1)

classes_sorted = np.unique(y_raw)
classes_sorted.sort()
y = np.searchsorted(classes_sorted, y_raw).astype(np.int32)

num_class = int(np.unique(y).size)
print("num_class:", num_class, "original classes:", classes_sorted)

X_np = np.ascontiguousarray(X_df.to_numpy(dtype=np.float32, copy=False))
y_np = np.ascontiguousarray(y, dtype=np.int32)
test_np = np.ascontiguousarray(test.to_numpy(dtype=np.float32, copy=False))

del train, test, X_df, y_raw, y
gc.collect()



## === cell 8
folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

proba_sum = np.zeros((test_np.shape[0], num_class), dtype=np.float64)
inv_n_splits = 1.0 / folds.n_splits

dtrain_full = xgb.DMatrix(X_np, label=y_np)
dtest = xgb.DMatrix(test_np)

params = {
    "tree_method": "hist",
    "learning_rate": 0.04,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "multi:softprob",
    "num_class": num_class,
    "eval_metric": "mlogloss",
    "seed": RANDOM_STATE,
    "nthread": NTHREAD,
    "max_bin": 256,
    "single_precision_histogram": True,
}

num_boost_round = 5000
early_stopping_rounds = 400

prev_booster = None
for fold, (trn_idx, val_idx) in enumerate(folds.split(X_np)):
    print(f"Fold: {fold}")

    dtrain = dtrain_full.slice(trn_idx, allow_groups=False)
    dval = dtrain_full.slice(val_idx, allow_groups=False)

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "validation")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False,
        xgb_model=prev_booster,  # warm-start to reuse histogram-building work
    )
    prev_booster = booster

    best_end = booster.best_iteration + 1

    val_proba = booster.predict(dval, iteration_range=(0, best_end))
    pred_val = np.argmax(val_proba, axis=1).astype(np.int32, copy=False)

    acc = accuracy_score(y_np[val_idx], pred_val)
    print(f" accuracy_score: {acc}")
    print("-" * 50)

    test_proba = booster.predict(dtest, iteration_range=(0, best_end))
    proba_sum += test_proba * inv_n_splits

    del dtrain, dval, val_proba, pred_val, test_proba
    if fold == folds.n_splits - 1:
        gc.collect()



## === cell 9
pred_idx = np.argmax(proba_sum, axis=1).astype(np.int32)
classes_arr = classes_sorted.astype(np.int32, copy=False)
pred_labels = classes_arr[pred_idx]

sample_submission["Cover_Type"] = pred_labels.astype(np.int16)
sample_submission.to_csv("submission.csv", index=False)

sample_submission.head()



## === cell 10
sample_submission
