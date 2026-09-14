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
lightgbm==4.6.0
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
optuna==4.5.0
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
plotly==5.24.1
plotly-express==0.4.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sklearn-pandas==2.2.0

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
import random
import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn  # scikit-learn-intelex

    patch_sklearn()
except Exception:
    pass

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from lightgbm import early_stopping
import lightgbm as lgb

SEED = 48
random.seed(SEED)
np.random.seed(SEED)

pd.options.mode.chained_assignment = None



## === cell 1
DATA_DIR = "/kaggle/input/tabular-playground-series-dec-2021"

train_path = f"{DATA_DIR}/train.csv"
test_path = f"{DATA_DIR}/test.csv"

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
test_cols = [c for c in train_cols if c != "Cover_Type"]

feature_cols = [c for c in test_cols if c != "Id"]
continous_features = feature_cols[:10]
categorical_features = feature_cols[10:]

dtype_train = {"Id": "int32", "Cover_Type": "uint8"}
dtype_test = {"Id": "int32"}
for c in feature_cols:
    if c in continous_features:
        dtype_train[c] = "float32"
        dtype_test[c] = "float32"
    else:
        dtype_train[c] = "uint8"
        dtype_test[c] = "uint8"

_read_engine = "c"
try:
    import pyarrow  # noqa: F401

    _read_engine = "pyarrow"
except Exception:
    _read_engine = "c"

usecols_train = train_cols
usecols_test = test_cols

if _read_engine == "pyarrow":
    train = pd.read_csv(
        train_path, dtype=dtype_train, usecols=usecols_train, engine="pyarrow"
    )
    test = pd.read_csv(
        test_path, dtype=dtype_test, usecols=usecols_test, engine="pyarrow"
    )
else:
    train = pd.read_csv(
        train_path,
        dtype=dtype_train,
        usecols=usecols_train,
        engine="c",
        low_memory=False,
        na_filter=False,
        memory_map=True,
    )
    test = pd.read_csv(
        test_path,
        dtype=dtype_test,
        usecols=usecols_test,
        engine="c",
        low_memory=False,
        na_filter=False,
        memory_map=True,
    )

cols = [c for c in test.columns if c != "Id"]



## === cell 2
pass



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
def reduce_mem_usage(df, verbose=True):
    return df




## === cell 11
pass



## === cell 12
train = train[train["Cover_Type"] != 5].reset_index(drop=True)



## === cell 13
cont_train = np.ascontiguousarray(
    train[continous_features].to_numpy(dtype=np.float32, copy=False)
)
cont_test = np.ascontiguousarray(
    test[continous_features].to_numpy(dtype=np.float32, copy=False)
)

train["mean"] = cont_train.mean(axis=1)
train["min"] = cont_train.min(axis=1)
train["max"] = cont_train.max(axis=1)

test["mean"] = cont_test.mean(axis=1)
test["min"] = cont_test.min(axis=1)
test["max"] = cont_test.max(axis=1)

cols = [c for c in test.columns if c != "Id"]

del cont_train, cont_test
gc.collect()



## === cell 14
params = {
    "objective": "multiclass",
    "random_state": 48,
    "n_estimators": 20000,
    "n_jobs": -1,  # kept to preserve user intent; we map to num_threads below
    "reg_alpha": 0.0010309124257626384,
    "reg_lambda": 9.48149567512538,
    "colsample_bytree": 0.5,
    "subsample": 1,
    "learning_rate": 0.3,
    "max_depth": 100,
    "num_leaves": 142,
    "min_child_samples": 204,
    "cat_smooth": 99,
    "force_row_wise": True,
    "verbosity": -1,
    "num_threads": max(1, os.cpu_count() or 1),
}



## === cell 15
X_df = train[cols]
y_ser = train["Cover_Type"]
X_test_df = test[cols]

X_np = np.ascontiguousarray(X_df.to_numpy(dtype=np.float32, copy=False))
y_np = y_ser.to_numpy(dtype=np.uint8, copy=False)
X_test_np = np.ascontiguousarray(X_test_df.to_numpy(dtype=np.float32, copy=False))

del X_df, y_ser, X_test_df
gc.collect()

n_test = X_test_np.shape[0]

LABELS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
NUM_CLASSES = len(LABELS)  # 7
vote_counts = np.zeros((n_test, NUM_CLASSES), dtype=np.uint16)

kf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)
acc = []
n = 0

es_cb = early_stopping(stopping_rounds=50, verbose=False)

lgb_params = dict(params)
n_estimators = int(lgb_params.pop("n_estimators"))
lgb_params.pop("n_jobs", None)
lgb_params["seed"] = int(lgb_params.pop("random_state", SEED))

unique_labels = np.unique(y_np)
NUM_CLASSES_TRAIN = int(unique_labels.size)
if NUM_CLASSES_TRAIN <= 1:
    raise ValueError(
        f"Multiclass training requires >1 class, but got {NUM_CLASSES_TRAIN} unique class(es)."
    )

y_np_enc = np.searchsorted(unique_labels, y_np).astype(np.int32, copy=False)
zero_to_label = unique_labels.astype(np.uint8, copy=False)
lgb_params["num_class"] = NUM_CLASSES_TRAIN

base_dtrain = lgb.Dataset(X_np, label=y_np_enc, free_raw_data=False)
base_dtrain.construct()

TEST_PRED_CHUNK = 200_000
fold_test_pred_enc = np.empty(n_test, dtype=np.int32)

rows = np.arange(n_test, dtype=np.int64)

for trn_idx, val_idx in kf.split(X_np, y_np_enc):
    dtrain = base_dtrain.subset(trn_idx)
    dvalid = base_dtrain.subset(val_idx)

    booster = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dvalid],
        callbacks=[es_cb],
    )

    best_iter = booster.best_iteration or n_estimators

    val_proba = booster.predict(X_np[val_idx], num_iteration=best_iter)
    val_pred_enc = np.argmax(val_proba, axis=1).astype(np.int32, copy=False)
    val_pred = zero_to_label[val_pred_enc]
    acc.append(accuracy_score(zero_to_label[y_np_enc[val_idx]], val_pred))

    for start in range(0, n_test, TEST_PRED_CHUNK):
        end = min(n_test, start + TEST_PRED_CHUNK)
        proba_chunk = booster.predict(X_test_np[start:end], num_iteration=best_iter)
        fold_test_pred_enc[start:end] = np.argmax(proba_chunk, axis=1).astype(
            np.int32, copy=False
        )

    fold_test_pred = zero_to_label[fold_test_pred_enc]

    fold_idx = np.clip(fold_test_pred.astype(np.int16) - 1, 0, NUM_CLASSES - 1).astype(
        np.int64, copy=False
    )
    np.add.at(vote_counts, (rows, fold_idx), 1)

    print(f"fold: {n+1} , accuracy: {round(acc[n]*100,3)}")
    n += 1

    del (
        dtrain,
        dvalid,
        booster,
        val_proba,
        val_pred_enc,
        val_pred,
        proba_chunk,
        fold_test_pred,
        fold_idx,
    )
    gc.collect()

gc.collect()



## === cell 16
print(f"the mean Accuracy is : {round(np.mean(acc)*100,3)} ")



## === cell 17
"""from sklearn.metrics import confusion_matrix, classification_report
pred_val = model.predict(X_val)
print(classification_report(y_val, model.predict(pred_val)))"""



## === cell 18
pass



## === cell 19
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

pred_idx = vote_counts.argmax(axis=1).astype(np.int16, copy=False)
prediction = LABELS[pred_idx].astype(np.int16, copy=False)

prediction[prediction == 0] = 1
prediction[prediction == 5] = 4

sub["Cover_Type"] = prediction
sub.to_csv("submission.csv", index=False)



## === cell 20
sub
