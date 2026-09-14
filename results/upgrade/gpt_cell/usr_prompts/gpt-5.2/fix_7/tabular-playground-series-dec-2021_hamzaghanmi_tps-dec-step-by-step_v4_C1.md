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

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

SEED = 48
random.seed(SEED)
np.random.seed(SEED)

RUN_EDA = False



## === cell 1
TRAIN_PATH = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

_train_cols = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
_test_cols = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()

dtype_train = {
    c: np.int32 for c in _train_cols if c != "Cover_Type"
}  # includes Id and all features
dtype_train["Cover_Type"] = np.int8
dtype_test = {c: np.int32 for c in _test_cols}  # includes Id and all features

_read_csv_kwargs = {"dtype": dtype_train}
try:
    train = pd.read_csv(TRAIN_PATH, engine="pyarrow", **_read_csv_kwargs)
except Exception:
    train = pd.read_csv(TRAIN_PATH, **_read_csv_kwargs)

_read_csv_kwargs = {"dtype": dtype_test}
try:
    test = pd.read_csv(TEST_PATH, engine="pyarrow", **_read_csv_kwargs)
except Exception:
    test = pd.read_csv(TEST_PATH, **_read_csv_kwargs)

cols = [e for e in test.columns if e not in ("Id",)]
continous_features = cols[:10]
categorical_features = cols[10:]



## === cell 2
if RUN_EDA:
    display(train.head())



## === cell 3
if RUN_EDA:
    display(test.head())



## === cell 4
if RUN_EDA:
    train.info()



## === cell 5
if RUN_EDA:
    test.info()



## === cell 6
if RUN_EDA:
    display(train.isnull().sum())



## === cell 7
if RUN_EDA:
    display(test.isnull().sum())



## === cell 8
if RUN_EDA:
    display(train[continous_features].describe())



## === cell 9
if RUN_EDA:
    display(test[continous_features].describe())



## === cell 10
if RUN_EDA:
    i = 1
    plt.figure()
    fig, ax = plt.subplots(2, 5, figsize=(20, 12))
    for feature in continous_features:
        plt.subplot(2, 5, i)
        sns.histplot(
            train[feature], color="blue", kde=True, bins=100, label="train_" + feature
        )
        sns.histplot(
            test[feature], color="olive", kde=True, bins=100, label="test_" + feature
        )
        plt.xlabel(feature, fontsize=9)
        plt.legend()
        i += 1
    plt.show()



## === cell 11
if RUN_EDA:
    sns.catplot(x="Cover_Type", kind="count", palette="ch:.25", data=train)



## === cell 12
if RUN_EDA:
    display(train.Cover_Type.value_counts())



## === cell 13
if RUN_EDA:
    corr = train[continous_features + ["Cover_Type"]].corr()
    display(corr.style.background_gradient(cmap="coolwarm").format(precision=3))




## === cell 14
def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_memory = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_memory = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage of dataframe after reduction {end_memory} MB")
        print(f"Reduced by {100 * (start_memory - end_memory) / start_memory} % ")
    return df




## === cell 15
if RUN_EDA:
    train[cols] = reduce_mem_usage(train[cols])



## === cell 16
if RUN_EDA:
    test[cols] = reduce_mem_usage(test[cols])



## === cell 17
train.drop(train[train["Cover_Type"] == 5].index, inplace=True)



## === cell 18
train_feat_np = train[cols].to_numpy(copy=False).astype(np.float32, copy=False)
test_feat_np = test[cols].to_numpy(copy=False).astype(np.float32, copy=False)

train["mean"] = train_feat_np.mean(axis=1)
train["min"] = train_feat_np.min(axis=1)
train["max"] = train_feat_np.max(axis=1)

test["mean"] = test_feat_np.mean(axis=1)
test["min"] = test_feat_np.min(axis=1)
test["max"] = test_feat_np.max(axis=1)

cols = [e for e in test.columns if e not in ("Id",)]

del train_feat_np, test_feat_np
gc.collect()



## === cell 19
params = {
    "objective": "multiclass",
    "random_state": 48,
    "n_estimators": 20000,
    "n_jobs": -1,
    "reg_alpha": 4.496508090229224,
    "reg_lambda": 9.692734853531013,
    "colsample_bytree": 0.6,
    "subsample": 0.8,
    "learning_rate": 0.2,
    "max_depth": 100,
    "num_leaves": 122,
    "min_child_samples": 47,
    "cat_smooth": 26,
}



## === cell 20
import lightgbm as lgb

X = np.ascontiguousarray(train[cols].to_numpy(copy=False), dtype=np.float32)
y = train["Cover_Type"].to_numpy(copy=False)
X_test = np.ascontiguousarray(test[cols].to_numpy(copy=False), dtype=np.float32)

orig_classes = np.sort(np.unique(y.astype(np.int32)))
y_int = y.astype(np.int32, copy=False)
y0 = np.searchsorted(orig_classes, y_int).astype(np.int32, copy=False)

lgb_params = dict(params)
n_estimators = int(lgb_params.pop("n_estimators"))
lgb_params["num_class"] = int(orig_classes.size)
lgb_params["verbosity"] = -1
lgb_params["deterministic"] = True
lgb_params["force_row_wise"] = True  # faster/more stable on large dense-ish data

lgb_params["num_threads"] = int(os.environ.get("OMP_NUM_THREADS", "0")) or -1
lgb_params.setdefault("max_bin", 255)

preds = []
kf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)
acc = []
n = 0

for trn_idx, val_idx in kf.split(X, y):
    X_tr, X_val = X[trn_idx], X[val_idx]
    y_tr, y_val = y0[trn_idx], y0[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=True)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=True)

    booster = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dvalid],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_proba = booster.predict(X_val, num_iteration=booster.best_iteration)
    val_pred = orig_classes[np.argmax(val_proba, axis=1)].astype(np.int32)

    test_proba = booster.predict(X_test, num_iteration=booster.best_iteration)
    test_pred = orig_classes[np.argmax(test_proba, axis=1)].astype(np.int32)

    preds.append(test_pred)
    acc.append(accuracy_score(y_int[val_idx], val_pred))

    print(f"fold: {n+1} , accuracy: {round(acc[n]*100,3)}")
    n += 1

    del (
        X_tr,
        X_val,
        y_tr,
        y_val,
        dtrain,
        dvalid,
        booster,
        val_proba,
        test_proba,
        val_pred,
        test_pred,
    )
    gc.collect()



## === cell 21
print(f"the mean Accuracy is : {round(np.mean(acc)*100,3)} ")



## === cell 22
if RUN_EDA:
    from optuna.integration import (
        lightgbm as lgb_optuna,
    )  # kept to mirror original dependency

    pass



## === cell 23
if RUN_EDA:
    display(preds)



## === cell 24
sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/sample_submission.csv"
)

preds_arr = np.stack(preds, axis=0)  # (n_folds, n_test)
classes = np.unique(preds_arr)
counts = (preds_arr[..., None] == classes[None, None, :]).sum(axis=0)
prediction = classes[np.argmax(counts, axis=1)]

sub["Cover_Type"] = prediction
sub.to_csv("submission.csv", index=False)



## === cell 25
if RUN_EDA:
    display(sub)
