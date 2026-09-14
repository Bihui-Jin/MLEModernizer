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
train_feat_np = train[cols].to_numpy(copy=False)
test_feat_np = test[cols].to_numpy(copy=False)

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

X = train[cols].to_numpy(copy=False)
y = train["Cover_Type"].to_numpy(copy=False)
X_test = test[cols].to_numpy(copy=False)

y0 = y.astype(np.int32) - 1

lgb_params = dict(params)
n_estimators = int(lgb_params.pop("n_estimators"))
lgb_params["num_class"] = int(np.unique(y0).size)
lgb_params["verbosity"] = -1
lgb_params["deterministic"] = True
lgb_params["force_row_wise"] = (
    True  # often faster and more stable on large dense-ish tabular data
)

preds = []
kf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)
acc = []
n = 0

for trn_idx, val_idx in kf.split(X, y):
    X_tr, X_val = X[trn_idx], X[val_idx]
    y_tr, y_val = y0[trn_idx], y0[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

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
    val_pred = (np.argmax(val_proba, axis=1) + 1).astype(np.int32)

    test_proba = booster.predict(X_test, num_iteration=booster.best_iteration)
    test_pred = (np.argmax(test_proba, axis=1) + 1).astype(np.int32)

    preds.append(test_pred)
    acc.append(accuracy_score((y_val + 1), val_pred))

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



## --- ERROR in cell 20, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mLightGBMError[0m                             Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1057680386.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     35[0m     [0mdvalid[0m [0;34m=[0m [0mlgb[0m[0;34m.[0m[0mDataset[0m[0;34m([0m[0mX_val[0m[0;34m,[0m [0mlabel[0m[0;34m=[0m[0my_val[0m[0;34m,[0m [0mreference[0m[0;34m=[0m[0mdtrain[0m[0;34m,[0m [0mfree_raw_data[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     36[0m [0;34m[0m[0m
[0;32m---> 37[0;31m     booster = lgb.train(
[0m[1;32m     38[0m         [0mparams[0m[0;34m=[0m[0mlgb_params[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     39[0m         [0mtrain_set[0m[0;34m=[0m[0mdtrain[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/engine.py[0m in [0;36mtrain[0;34m(params, train_set, num_boost_round, valid_sets, valid_names, feval, init_model, keep_training_booster, callbacks)[0m
[1;32m    295[0m     [0;31m# construct booster[0m[0;34m[0m[0;34m[0m[0m
[1;32m    296[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 297[0;31m         [0mbooster[0m [0;34m=[0m [0mBooster[0m[0;34m([0m[0mparams[0m[0;34m=[0m[0mparams[0m[0;34m,[0m [0mtrain_set[0m[0;34m=[0m[0mtrain_set[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    298[0m         [0;32mif[0m [0mis_valid_contain_train[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    299[0m             [0mbooster[0m[0;34m.[0m[0mset_train_data_name[0m[0;34m([0m[0mtrain_data_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m__init__[0;34m(self, params, train_set, model_file, model_str)[0m
[1;32m   3658[0m             [0mparams[0m[0;34m.[0m[0mupdate[0m[0;34m([0m[0mtrain_set[0m[0;34m.[0m[0mget_params[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3659[0m             [0mparams_str[0m [0;34m=[0m [0m_param_dict_to_str[0m[0;34m([0m[0mparams[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3660[0;31m             _safe_call(
[0m[1;32m   3661[0m                 _LIB.LGBM_BoosterCreate(
[1;32m   3662[0m                     [0mtrain_set[0m[0;34m.[0m[0m_handle[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m_safe_call[0;34m(ret)[0m
[1;32m    311[0m     """
[1;32m    312[0m     [0;32mif[0m [0mret[0m [0;34m!=[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 313[0;31m         [0;32mraise[0m [0mLightGBMError[0m[0;34m([0m[0m_LIB[0m[0;34m.[0m[0mLGBM_GetLastError[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mdecode[0m[0;34m([0m[0;34m"utf-8"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    314[0m [0;34m[0m[0m
[1;32m    315[0m [0;34m[0m[0m

[0;31mLightGBMError[0m: Label must be in [0, 6), but found 6 in label

## === cell 21
print(f"the mean Accuracy is : {round(np.mean(acc)*100,3)} ")
