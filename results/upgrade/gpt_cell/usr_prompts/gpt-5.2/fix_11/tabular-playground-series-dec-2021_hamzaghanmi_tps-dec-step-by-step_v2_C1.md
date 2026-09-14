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
pass



## === cell 11
pass




## === cell 12
def reduce_mem_usage(df, verbose=True):
    return df




## === cell 13
pass



## === cell 14
train = train[train["Cover_Type"] != 5].reset_index(drop=True)



## === cell 15
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



## === cell 16
params = {
    "objective": "multiclass",
    "random_state": 48,
    "n_estimators": 20000,
    "n_jobs": -1,  # kept to preserve user intent; LightGBM sklearn wrapper maps to num_threads internally
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



## === cell 17
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
if "n_jobs" in lgb_params:
    lgb_params.pop("n_jobs", None)
lgb_params["seed"] = int(lgb_params.pop("random_state", SEED))

for trn_idx, val_idx in kf.split(X_np, y_np):
    X_tr, X_val = X_np[trn_idx], X_np[val_idx]
    y_tr, y_val = y_np[trn_idx], y_np[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr, free_raw_data=False)
    dvalid = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False)

    booster = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dvalid],
        callbacks=[es_cb],
    )

    best_iter = booster.best_iteration or n_estimators

    val_proba = booster.predict(X_val, num_iteration=best_iter)
    val_pred = (np.argmax(val_proba, axis=1) + 1).astype(np.uint8, copy=False)
    acc.append(accuracy_score(y_val, val_pred))

    test_proba = booster.predict(X_test_np, num_iteration=best_iter)
    fold_test_pred = (np.argmax(test_proba, axis=1) + 1).astype(np.uint8, copy=False)

    fold_idx = np.clip(fold_test_pred.astype(np.int16) - 1, 0, NUM_CLASSES - 1)
    np.add.at(
        vote_counts, (np.arange(n_test, dtype=np.int64), fold_idx.astype(np.int64)), 1
    )

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
        val_pred,
        test_proba,
        fold_test_pred,
        fold_idx,
    )
    gc.collect()

gc.collect()



## --- ERROR in cell 17, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mLightGBMError[0m                             Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/902617252.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     43[0m     [0mdvalid[0m [0;34m=[0m [0mlgb[0m[0;34m.[0m[0mDataset[0m[0;34m([0m[0mX_val[0m[0;34m,[0m [0mlabel[0m[0;34m=[0m[0my_val[0m[0;34m,[0m [0mreference[0m[0;34m=[0m[0mdtrain[0m[0;34m,[0m [0mfree_raw_data[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     44[0m [0;34m[0m[0m
[0;32m---> 45[0;31m     booster = lgb.train(
[0m[1;32m     46[0m         [0mlgb_params[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     47[0m         [0mdtrain[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/engine.py[0m in [0;36mtrain[0;34m(params, train_set, num_boost_round, valid_sets, valid_names, feval, init_model, keep_training_booster, callbacks)[0m
[1;32m    295[0m     [0;31m# construct booster[0m[0;34m[0m[0;34m[0m[0m
[1;32m    296[0m     [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 297[0;31m         [0mbooster[0m [0;34m=[0m [0mBooster[0m[0;34m([0m[0mparams[0m[0;34m=[0m[0mparams[0m[0;34m,[0m [0mtrain_set[0m[0;34m=[0m[0mtrain_set[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    298[0m         [0;32mif[0m [0mis_valid_contain_train[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    299[0m             [0mbooster[0m[0;34m.[0m[0mset_train_data_name[0m[0;34m([0m[0mtrain_data_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m__init__[0;34m(self, params, train_set, model_file, model_str)[0m
[1;32m   3654[0m                 )
[1;32m   3655[0m             [0;31m# construct booster object[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 3656[0;31m             [0mtrain_set[0m[0;34m.[0m[0mconstruct[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   3657[0m             [0;31m# copy the parameters from train_set[0m[0;34m[0m[0;34m[0m[0m
[1;32m   3658[0m             [0mparams[0m[0;34m.[0m[0mupdate[0m[0;34m([0m[0mtrain_set[0m[0;34m.[0m[0mget_params[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36mconstruct[0;34m(self)[0m
[1;32m   2588[0m             [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2589[0m                 [0;31m# create train[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2590[0;31m                 self._lazy_init(
[0m[1;32m   2591[0m                     [0mdata[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mdata[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2592[0m                     [0mlabel[0m[0;34m=[0m[0mself[0m[0;34m.[0m[0mlabel[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m_lazy_init[0;34m(self, data, label, reference, weight, group, init_score, predictor, feature_name, categorical_feature, params, position)[0m
[1;32m   2185[0m             [0mself[0m[0;34m.[0m[0m__init_from_csc[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mparams_str[0m[0;34m,[0m [0mref_dataset[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2186[0m         [0;32melif[0m [0misinstance[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mnp[0m[0;34m.[0m[0mndarray[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2187[0;31m             [0mself[0m[0;34m.[0m[0m__init_from_np2d[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mparams_str[0m[0;34m,[0m [0mref_dataset[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   2188[0m         [0;32melif[0m [0m_is_pyarrow_table[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2189[0m             [0mself[0m[0;34m.[0m[0m__init_from_pyarrow_table[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mparams_str[0m[0;34m,[0m [0mref_dataset[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m__init_from_np2d[0;34m(self, mat, params_str, ref_dataset)[0m
[1;32m   2316[0m         [0mdata[0m[0;34m,[0m [0mlayout[0m [0;34m=[0m [0m_np2d_to_np1d[0m[0;34m([0m[0mmat[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   2317[0m         [0mptr_data[0m[0;34m,[0m [0mtype_ptr_data[0m[0;34m,[0m [0m_[0m [0;34m=[0m [0m_c_float_array[0m[0;34m([0m[0mdata[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 2318[0;31m         _safe_call(
[0m[1;32m   2319[0m             _LIB.LGBM_DatasetCreateFromMat(
[1;32m   2320[0m                 [0mptr_data[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/lightgbm/basic.py[0m in [0;36m_safe_call[0;34m(ret)[0m
[1;32m    311[0m     """
[1;32m    312[0m     [0;32mif[0m [0mret[0m [0;34m!=[0m [0;36m0[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 313[0;31m         [0;32mraise[0m [0mLightGBMError[0m[0;34m([0m[0m_LIB[0m[0;34m.[0m[0mLGBM_GetLastError[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0mdecode[0m[0;34m([0m[0;34m"utf-8"[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    314[0m [0;34m[0m[0m
[1;32m    315[0m [0;34m[0m[0m

[0;31mLightGBMError[0m: Number of classes should be specified and greater than 1 for multiclass training

## === cell 18
print(f"the mean Accuracy is : {round(np.mean(acc)*100,3)} ")
