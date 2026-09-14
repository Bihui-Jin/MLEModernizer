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
import numpy as np
import pandas as pd
import random
import math
import gc
import os

SEED = 48
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

random.seed(SEED)
np.random.seed(SEED)



## === cell 1
TRAIN_PATH = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

_train_head = pd.read_csv(TRAIN_PATH, nrows=0)
_test_head = pd.read_csv(TEST_PATH, nrows=0)

dtype_map_train = {
    c: np.int16 for c in _train_head.columns if c not in ("Id", "Cover_Type")
}
dtype_map_train["Id"] = np.int32
dtype_map_train["Cover_Type"] = np.int8

dtype_map_test = {c: np.int16 for c in _test_head.columns if c != "Id"}
dtype_map_test["Id"] = np.int32

train = pd.read_csv(TRAIN_PATH, dtype=dtype_map_train, low_memory=False)
test = pd.read_csv(TEST_PATH, dtype=dtype_map_test, low_memory=False)

cols = [c for c in test.columns if c != "Id"]
continous_features = cols[:10]
categorical_features = cols[10:]



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
pass



## === cell 13
pass



## === cell 14
pass



## === cell 15
elev_tr = train["Elevation"].to_numpy(copy=False)
elev_te = test["Elevation"].to_numpy(copy=False)
train["binned_elevation"] = (elev_tr // 50).astype(np.int16, copy=False)
test["binned_elevation"] = (elev_te // 50).astype(np.int16, copy=False)

hroad_tr = train["Horizontal_Distance_To_Roadways"].to_numpy(copy=False)
hroad_te = test["Horizontal_Distance_To_Roadways"].to_numpy(copy=False)
train["Horizontal_Distance_To_Roadways_Log"] = np.log(
    hroad_tr.astype(np.float32, copy=False) + 300.0
)
test["Horizontal_Distance_To_Roadways_Log"] = np.log(
    hroad_te.astype(np.float32, copy=False) + 300.0
)

st32_tr = train["Soil_Type32"].to_numpy(copy=False)
st12_tr = train["Soil_Type12"].to_numpy(copy=False)
st32_te = test["Soil_Type32"].to_numpy(copy=False)
st12_te = test["Soil_Type12"].to_numpy(copy=False)
train["Soil_Type12_32"] = (st32_tr + st12_tr).astype(np.int16, copy=False)
test["Soil_Type12_32"] = (st32_te + st12_te).astype(np.int16, copy=False)

st23_tr = train["Soil_Type23"].to_numpy(copy=False)
st22_tr = train["Soil_Type22"].to_numpy(copy=False)
st33_tr = train["Soil_Type33"].to_numpy(copy=False)
st23_te = test["Soil_Type23"].to_numpy(copy=False)
st22_te = test["Soil_Type22"].to_numpy(copy=False)
st33_te = test["Soil_Type33"].to_numpy(copy=False)
train["Soil_Type23_22_32_33"] = (st23_tr + st22_tr + st32_tr + st33_tr).astype(
    np.int16, copy=False
)
test["Soil_Type23_22_32_33"] = (st23_te + st22_te + st32_te + st33_te).astype(
    np.int16, copy=False
)

cols = [c for c in test.columns if c != "Id"]



## === cell 16
scaler = StandardScaler(copy=False)

X_train_np = train[cols].to_numpy(dtype=np.float32, copy=False)
X_test_np = test[cols].to_numpy(dtype=np.float32, copy=False)

X_train = scaler.fit_transform(X_train_np)
X_test = scaler.transform(X_test_np)

X_train = np.ascontiguousarray(X_train, dtype=np.float32)
X_test = np.ascontiguousarray(X_test, dtype=np.float32)

y = train["Cover_Type"].to_numpy(copy=False)

del (
    X_train_np,
    X_test_np,
    elev_tr,
    elev_te,
    hroad_tr,
    hroad_te,
    st32_tr,
    st12_tr,
    st32_te,
    st12_te,
    st23_tr,
    st22_tr,
    st33_tr,
    st23_te,
    st22_te,
    st33_te,
)
gc.collect()



## === cell 17
params = {
    "objective": "multiclass",
    "random_state": 48,
    "n_estimators": 20000,
    "n_jobs": -1,  # sklearn wrapper compatibility
    "num_threads": 4,  # bound threads to avoid oversubscription
    "reg_alpha": 0.9481920810028138,
    "reg_lambda": 8.15049828410672,
    "colsample_bytree": 0.5,
    "subsample": 0.8,
    "learning_rate": 0.2,
    "max_depth": 100,
    "num_leaves": 26,
    "min_child_samples": 88,
    "cat_smooth": 78,
    "verbose": -1,
    "force_row_wise": True,
    "histogram_pool_size": 1024,
}



## === cell 18
import lightgbm as lgb

kf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)
acc = []
n = 0

n_test = X_test.shape[0]
n_classes = int(np.max(y)) + 1  # -> 8
vote_counts = np.zeros((n_test, n_classes), dtype=np.uint16)

lgb_params = dict(params)
num_boost_round = int(lgb_params.pop("n_estimators"))
lgb_params.pop("n_jobs", None)
lgb_params["seed"] = lgb_params.pop("random_state", SEED)
lgb_params["num_class"] = 7
lgb_params["predict_disable_shape_check"] = True
lgb_params["use_prediction_cache"] = True
lgb_params.setdefault("metric", "multi_logloss")

y_lgb = y.astype(np.int32, copy=False) - 1  # 0..6

dall = lgb.Dataset(X_train, label=y_lgb, free_raw_data=False)

folds = list(kf.split(X_train, y_lgb))

cv_res = lgb.cv(
    params=lgb_params,
    train_set=dall,
    folds=folds,
    num_boost_round=num_boost_round,
    stratified=False,  # we already provide stratified folds explicitly
    callbacks=[
        lgb.log_evaluation(period=0),
        lgb.early_stopping(stopping_rounds=200, first_metric_only=True, verbose=False),
    ],
    seed=SEED,
)

best_iter_cv = len(next(iter(cv_res.values()))) if len(cv_res) else num_boost_round
print(f"CV-selected best iteration: {best_iter_cv}")

test_rows = np.arange(n_test, dtype=np.int32)
init_booster = None

for trn_idx, val_idx in folds:
    dtrain = lgb.Dataset(
        X_train[trn_idx],
        label=y_lgb[trn_idx],
        free_raw_data=False,
    )

    booster = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=best_iter_cv,
        init_model=init_booster,
        callbacks=[lgb.log_evaluation(period=0)],
    )

    fold_test_pred = booster.predict(
        X_test, num_iteration=best_iter_cv, predict_type="predict"
    )
    fold_test_pred = fold_test_pred.astype(np.int16, copy=False) + 1  # back to 1..7
    vote_counts[test_rows, fold_test_pred] += 1

    y_val = y[val_idx]
    val_pred = booster.predict(
        X_train[val_idx], num_iteration=best_iter_cv, predict_type="predict"
    )
    val_pred = val_pred.astype(np.int16, copy=False) + 1
    acc.append(accuracy_score(y_val, val_pred))

    print(f"fold: {n+1} , accuracy: {round(acc[n]*100,3)}")
    n += 1

    init_booster = booster

    del dtrain, fold_test_pred, val_pred, y_val
    gc.collect()



## === cell 19
print(f"the mean Accuracy is : {round(np.mean(acc)*100,3)} ")



## === cell 20
pass



## === cell 21
prediction = vote_counts[:, 1:].argmax(axis=1).astype(np.int16) + 1



## === cell 22
sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/sample_submission.csv"
)
sub["Cover_Type"] = prediction
sub.to_csv("submission.csv", index=False)
sub.head()
