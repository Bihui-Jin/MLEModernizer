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

SEED = 48
os.environ.setdefault("PYTHONHASHSEED", str(SEED))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import random
import gc

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

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass



## === cell 1
train_path = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
test_path = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

tmp_cols = pd.read_csv(test_path, nrows=0).columns.tolist()
feature_cols = [c for c in tmp_cols if c != "Id"]

continous_features = feature_cols[:10]
categorical_features = feature_cols[10:]

dtype_test = {"Id": np.int32}
for c in continous_features:
    dtype_test[c] = np.float32
for c in categorical_features:
    dtype_test[c] = np.int8

dtype_train = dtype_test.copy()
dtype_train["Cover_Type"] = np.int8

train = pd.read_csv(train_path, dtype=dtype_train, low_memory=False)
test = pd.read_csv(test_path, dtype=dtype_test, low_memory=False)

cols = feature_cols



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
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
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




## === cell 12
pass



## === cell 13
pass



## === cell 14
train = train.loc[train["Cover_Type"] != 5].reset_index(drop=True)



## === cell 15
train_feat = train[cols].to_numpy(copy=False)
test_feat = test[cols].to_numpy(copy=False)

train_feat_f = np.asarray(train_feat, dtype=np.float32, order="C")
test_feat_f = np.asarray(test_feat, dtype=np.float32, order="C")

train["mean"] = train_feat_f.mean(axis=1)
train["min"] = train_feat_f.min(axis=1)
train["max"] = train_feat_f.max(axis=1)

test["mean"] = test_feat_f.mean(axis=1)
test["min"] = test_feat_f.min(axis=1)
test["max"] = test_feat_f.max(axis=1)

cols = [c for c in test.columns if c != "Id"]

del train_feat, test_feat, train_feat_f, test_feat_f
gc.collect()



## === cell 16
scaler = StandardScaler(copy=False)

X_train_np = np.asarray(train[cols].to_numpy(copy=False), dtype=np.float32, order="C")
X_test_np = np.asarray(test[cols].to_numpy(copy=False), dtype=np.float32, order="C")

train_scaled = scaler.fit_transform(X_train_np)
test_scaled = scaler.transform(X_test_np)

del X_train_np, X_test_np
gc.collect()



## === cell 17
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

params.update(
    {
        "metric": "multi_error",  # 1-accuracy; early stopping unchanged
        "force_row_wise": True,
        "max_bin": 255,
        "verbosity": -1,
        "num_threads": params.get("n_jobs", -1),
        "boosting_type": "gbdt",
        "tree_learner": "serial",
        "data_sample_strategy": "bagging",
        "histogram_pool_size": 1024,
        "bin_construct_sample_cnt": 200000,
    }
)



## === cell 18
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

X_all = np.asarray(train_scaled, dtype=np.float32, order="C")
y_all = train["Cover_Type"].to_numpy(copy=False).astype(np.int16, copy=False)
X_test = np.asarray(test_scaled, dtype=np.float32, order="C")

del train_scaled, test_scaled
del train, test
gc.collect()

kf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)
acc = []

n_folds = kf.get_n_splits()
n_test = X_test.shape[0]
preds_arr = np.empty((n_folds, n_test), dtype=np.int16)

y_lgb = (y_all - 1).astype(np.int16, copy=False)
num_class = int(np.max(y_all))

lgb_params = {
    "objective": "multiclass",
    "num_class": num_class,
    "learning_rate": params["learning_rate"],
    "max_depth": params["max_depth"],
    "num_leaves": params["num_leaves"],
    "min_child_samples": params["min_child_samples"],
    "feature_fraction": params["colsample_bytree"],
    "bagging_fraction": params["subsample"],
    "lambda_l1": params["reg_alpha"],
    "lambda_l2": params["reg_lambda"],
    "cat_smooth": params["cat_smooth"],
    "metric": params["metric"],
    "verbosity": params["verbosity"],
    "seed": params["random_state"],
    "force_row_wise": params["force_row_wise"],
    "max_bin": params["max_bin"],
    "num_threads": params.get("num_threads", -1),
    "boosting_type": params["boosting_type"],
    "tree_learner": params["tree_learner"],
    "data_sample_strategy": params["data_sample_strategy"],
    "histogram_pool_size": params["histogram_pool_size"],
    "bin_construct_sample_cnt": params["bin_construct_sample_cnt"],
}

base_train_set = lgb.Dataset(
    X_all,
    label=y_lgb,
    free_raw_data=False,
)

for fold_i, (trn_idx, val_idx) in enumerate(kf.split(X_all, y_all), start=0):
    train_set = base_train_set.subset(trn_idx)
    val_set = base_train_set.subset(val_idx)

    booster = lgb.train(
        params=lgb_params,
        train_set=train_set,
        num_boost_round=params["n_estimators"],
        valid_sets=[val_set],
        valid_names=["valid"],
        callbacks=[
            early_stopping(stopping_rounds=100, verbose=False),
            log_evaluation(period=0),
        ],
    )

    best_iter = booster.best_iteration
    val_proba = booster.predict(X_all[val_idx], num_iteration=best_iter)
    val_pred = np.argmax(val_proba, axis=1).astype(np.int16) + 1
    acc_fold = accuracy_score(y_all[val_idx], val_pred)
    acc.append(acc_fold)

    test_proba = booster.predict(X_test, num_iteration=best_iter)
    test_pred = np.argmax(test_proba, axis=1).astype(np.int16) + 1
    preds_arr[fold_i] = test_pred

    print(f"fold: {fold_i + 1} , accuracy: {round(acc_fold * 100, 3)}")

    del (
        train_set,
        val_set,
        booster,
        trn_idx,
        val_idx,
        val_proba,
        val_pred,
        test_proba,
        test_pred,
    )
    gc.collect()



## === cell 19
print(f"the mean Accuracy is : {round(np.mean(acc) * 100, 3)} ")



## === cell 20
pass



## === cell 21
classes_ = np.unique(y_all)
n_classes = int(classes_.max())  # expected 7 for this competition, but keep safe

preds_safe = np.clip(preds_arr, 1, n_classes).astype(np.int16, copy=False)

n_folds, n_samples = preds_safe.shape
flat = (
    (preds_safe - 1).ravel().astype(np.int32, copy=False)
)  # now guaranteed non-negative
sample_ids = np.tile(np.arange(n_samples, dtype=np.int32), n_folds)

counts = np.bincount(
    sample_ids * n_classes + flat, minlength=n_samples * n_classes
).reshape(n_samples, n_classes)
prediction = counts.argmax(axis=1).astype(np.int16) + 1

sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/sample_submission.csv"
)
sub["Cover_Type"] = prediction.astype(int)
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
