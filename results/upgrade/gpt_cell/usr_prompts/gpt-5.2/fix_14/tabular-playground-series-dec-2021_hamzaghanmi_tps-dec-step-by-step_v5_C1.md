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

SEED = 48
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass




## === cell 1
train_path = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
test_path = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()

feature_cols = [c for c in test_cols if c != "Id"]
continuous_features = feature_cols[:10]
categorical_features = feature_cols[10:]

dtype_train = {"Id": np.int32, "Cover_Type": np.int8}
dtype_test = {"Id": np.int32}
for c in feature_cols:
    if c in continuous_features:
        dtype_train[c] = np.float32
        dtype_test[c] = np.float32
    else:
        dtype_train[c] = np.int8
        dtype_test[c] = np.int8

usecols_train = ["Id"] + feature_cols + ["Cover_Type"]
usecols_test = ["Id"] + feature_cols

try:
    import pyarrow  # noqa: F401

    train = pd.read_csv(
        train_path,
        dtype=dtype_train,
        engine="pyarrow",
        usecols=usecols_train,
        low_memory=False,
    )
    test = pd.read_csv(
        test_path,
        dtype=dtype_test,
        engine="pyarrow",
        usecols=usecols_test,
        low_memory=False,
    )
except Exception:
    train = pd.read_csv(
        train_path,
        dtype=dtype_train,
        engine="c",
        usecols=usecols_train,
        low_memory=False,
    )
    test = pd.read_csv(
        test_path,
        dtype=dtype_test,
        engine="c",
        usecols=usecols_test,
        low_memory=False,
    )




## === cell 2
cols = feature_cols
continous_features = continuous_features
categorical_features = categorical_features




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
def reduce_mem_usage(df, verbose=True):
    start_memory = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        dt = df[col].dtype
        if pd.api.types.is_integer_dtype(dt):
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
        elif pd.api.types.is_float_dtype(dt):
            df[col] = df[col].astype(np.float32)

    end_memory = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage of dataframe after reduction {end_memory:.2f} MB")
        if start_memory > 0:
            print(
                f"Reduced by {100 * (start_memory - end_memory) / start_memory:.2f} %"
            )
    return df




## === cell 10
pass




## === cell 11
pass




## === cell 12
train.drop(train[train["Cover_Type"] == 5].index, inplace=True)

train_mat = train[feature_cols].to_numpy(dtype=np.float32, copy=False)
train["mean"] = train_mat.mean(axis=1)
train["min"] = train_mat.min(axis=1)
train["max"] = train_mat.max(axis=1)
del train_mat
gc.collect()

test_mat = test[feature_cols].to_numpy(dtype=np.float32, copy=False)
test["mean"] = test_mat.mean(axis=1)
test["min"] = test_mat.min(axis=1)
test["max"] = test_mat.max(axis=1)
del test_mat
gc.collect()

cols = feature_cols + ["mean", "min", "max"]




## === cell 13
scaler = StandardScaler(copy=False)

X_all = train[cols].to_numpy(dtype=np.float32, copy=False)
X_test = test[cols].to_numpy(dtype=np.float32, copy=False)

X_all = np.ascontiguousarray(X_all, dtype=np.float32)
X_test = np.ascontiguousarray(X_test, dtype=np.float32)

scaler.fit(X_all)
scaler.transform(X_all)
scaler.transform(X_test)

y_all = train["Cover_Type"].to_numpy(copy=False)

del train, test
gc.collect()




## === cell 14
cat_feature_names = [c for c in categorical_features if c in cols]
cat_feature_indices = [cols.index(c) for c in cat_feature_names]

params = {
    "objective": "multiclass",
    "random_state": 48,
    "seed": 48,
    "deterministic": True,
    "num_class": 7,  # Cover_Type is 1..7 (even though we drop 5, class count stays 7)
    "learning_rate": 0.2,
    "max_depth": 100,
    "num_leaves": 122,
    "min_child_samples": 47,
    "cat_smooth": 26,
    "feature_fraction": 0.6,  # colsample_bytree
    "bagging_fraction": 0.8,  # subsample
    "bagging_freq": 1,
    "lambda_l1": 4.496508090229224,  # reg_alpha
    "lambda_l2": 9.692734853531013,  # reg_lambda
    "force_row_wise": True,
    "histogram_pool_size": 1024,
    "verbosity": -1,
    "num_threads": -1,
    "metric": "multi_logloss",
}

num_boost_round = 20000
early_stopping_rounds = 100




## === cell 15
kf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)
acc = []

n_test = X_test.shape[0]
n_folds = kf.get_n_splits()
preds_mat = np.empty((n_folds, n_test), dtype=np.int16)

y_all_lgb = (y_all - 1).astype(np.int8, copy=False)

dall = lgb.Dataset(
    X_all,
    label=y_all_lgb,
    free_raw_data=False,
    categorical_feature=cat_feature_indices if len(cat_feature_indices) else "auto",
)

for fold_i, (trn_idx, val_idx) in enumerate(kf.split(X_all, y_all)):
    dtrain = dall.subset(trn_idx, params={"free_raw_data": False})
    dvalid = dall.subset(val_idx, params={"free_raw_data": False})

    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)
        ],
    )

    val_proba = booster.predict(X_all[val_idx], num_iteration=booster.best_iteration)
    val_pred = (np.argmax(val_proba, axis=1) + 1).astype(y_all.dtype, copy=False)
    fold_acc = float((val_pred == y_all[val_idx]).mean())
    acc.append(fold_acc)

    test_proba = booster.predict(X_test, num_iteration=booster.best_iteration)
    preds_mat[fold_i] = (np.argmax(test_proba, axis=1) + 1).astype(np.int16, copy=False)

    print(f"fold: {fold_i+1} , accuracy: {round(fold_acc*100,3)}")

    del dtrain, dvalid, booster, val_proba, val_pred, test_proba
    gc.collect()




## === cell 16
print(f"the mean Accuracy is : {round(np.mean(acc)*100,3)} ")




## === cell 17
pass




## === cell 18
pass




## === cell 19
sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/sample_submission.csv"
)

preds0 = (preds_mat - 1).astype(np.int16, copy=False)  # classes 0..6

n_folds, n_test = preds0.shape
n_classes = 7

flat = (
    preds0.astype(np.int64, copy=False) * n_test
    + np.arange(n_test, dtype=np.int64)[None, :]
).ravel()
counts = np.bincount(flat, minlength=n_classes * n_test).reshape(n_classes, n_test)

prediction0 = counts.argmax(axis=0).astype(np.int16, copy=False)
sub["Cover_Type"] = (prediction0 + 1).astype(np.int16, copy=False)
sub.to_csv("submission.csv", index=False)




## === cell 20
sub
