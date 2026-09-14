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
import gc
import os

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from lightgbm import LGBMClassifier
import lightgbm as lgb

from scipy import stats

SEED = 48
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
TRAIN_PATH = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

test_cols = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()
cols = [c for c in test_cols if c != "Id"]
continous_features = cols[:10]
categorical_features = cols[10:]

dtype_test = {c: (np.int32 if c in continous_features else np.int8) for c in cols}
dtype_train = dict(dtype_test)
dtype_train["Cover_Type"] = np.int8  # classes 1..7

usecols_train = ["Id"] + cols + ["Cover_Type"]
usecols_test = ["Id"] + cols

read_csv_kwargs = dict()
try:
    import pyarrow  # noqa: F401

    read_csv_kwargs["engine"] = "pyarrow"
except Exception:
    pass

train = pd.read_csv(
    TRAIN_PATH, usecols=usecols_train, dtype=dtype_train, **read_csv_kwargs
)
test = pd.read_csv(TEST_PATH, usecols=usecols_test, dtype=dtype_test, **read_csv_kwargs)




## === cell 2
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




## === cell 3
pass



## === cell 4
pass



## === cell 5
train.drop(train[train["Cover_Type"] == 5].index, inplace=True)



## === cell 6
train_vals = train[cols].to_numpy(dtype=np.float32, copy=False)
test_vals = test[cols].to_numpy(dtype=np.float32, copy=False)

train["mean"] = train_vals.mean(axis=1)
train["min"] = train_vals.min(axis=1)
train["max"] = train_vals.max(axis=1)

test["mean"] = test_vals.mean(axis=1)
test["min"] = test_vals.min(axis=1)
test["max"] = test_vals.max(axis=1)

cols = [c for c in test.columns if c != "Id"]

del train_vals, test_vals
gc.collect()



## === cell 7
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
    "boosting_type": "gbdt",
    "force_col_wise": True,  # avoids costly auto-detection on wide dense matrices
    "max_bin": 255,  # histogram bins (LightGBM default is 255; keep explicit to avoid environment differences)
    "verbosity": -1,
}



## === cell 8
X = np.ascontiguousarray(train[cols].to_numpy(dtype=np.float32, copy=False))
y = train["Cover_Type"].to_numpy(copy=False)
X_test = np.ascontiguousarray(test[cols].to_numpy(dtype=np.float32, copy=False))

kf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)
acc = []

n_folds = kf.get_n_splits()
n_test = X_test.shape[0]
preds_arr = np.empty((n_folds, n_test), dtype=np.int16)

for n, (trn_idx, val_idx) in enumerate(kf.split(X, y), start=1):
    X_tr, X_val = X[trn_idx], X[val_idx]
    y_tr, y_val = y[trn_idx], y[val_idx]

    model = LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
    )

    preds_arr[n - 1] = model.predict(X_test).astype(np.int16, copy=False)
    acc.append(accuracy_score(y_val, model.predict(X_val)))

    print(f"fold: {n} , accuracy: {round(acc[n-1]*100,3)}")

    del X_tr, X_val, y_tr, y_val, model
    gc.collect()



## === cell 9
print(f"the mean Accuracy is : {round(np.mean(acc)*100,3)} ")



## === cell 10
pass



## === cell 11
pass



## === cell 12
sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/sample_submission.csv",
    **read_csv_kwargs,
)

prediction = stats.mode(preds_arr, axis=0, keepdims=False).mode  # (n_test,)

sub["Cover_Type"] = prediction.astype(int)
sub.to_csv("submission.csv", index=False)



## === cell 13
pass
