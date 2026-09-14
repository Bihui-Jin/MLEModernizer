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
import numpy as np
import pandas as pd
import random
import gc

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import lightgbm as lgb

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

os.environ.setdefault("PYTHONHASHSEED", "48")
np.random.seed(48)
random.seed(48)



## === cell 1
PASS_EDA = True



## === cell 2
TRAIN_PATH = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

_train_cols = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
_test_cols = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()

feature_cols = [c for c in _test_cols if c != "Id"]
continous_features = feature_cols[:10]
categorical_features = feature_cols[10:]

dtype_train = {c: np.int32 for c in _train_cols if c == "Id"}
dtype_train.update({c: np.int16 for c in feature_cols})
dtype_train["Cover_Type"] = np.int8

dtype_test = {c: np.int32 for c in _test_cols if c == "Id"}
dtype_test.update({c: np.int16 for c in feature_cols})

usecols_train = ["Id"] + feature_cols + ["Cover_Type"]
usecols_test = ["Id"] + feature_cols

train = pd.read_csv(TRAIN_PATH, usecols=usecols_train, dtype=dtype_train)
test = pd.read_csv(TEST_PATH, usecols=usecols_test, dtype=dtype_test)



## === cell 3
if not PASS_EDA:
    train.head()



## === cell 4
if not PASS_EDA:
    test.head()



## === cell 5
if not PASS_EDA:
    train.info()



## === cell 6
if not PASS_EDA:
    test.info()



## === cell 7
if not PASS_EDA:
    train.isnull().sum()



## === cell 8
if not PASS_EDA:
    test.isnull().sum()



## === cell 9
if not PASS_EDA:
    train[continous_features].describe()



## === cell 10
if not PASS_EDA:
    test[continous_features].describe()



## === cell 11
if not PASS_EDA:
    import seaborn as sns
    import matplotlib.pyplot as plt

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



## === cell 12
if not PASS_EDA:
    import seaborn as sns

    sns.catplot(x="Cover_Type", kind="count", palette="ch:.25", data=train)



## === cell 13
if not PASS_EDA:
    train.Cover_Type.value_counts()



## === cell 14
if not PASS_EDA:
    corr = train[continous_features + ["Cover_Type"]].corr()
    corr.style.background_gradient(cmap="coolwarm").format(precision=3)




## === cell 15
def reduce_mem_usage_fast(df, verbose=True):
    start_memory = df.memory_usage(deep=True).sum() / 1024**2
    int_cols = df.select_dtypes(include=["int8", "int16", "int32", "int64"]).columns
    float_cols = df.select_dtypes(include=["float16", "float32", "float64"]).columns

    df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast="integer")
    df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast="float")

    end_memory = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Memory usage of dataframe after reduction {end_memory:.2f} MB")
        if start_memory > 0:
            print(
                f"Reduced by {100 * (start_memory - end_memory) / start_memory:.2f} %"
            )
    return df




## === cell 16
if any(
    train[feature_cols]
    .dtypes.astype(str)
    .isin(["int64", "float64", "float32", "int32"])
):
    pass



## === cell 17
if any(
    test[feature_cols].dtypes.astype(str).isin(["int64", "float64", "float32", "int32"])
):
    pass



## === cell 18
train.drop(train[train["Cover_Type"] == 5].index, inplace=True)



## === cell 19
tr_cont = train[continous_features].to_numpy(copy=False)
te_cont = test[continous_features].to_numpy(copy=False)

train["mean"] = tr_cont.mean(axis=1)
train["min"] = tr_cont.min(axis=1)
train["max"] = tr_cont.max(axis=1)

test["mean"] = te_cont.mean(axis=1)
test["min"] = te_cont.min(axis=1)
test["max"] = te_cont.max(axis=1)

cols = [c for c in test.columns if c != "Id"]



## === cell 20
params = {
    "objective": "multiclass",
    "random_state": 48,
    "n_estimators": 20000,
    "n_jobs": -1,
    "reg_alpha": 0.0010309124257626384,
    "reg_lambda": 9.48149567512538,
    "colsample_bytree": 0.5,
    "subsample": 1,
    "learning_rate": 0.2,
    "max_depth": 100,
    "num_leaves": 142,
    "min_child_samples": 204,
    "cat_smooth": 99,
}



## === cell 21
X_df = train[cols]
y_ser = train["Cover_Type"]
X_test_df = test[cols]

X = np.ascontiguousarray(X_df.to_numpy(copy=False).astype(np.float32, copy=False))
X_test = np.ascontiguousarray(
    X_test_df.to_numpy(copy=False).astype(np.float32, copy=False)
)
y = np.ascontiguousarray(y_ser.to_numpy(copy=False).astype(np.int32, copy=False))

num_class = int(np.max(y))  # Cover_Type is 1..7 after drop of 5 still <= 7
kf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)

acc = []
n = 0

vote_counts = np.zeros((X_test.shape[0], num_class), dtype=np.uint32)

lgb_params = {
    "objective": "multiclass",
    "num_class": num_class,
    "learning_rate": params["learning_rate"],
    "num_leaves": params["num_leaves"],
    "max_depth": params["max_depth"],
    "min_child_samples": params["min_child_samples"],
    "subsample": params["subsample"],
    "colsample_bytree": params["colsample_bytree"],
    "reg_alpha": params["reg_alpha"],
    "reg_lambda": params["reg_lambda"],
    "cat_smooth": params["cat_smooth"],
    "seed": params["random_state"],
    "feature_fraction_seed": params["random_state"],
    "bagging_seed": params["random_state"],
    "data_random_seed": params["random_state"],
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": -1,
    "verbosity": -1,
}

lgb_fit_common = {
    "callbacks": [
        lgb.early_stopping(stopping_rounds=100, verbose=False),
        lgb.log_evaluation(period=0),
    ],
    "keep_training_booster": False,
}

y0 = y - 1  # LightGBM expects 0..num_class-1
base_dataset = lgb.Dataset(X, label=y0, free_raw_data=False)

for trn_idx, val_idx in kf.split(X, y):
    dtrain = base_dataset.subset(trn_idx)
    dvalid = base_dataset.subset(val_idx, reference=dtrain)

    booster = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=params["n_estimators"],
        valid_sets=[dvalid],
        valid_names=["valid"],
        **lgb_fit_common,
    )

    val_pred0 = booster.predict(
        X[val_idx],
        num_iteration=booster.best_iteration,
        pred_type="class",
    ).astype(np.int32, copy=False)
    acc.append(accuracy_score(y0[val_idx], val_pred0))

    test_pred0 = booster.predict(
        X_test,
        num_iteration=booster.best_iteration,
        pred_type="class",
    ).astype(np.int32, copy=False)
    vote_counts[np.arange(X_test.shape[0]), test_pred0] += 1

    print(f"fold: {n+1} , accuracy: {round(acc[n]*100,3)}")
    n += 1

    del dtrain, dvalid, booster, val_pred0, test_pred0
    gc.collect()



## --- ERROR in cell 21, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mTypeError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/648971429.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     55[0m [0;32mfor[0m [0mtrn_idx[0m[0;34m,[0m [0mval_idx[0m [0;32min[0m [0mkf[0m[0;34m.[0m[0msplit[0m[0;34m([0m[0mX[0m[0;34m,[0m [0my[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     56[0m     [0mdtrain[0m [0;34m=[0m [0mbase_dataset[0m[0;34m.[0m[0msubset[0m[0;34m([0m[0mtrn_idx[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 57[0;31m     [0mdvalid[0m [0;34m=[0m [0mbase_dataset[0m[0;34m.[0m[0msubset[0m[0;34m([0m[0mval_idx[0m[0;34m,[0m [0mreference[0m[0;34m=[0m[0mdtrain[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     58[0m [0;34m[0m[0m
[1;32m     59[0m     booster = lgb.train(

[0;31mTypeError[0m: Dataset.subset() got an unexpected keyword argument 'reference'

## === cell 22
print(f"the mean Accuracy is : {round(np.mean(acc)*100,3)} ")
