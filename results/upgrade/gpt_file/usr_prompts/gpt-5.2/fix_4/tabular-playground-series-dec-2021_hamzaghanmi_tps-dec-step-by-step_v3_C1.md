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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import gc
import os

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier, early_stopping
from sklearn.metrics import accuracy_score
from scipy import stats

from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

SEED = 48
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
RUN_EDA = False

TRAIN_PATH = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

_train_head = pd.read_csv(TRAIN_PATH, nrows=0)
_test_head = pd.read_csv(TEST_PATH, nrows=0)

dtype_train = {}
dtype_test = {}

for c in _train_head.columns:
    if c == "Id":
        dtype_train[c] = np.int32
    elif c == "Cover_Type":
        dtype_train[c] = np.int8
    else:
        dtype_train[c] = np.float32  # overridden to int8 for binaries below if needed

for c in _test_head.columns:
    if c == "Id":
        dtype_test[c] = np.int32
    else:
        dtype_test[c] = np.float32  # overridden below if needed

cols_test = [c for c in _test_head.columns if c != "Id"]
continous_features = cols_test[:10]
categorical_features = cols_test[10:]

for c in cols_test:
    if c in continous_features:
        dtype_train[c] = np.float32
        dtype_test[c] = np.float32
    else:
        dtype_train[c] = np.int8
        dtype_test[c] = np.int8

del _train_head, _test_head
gc.collect()

train = pd.read_csv(TRAIN_PATH, dtype=dtype_train, low_memory=False)
test = pd.read_csv(TEST_PATH, dtype=dtype_test, low_memory=False)

cols = [c for c in test.columns if c != "Id"]



## === cell 2
if RUN_EDA:
    train.head()



## === cell 3
if RUN_EDA:
    test.head()



## === cell 4
if RUN_EDA:
    train.info()



## === cell 5
if RUN_EDA:
    test.info()



## === cell 6
if RUN_EDA:
    train.isnull().sum()



## === cell 7
if RUN_EDA:
    test.isnull().sum()



## === cell 8
if RUN_EDA:
    train[continous_features].describe()



## === cell 9
if RUN_EDA:
    test[continous_features].describe()



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
"""# plot the categorical features 
i = 1
plt.figure()
fig, ax = plt.subplots(11, 4,figsize=(20, 22))
for feature in categorical_features:
    plt.subplot(11, 4,i)
    sns.catplot(x=feature,color="blue", data=train, label='train_'+feature)
    sns.catplot(x=feature,color="olive", data=test, label='test_'+feature)
    plt.xlabel(feature, fontsize=9); plt.legend()
    i += 1
plt.show()"""



## === cell 12
if RUN_EDA:
    sns.catplot(x="Cover_Type", kind="count", palette="ch:.25", data=train)



## === cell 13
if RUN_EDA:
    train.Cover_Type.value_counts()



## === cell 14
if RUN_EDA:
    corr = train[continous_features + ["Cover_Type"]].corr()
    try:
        _ = corr.style.background_gradient(cmap="coolwarm").format(precision=3)
    except Exception:
        pass




## === cell 15
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




## === cell 16
pass



## === cell 17
pass



## === cell 18
train.drop(train[train["Cover_Type"] == 5].index, inplace=True)



## === cell 19
train_cont = train[continous_features].to_numpy(dtype=np.float32, copy=False)
test_cont = test[continous_features].to_numpy(dtype=np.float32, copy=False)

train["mean"] = train_cont.mean(axis=1)
train["min"] = train_cont.min(axis=1)
train["max"] = train_cont.max(axis=1)

test["mean"] = test_cont.mean(axis=1)
test["min"] = test_cont.min(axis=1)
test["max"] = test_cont.max(axis=1)

cols = [c for c in test.columns if c != "Id"]

del train_cont, test_cont
gc.collect()



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
X = np.ascontiguousarray(train[cols].to_numpy(copy=False))
y = train["Cover_Type"].to_numpy(copy=False)
X_test = np.ascontiguousarray(test[cols].to_numpy(copy=False))

preds = []
kf = StratifiedKFold(n_splits=5, random_state=48, shuffle=True)
acc = []
n = 0

for trn_idx, val_idx in kf.split(X, y):
    X_tr, X_val = X[trn_idx], X[val_idx]
    y_tr, y_val = y[trn_idx], y[val_idx]

    model = LGBMClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=100, verbose=False)],
    )

    best_iter = getattr(model, "best_iteration_", None)
    val_pred = model.predict(X_val, num_iteration=best_iter)
    test_pred = model.predict(X_test, num_iteration=best_iter)

    preds.append(test_pred)
    acc.append(accuracy_score(y_val, val_pred))

    print(f"fold: {n+1} , accuracy: {round(acc[n]*100, 3)}")
    n += 1

    del X_tr, X_val, y_tr, y_val, val_pred, test_pred, model
    gc.collect()



## === cell 22
print(f"the mean Accuracy is : {round(np.mean(acc)*100, 3)} ")



## === cell 23
if RUN_EDA:
    try:
        import lightgbm as lgb

        ax = lgb.plot_importance(model, max_num_features=20, figsize=(10, 10))
        plt.show()
    except Exception as e:
        print("Skipping feature importance plot due to:", repr(e))



## === cell 24
preds



## === cell 25
sub = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/sample_submission.csv"
)

preds_arr = np.vstack(preds).astype(np.int16, copy=False)  # (n_folds, n_test)

max_label = int(preds_arr.max())
vote_counts = np.zeros((max_label + 1, preds_arr.shape[1]), dtype=np.int16)
for i in range(preds_arr.shape[0]):
    vote_counts[preds_arr[i]] += 1
prediction = vote_counts.argmax(axis=0)

sub["Cover_Type"] = prediction.astype(int)
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## === cell 26
sub
