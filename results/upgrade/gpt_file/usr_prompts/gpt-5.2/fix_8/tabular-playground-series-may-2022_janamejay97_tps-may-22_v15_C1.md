# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given simulated manufacturing control data, predict whether the machine is in state `0` or state `1`.

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
900000,0.65
900001,0.97
900002,0.02
etc.
```

## Dataset
- **train.csv** - the training data, which includes normalized continuous data and categorical data
- **test.csv** - the test set; your task is to predict binary `target` variable which represents the state of a manufacturing process
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
xgboost==2.0.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        input/
            description.md (79 lines)
            sample_submission.csv (100001 lines)
            sample_submission.csv.zip (224.9 kB)
            test.csv (100001 lines)
            test.csv.zip (16.6 MB)
            train.csv (800001 lines)
            train.csv.zip (133.3 MB)
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
        working/
            tabular-playground-series-may-2022/
                description.md (79 lines)
                sample_submission.csv (100001 lines)
                ... and 5 other files
                tabular-playground-series-may-2022/
```

-> data/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> data/tabular-playground-series-may-2022/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/tabular-playground-series-may-2022/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> data/test.csv has 100000 rows and 32 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 17 more columns

-> data/train.csv has 800000 rows and 33 columns.
The columns are: id, f_00, f_01, f_02, f_03, f_04, f_05, f_06, f_07, f_08, f_09, f_10, f_11, f_12, f_13... and 18 more columns

-> input/sample_submission.csv has 100000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import string
import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
import warnings

warnings.filterwarnings("ignore")

import os

DEBUG_IO_LISTING = False
if DEBUG_IO_LISTING:
    for dirname, _, filenames in os.walk("/kaggle/input"):
        for filename in filenames:
            print(os.path.join(dirname, filename))

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
except Exception:
    pass

np.random.seed(46)



## === cell 1
print("Train data:")
train = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/train.csv",
    dtype={
        "id": "int32",
        "target": "int8",
        "f_27": "string",
    },
)
print("Shape of train data: " + str(train.shape))
print()
train.head(5)



## === cell 2
print("Test data:")
test = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/test.csv",
    dtype={
        "id": "int32",
        "f_27": "string",
    },
)
print("Shape of test data: " + str(test.shape))
print()
test.head(5)



## === cell 3
print(train["target"].value_counts())



## === cell 4
DEBUG_EDA = False
if DEBUG_EDA:
    train.info()



## === cell 5
if DEBUG_EDA:
    train.isnull().sum()



## === cell 6
if DEBUG_EDA:
    train.describe()



## === cell 7
if DEBUG_EDA:
    train.nunique().sort_values(ascending=True)



## === cell 8
categorical_feats = []

for col in train.columns:
    if train[col].dtype == "int64":
        categorical_feats.append(col)

if DEBUG_EDA:
    train[categorical_feats].sample(10)



## === cell 9
if DEBUG_EDA:
    for feature in categorical_feats:
        print("Possible values for " + feature)
        print(train[feature].unique())
        print()



## === cell 10
if DEBUG_EDA:
    fig, ax = plt.subplots(figsize=(50, 50))
    sns.heatmap(train[categorical_feats].corr(), annot=True, fmt=".2f", ax=ax)
    plt.show()



## === cell 11
float_feats = []
for col in train.columns:
    if train[col].dtype == "float64":
        float_feats.append(col)

if DEBUG_EDA:
    print(float_feats)
else:
    print(float_feats)



## === cell 12
if DEBUG_EDA:
    fig, axs = plt.subplots(4, 4, figsize=(18, 18))
    axs = axs.ravel()

    for ind, ax in enumerate(axs):
        ax.hist(train[float_feats[ind]], density=True, bins=100)
        ax.set_title(
            f"Train: {float_feats[ind]}, Std. dev: {train[float_feats[ind]].std():.1f}"
        )
    plt.show()



## === cell 13
if DEBUG_EDA:
    fig, axs = plt.subplots(4, 4, figsize=(14, 24))
    axs = axs.ravel()

    for ind, ax in enumerate(axs):
        ax.boxplot(
            train[train.target == 0][float_feats[ind]], positions=[0], widths=0.7
        )
        ax.boxplot(
            train[train.target == 1][float_feats[ind]], positions=[1], widths=0.7
        )
        ax.set_title(f"{float_feats[ind]}")
    plt.show()



## === cell 14
if DEBUG_EDA:
    fig, ax = plt.subplots(figsize=(50, 50))
    sns.heatmap(train[float_feats + ["target"]].corr(), annot=True, fmt=".2f", ax=ax)
    plt.show()



## === cell 15
train["f_27"].str.len().value_counts()



## === cell 16
if DEBUG_EDA:
    print("No. of unique strings")
    print(train["f_27"].nunique())
    print()

    print("Difference between train and test")
    print(len(set(test["f_27"]).difference(set(train["f_27"]))))



## === cell 17
if DEBUG_EDA:
    print("Top 20 frequent strings")
    train.f_27.value_counts()[:20].sort_values().plot(
        kind="barh", figsize=(15, 15), colormap="Paired"
    )



## === cell 18
if DEBUG_EDA:
    for charind in range(10):
        print(f"Position {charind + 1}:")
        char_group = train.groupby(train["f_27"].str.get(charind))
        char_info = pd.DataFrame(
            {"Length": char_group.size(), "Prob": char_group.target.mean().round(2)}
        )
        print(char_info)
        print()




## === cell 19
def _add_f27_features_from_ascii_bytes(
    df: pd.DataFrame, col: str = "f_27", n_chars: int = 10
) -> None:
    s = df[col].astype("string").fillna("").to_numpy()

    b = np.asarray(s, dtype=f"S{n_chars}")
    arr = b.view(np.uint8).reshape(-1, n_chars)

    codes = arr.astype(np.int16) - 65
    valid = (arr >= 65) & (arr <= 90)
    codes[~valid] = -1

    for i in range(n_chars):
        df[f"char_{i}"] = codes[:, i].astype(np.int16, copy=False)

    mat2 = np.where(codes >= 0, codes, 255).astype(np.int16, copy=False)  # sentinel 255
    mat2.sort(axis=1)
    diffs = mat2[:, 1:] != mat2[:, :-1]
    uniq_cnt = diffs.sum(axis=1).astype(np.int16) + 1
    has_sentinel = (mat2[:, -1] == 255).astype(np.int16)
    uniq_cnt = uniq_cnt - has_sentinel
    df["unique_letters"] = uniq_cnt.astype(np.int16, copy=False)


_add_f27_features_from_ascii_bytes(train, "f_27", 10)
_add_f27_features_from_ascii_bytes(test, "f_27", 10)



## === cell 20
exclude_feats = ["id", "f_27", "target"]
features = [feature for feature in train.columns if feature not in exclude_feats]



## === cell 21
xgb_params = {
    "n_estimators": 8192,
    "min_child_weight": 96,
    "max_depth": 6,
    "learning_rate": 0.15,
    "subsample": 0.95,
    "colsample_bytree": 0.95,
    "reg_lambda": 1.50,
    "reg_alpha": 1.50,
    "gamma": 1.50,
    "max_bin": 512,
    "random_state": 46,
    "objective": "binary:logistic",
    "tree_method": "hist",
    "predictor": "cpu_predictor",
    "eval_metric": "auc",
    "n_jobs": -1,
    "enable_categorical": False,
}



## === cell 22
import xgboost as xgb

X_all = np.ascontiguousarray(train[features].to_numpy(dtype=np.float32, copy=False))
y_all = train["target"].to_numpy(dtype=np.int8, copy=False)
X_test = np.ascontiguousarray(test[features].to_numpy(dtype=np.float32, copy=False))

kf = KFold(n_splits=5, shuffle=True, random_state=46)

scores = []
test_pred_sum = np.zeros(X_test.shape[0], dtype=np.float64)

dtest = xgb.DMatrix(X_test, nthread=-1)

for fold, (train_ind, cv_ind) in enumerate(kf.split(X_all)):
    print("Train fold " + str(fold))

    X_train, y_train = X_all[train_ind], y_all[train_ind]
    X_cv, y_cv = X_all[cv_ind], y_all[cv_ind]

    mdl = XGBClassifier(**xgb_params)

    mdl.fit(
        X_train,
        y_train,
        eval_set=[(X_cv, y_cv)],
        verbose=0,
    )

    booster = mdl.get_booster()
    best_ntree = int(xgb_params["n_estimators"])

    dcv = xgb.DMatrix(X_cv, nthread=-1)
    y_cv_pred = booster.predict(dcv, iteration_range=(0, best_ntree))
    score = roc_auc_score(y_cv, y_cv_pred)

    scores.append(score)
    print(f"Fold {fold}, AUC = {score:.3f}")
    print((""))

    test_pred_sum += booster.predict(dtest, iteration_range=(0, best_ntree)).astype(
        np.float64, copy=False
    )

print("AUC" + str(np.mean(scores)))



## === cell 23
submission = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)
submission.head(5)



## === cell 24
submission["target"] = test_pred_sum / 5.0
submission.to_csv("submission.csv", index=False)



## === cell 25
submission.head(5)
