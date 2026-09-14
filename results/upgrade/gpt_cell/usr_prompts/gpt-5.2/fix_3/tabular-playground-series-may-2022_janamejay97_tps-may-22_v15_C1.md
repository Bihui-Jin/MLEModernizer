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
import os
import numpy as np
import pandas as pd
import warnings

from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

np.random.seed(46)





## === cell 1
TRAIN_PATH = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-may-2022/test.csv"

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape)
print("Test shape:", test.shape)




## === cell 2
pass




## === cell 3

f27_train = train["f_27"].astype(str)
f27_test = test["f_27"].astype(str)

for ite in range(10):
    c_tr = f27_train.str.slice(ite, ite + 1)
    c_te = f27_test.str.slice(ite, ite + 1)

    train[f"char_{ite}"] = (c_tr.map(ord).astype(np.int16) - ord("A")).astype(np.int16)
    test[f"char_{ite}"] = (c_te.map(ord).astype(np.int16) - ord("A")).astype(np.int16)


def _unique_letters_count(arr_str: np.ndarray) -> np.ndarray:
    chars = np.stack(
        [np.char.array(arr_str).str_slice(i, i + 1) for i in range(10)], axis=1
    )
    chars_sorted = np.sort(chars, axis=1)
    changes = (chars_sorted[:, 1:] != chars_sorted[:, :-1]).sum(axis=1)
    return (changes + 1).astype(np.int16)


train["unique_letters"] = _unique_letters_count(f27_train.to_numpy())
test["unique_letters"] = _unique_letters_count(f27_test.to_numpy())




## === cell 4
exclude_feats = ["id", "f_27", "target"]
features = [c for c in train.columns if c not in exclude_feats]

X = train[features].to_numpy()
y = train["target"].to_numpy()
X_test = test[features].to_numpy()




## === cell 5
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
    "tree_method": "gpu_hist",
}

xgb_params.setdefault("nthread", os.cpu_count() or 4)




## === cell 6
scores, predictions = [], []

kf = KFold(n_splits=5)

for fold, (train_ind, cv_ind) in enumerate(kf.split(X)):
    print("Train fold " + str(fold))

    X_train, y_train = X[train_ind], y[train_ind]
    X_cv, y_cv = X[cv_ind], y[cv_ind]

    xgb_params_run = dict(xgb_params)
    xgb_params_run["tree_method"] = (
        "hist"  # keep identical to original runtime override
    )

    mdl = XGBClassifier(**xgb_params_run)

    mdl.fit(
        X_train,
        y_train,
        eval_set=[(X_cv, y_cv)],
        eval_metric=["auc"],
        early_stopping_rounds=256,
        verbose=0,
    )

    y_cv_pred = mdl.predict_proba(X_cv)[:, 1]
    score = roc_auc_score(y_cv, y_cv_pred)

    scores.append(score)
    print(f"Fold {fold}, AUC = {score:.3f}")
    print("")

    test_pred = mdl.predict_proba(X_test)[:, 1]
    predictions.append(test_pred)

print("AUC" + str(np.mean(scores)))




## === cell 7
submission = pd.read_csv(
    "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"
)
submission["target"] = np.mean(np.vstack(predictions), axis=0)
submission.to_csv("submission.csv", index=False)
submission.head(5)
