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

catboost==1.2.8
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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")



## === cell 1
TRAIN_PATH = r"../input/tabular-playground-series-may-2022/train.csv"

train_dtypes = {"id": "int32", "target": "int8", "f_27": "category"}
train_cols = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()
for c in train_cols:
    if c not in train_dtypes and c != "target":
        train_dtypes[c] = "float32"

train = pd.read_csv(
    TRAIN_PATH,
    dtype=train_dtypes,
    engine="c",
    low_memory=False,
)
train.head()



## === cell 2
TEST_PATH = r"../input/tabular-playground-series-may-2022/test.csv"
test_dtypes = {"id": "int32", "f_27": "category"}
test_cols = pd.read_csv(TEST_PATH, nrows=0).columns.tolist()
for c in test_cols:
    if c not in test_dtypes:
        test_dtypes[c] = "float32"

test = pd.read_csv(
    TEST_PATH,
    dtype=test_dtypes,
    engine="c",
    low_memory=False,
)
test.head()



## === cell 3
sub = pd.read_csv(r"../input/tabular-playground-series-may-2022/sample_submission.csv")
sub.head()



## === cell 4
train_id = train["id"].copy()
test_id = test["id"].copy()

train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)



## === cell 5
print(f"train set have {train.shape[0]} rows and {train.shape[1]} columns.")
print(f"test set have {test.shape[0]} rows and {test.shape[1]} columns.")
print(f"sample_submission set have {sub.shape[0]} rows and {sub.shape[1]} columns.")



## === cell 6
train.isnull().sum()



## === cell 7
train.select_dtypes(include=[np.number]).agg(["mean", "std", "min", "max"]).T.head(10)



## === cell 8
pass



## === cell 9
cat = ["f_27"]
X = train.drop("target", axis=1)
y = train["target"]



## === cell 10
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

cat_features_idx = [X.columns.get_loc(c) for c in cat]

folds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof = np.zeros(len(X), dtype=np.float64)
test_pred = np.zeros(len(test), dtype=np.float64)

test_pool = Pool(test, cat_features=cat_features_idx)

for fold, (trn_idx, val_idx) in enumerate(folds.split(X)):
    print(f"Fold: {fold}")

    X_train, X_valid = X.iloc[trn_idx], X.iloc[val_idx]
    y_train, y_valid = y.iloc[trn_idx], y.iloc[val_idx]

    train_pool = Pool(X_train, y_train, cat_features=cat_features_idx)
    valid_pool = Pool(X_valid, y_valid, cat_features=cat_features_idx)

    model = CatBoostClassifier(
        n_estimators=1500,
        cat_features=cat_features_idx,
        task_type="CPU",
        bootstrap_type="Bayesian",
        random_seed=RANDOM_STATE,
        loss_function="Logloss",
        eval_metric="AUC",
        thread_count=-1,
    )

    model.fit(
        train_pool,
        eval_set=valid_pool,
        early_stopping_rounds=400,
        verbose=False,
    )

    y_pred = model.predict_proba(valid_pool)[:, 1]
    oof[val_idx] = y_pred
    roc = roc_auc_score(y_valid, y_pred)
    print(f" roc_auc_score: {roc}")
    print("-" * 50)

    test_pred += model.predict_proba(test_pool)[:, 1] / folds.n_splits

print(f"Overall OOF ROC AUC: {roc_auc_score(y, oof)}")



## === cell 11
pred = test_pred



## === cell 12
sub = sub.copy()
sub["id"] = test_id.values  # keep exact ordering from test read
sub["target"] = pred.astype(float)
sub = sub[["id", "target"]]
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
