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
import os
import random
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



## === cell 1
DATA_DIR = r"../input/tabular-playground-series-may-2022"

feature_cols = [f"f_{i:02d}" for i in range(31)]
train_cols = ["id"] + feature_cols + ["target"]
test_cols = ["id"] + feature_cols

dtype_map = {c: "float32" for c in feature_cols}
dtype_map["f_27"] = "string"
dtype_train = dtype_map | {"target": "int8", "id": "int32"}
dtype_test = dtype_map | {"id": "int32"}

train = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=train_cols, dtype=dtype_train)
test = pd.read_csv(f"{DATA_DIR}/test.csv", usecols=test_cols, dtype=dtype_test)
sub = pd.read_csv(
    f"{DATA_DIR}/sample_submission.csv", dtype={"id": "int32", "target": "float32"}
)



## === cell 2
train.drop("id", axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)



## === cell 3
print(f"train set have {train.shape[0]} rows and {train.shape[1]} columns.")
print(f"test set have {test.shape[0]} rows and {test.shape[1]} columns.")
print(f"sample_submission set have {sub.shape[0]} rows and {sub.shape[1]} columns.")



## === cell 4
pass



## === cell 5
cat = ["f_27"]
X = train.drop("target", axis=1)
y = train["target"]



## === cell 6
from sklearn.model_selection import KFold
from catboost import CatBoostClassifier, Pool, cv
from sklearn.metrics import roc_auc_score

folds = KFold(n_splits=5, shuffle=True, random_state=SEED)
idx = np.arange(len(X))

full_pool = Pool(X, y, cat_features=cat)

thread_count = os.cpu_count() or 1

params = dict(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=1500,  # same as n_estimators
    random_seed=SEED,
    task_type="CPU",
    thread_count=thread_count,
    allow_writing_files=False,
)

cv_results = cv(
    pool=full_pool,
    params=params,
    fold_count=5,
    partition_random_seed=SEED,
    shuffle=True,
    early_stopping_rounds=400,  # same early stopping
    verbose=False,
)

best_iter = int(np.argmax(cv_results["test-AUC-mean"]) + 1)
best_auc = float(np.max(cv_results["test-AUC-mean"]))
print(f"CV best iteration: {best_iter} | CV test-AUC-mean: {best_auc}")

model = CatBoostClassifier(
    n_estimators=best_iter,
    cat_features=cat,
    task_type="CPU",
    random_seed=SEED,
    thread_count=thread_count,
    allow_writing_files=False,
)
model.fit(full_pool, verbose=False)


## === cell 7
test_pool = Pool(test, cat_features=cat)
pred = model.predict_proba(test_pool)[:, 1]



## === cell 8
sub["target"] = pred
sub.to_csv("cat.csv", index=False)



## === cell 9
sub.head()
