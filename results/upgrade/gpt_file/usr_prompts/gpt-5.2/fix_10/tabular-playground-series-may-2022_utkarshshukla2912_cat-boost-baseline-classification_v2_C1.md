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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
from sklearnex import patch_sklearn

patch_sklearn()

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from catboost import CatBoostClassifier, CatBoostError, Pool
import pandas as pd
import numpy as np
import os
import random

pd.set_option("display.max_columns", 200)

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)



## === cell 1
base_dir = "/kaggle/input/tabular-playground-series-may-2022"
_ = base_dir  # keep variable for parity; no-op



## === cell 2
train_path = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
test_path = "/kaggle/input/tabular-playground-series-may-2022/test.csv"
sample_path = "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"

all_feature_cols = [f"f_{i:02d}" for i in range(0, 31)]
train_usecols = ["id", "target"] + all_feature_cols
test_usecols = ["id"] + all_feature_cols

categorical_columns = [f"f_{i:02d}" for i in range(7, 19)] + ["f_27", "f_29", "f_30"]

dtype_train = {"id": "int32", "target": "int8"}
dtype_test = {"id": "int32"}

cat_set = set(categorical_columns)
for c in all_feature_cols:
    if c in cat_set:
        dtype_train[c] = "category"
        dtype_test[c] = "category"
    else:
        dtype_train[c] = "float32"
        dtype_test[c] = "float32"

train_df = pd.read_csv(train_path, usecols=train_usecols, dtype=dtype_train)
test_df = pd.read_csv(test_path, usecols=test_usecols, dtype=dtype_test)
sample_sub = pd.read_csv(sample_path, usecols=["id"], dtype={"id": "int32"})

print(train_df.shape, test_df.shape, sample_sub.shape)
print("train cols:", train_df.columns[:10].tolist(), "...")

categorical_columns = [c for c in categorical_columns if c in train_df.columns]

X_df = train_df[all_feature_cols].copy()
X_test_df = test_df[all_feature_cols].copy()

feat_idx = {c: i for i, c in enumerate(all_feature_cols)}
cat_features_for_pool = [feat_idx[c] for c in categorical_columns]

for c in categorical_columns:
    X_df[c] = X_df[c].cat.codes.astype(np.int32, copy=False)
    X_test_df[c] = X_test_df[c].cat.codes.astype(np.int32, copy=False)

X_all = np.ascontiguousarray(X_df.to_numpy(copy=False))
X_test_final = np.ascontiguousarray(X_test_df.to_numpy(copy=False))

y = train_df["target"].to_numpy(dtype=np.int32, copy=False)



## === cell 3
x_train_df, x_valid_df, y_train, y_valid = train_test_split(
    X_df, y, test_size=0.1, random_state=RANDOM_STATE, stratify=y
)



## === cell 4
params = dict(
    n_estimators=7000,
    learning_rate=0.10315154739037707,
    depth=2,
    l2_leaf_reg=1,
    verbose=300,
    random_seed=RANDOM_STATE,
    allow_writing_files=False,
)

train_pool = Pool(x_train_df, y_train, cat_features=cat_features_for_pool)
valid_pool = Pool(x_valid_df, y_valid, cat_features=cat_features_for_pool)

fit_kwargs = dict(
    eval_set=valid_pool,
    use_best_model=False,
    early_stopping_rounds=200,  # keep semantics as provided
)

try:
    model = CatBoostClassifier(task_type="GPU", **params)
    model.fit(train_pool, **fit_kwargs)
except CatBoostError as e:
    print("GPU training failed, falling back to CPU. Error was:")
    print(str(e)[:500])
    model = CatBoostClassifier(task_type="CPU", thread_count=os.cpu_count(), **params)
    model.fit(train_pool, **fit_kwargs)



## === cell 5
y_valid_np = np.asarray(y_valid, dtype=np.int32)

valid_pred_proba = model.predict_proba(valid_pool)[:, 1]
valid_auc = roc_auc_score(y_valid_np, valid_pred_proba)
print(f"Validation ROC AUC: {valid_auc:.6f}")

valid_pred_label = (valid_pred_proba >= 0.5).astype(np.int32, copy=False)
print(classification_report(y_valid_np, valid_pred_label))



## === cell 6
test_pool = Pool(X_test_df, cat_features=cat_features_for_pool)
test_pred_proba = model.predict_proba(test_pool)[:, 1]
test_pred_proba = np.clip(np.asarray(test_pred_proba, dtype=float), 0.0, 1.0)



## === cell 7
submission = pd.DataFrame(
    {"id": test_df["id"].to_numpy(copy=False), "target": test_pred_proba}
)

if "id" in sample_sub.columns and len(sample_sub) == len(submission):
    submission = sample_sub.merge(submission, on="id", how="left", sort=False)

assert list(submission.columns) == ["id", "target"]
assert submission["target"].notna().all()

submission.head()



## === cell 8
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
