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

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, CatBoostError, Pool
import pandas as pd
import numpy as np
import os
import random

pd.set_option("display.max_columns", 200)

RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

N_THREADS = os.cpu_count() or 4

os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(N_THREADS))




## === cell 1
base_dir = "/kaggle/input/tabular-playground-series-may-2022"
_ = base_dir  # keep variable for parity; no-op




## === cell 2
train_path = "/kaggle/input/tabular-playground-series-may-2022/train.csv"
test_path = "/kaggle/input/tabular-playground-series-may-2022/test.csv"
sample_path = "/kaggle/input/tabular-playground-series-may-2022/sample_submission.csv"

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()

feature_cols = [c for c in train_cols if c not in ("id", "target")]
test_col_set = set(test_cols)
feature_cols = [c for c in feature_cols if c in test_col_set]

categorical_columns = [f"f_{i:02d}" for i in range(7, 19)] + ["f_27", "f_29", "f_30"]
categorical_columns = [c for c in categorical_columns if c in feature_cols]

train_usecols = ["id", "target"] + feature_cols
test_usecols = ["id"] + feature_cols

dtype_train = {"id": "int32", "target": "int8"}
dtype_test = {"id": "int32"}

cat_set = set(categorical_columns)
for c in feature_cols:
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
print("First train cols:", train_df.columns[:12].tolist(), "...")

X_df = train_df[feature_cols]
X_test_df = test_df[feature_cols]
y = train_df["target"].to_numpy(dtype=np.int32, copy=False)

cat_features_for_pool = [feature_cols.index(c) for c in categorical_columns]

print("Num features:", len(feature_cols), "| Cat features:", len(cat_features_for_pool))
print("Example categorical cols:", categorical_columns[:10])




## === cell 3
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
train_idx, valid_idx = next(sss.split(np.zeros(len(y), dtype=np.uint8), y))

X_train_df = X_df.iloc[train_idx]
X_valid_df = X_df.iloc[valid_idx]
y_train = y[train_idx]
y_valid = y[valid_idx]

train_pool = Pool(X_train_df, y_train, cat_features=cat_features_for_pool)
valid_pool = Pool(X_valid_df, y_valid, cat_features=cat_features_for_pool)




## === cell 4
params = dict(
    n_estimators=7000,
    learning_rate=0.10315154739037707,
    depth=2,
    l2_leaf_reg=1,
    verbose=300,
    random_seed=RANDOM_STATE,
    allow_writing_files=False,
    pinned_memory_size=512 * 1024 * 1024,
)

fit_kwargs = dict(
    eval_set=valid_pool,
    use_best_model=True,
)

try:
    model = CatBoostClassifier(task_type="GPU", thread_count=N_THREADS, **params)
    model.fit(train_pool, **fit_kwargs)
except CatBoostError as e:
    print("GPU training failed, falling back to CPU. Error was:")
    print(str(e)[:500])
    model = CatBoostClassifier(task_type="CPU", thread_count=N_THREADS, **params)
    model.fit(train_pool, **fit_kwargs)




## === cell 5
y_valid_np = np.asarray(y_valid, dtype=np.int32)

valid_pred_proba = model.predict_proba(valid_pool)[:, 1]
valid_auc = roc_auc_score(y_valid_np, valid_pred_proba)
print(f"Validation ROC AUC: {valid_auc:.6f}")




## === cell 6
test_pool = Pool(X_test_df, cat_features=cat_features_for_pool)

test_pred_proba = model.predict_proba(test_pool)[:, 1]
test_pred_proba = np.clip(np.asarray(test_pred_proba, dtype=float), 0.0, 1.0)

submission = pd.DataFrame(
    {"id": test_df["id"].to_numpy(copy=False), "target": test_pred_proba}
)

if "id" in sample_sub.columns and len(sample_sub) == len(submission):
    submission = sample_sub.merge(submission, on="id", how="left", sort=False)

assert list(submission.columns) == ["id", "target"]
assert submission["target"].notna().all()

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
