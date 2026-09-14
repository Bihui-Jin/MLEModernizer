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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
import pandas as pd
import numpy as np
import os
from pathlib import Path

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

np.random.seed(42)




## === cell 1
def resolve_path(*parts):
    """
    Return a string path to the file described by *parts.
    Tries several common locations and finally walks the current
    directory tree to find the file by name.
    """
    candidate_paths = [
        Path(*parts),  # absolute / relative as given
        Path.cwd() / Path(*parts),  # cwd / given relative path
    ]

    base_dirs = [
        Path("/") / "kaggle" / "input",
        Path.cwd() / "kaggle" / "input",
    ]
    for base in base_dirs:
        candidate_paths.append(
            base / Path(*parts[2:])
        )  # skip leading 'kaggle', 'input'

    for p in candidate_paths:
        if p.is_file():
            return str(p)

    filename = parts[-1]
    for root, _, files in os.walk(Path.cwd()):
        if filename in files:
            return str(Path(root) / filename)

    raise FileNotFoundError(f"Could not find file: {Path(*parts)}")


train_path = resolve_path(
    "kaggle", "input", "tabular-playground-series-may-2022", "train.csv"
)
test_path = resolve_path(
    "kaggle", "input", "tabular-playground-series-may-2022", "test.csv"
)

numeric_cols = [f"f_{i:02d}" for i in range(0, 7)] + [
    f"f_{i}" for i in range(19, 29) if i != 27
]
dtype_spec = {col: np.float32 for col in numeric_cols}

train_df = pd.read_csv(train_path, dtype=dtype_spec)
test_df = pd.read_csv(test_path, dtype=dtype_spec)

categorical_columns = [f"f_{i:02d}" for i in range(7, 19)] + ["f_27", "f_29", "f_30"]




## === cell 2
cat_feature_indices = [
    train_df.drop(["id", "target"], axis=1).columns.get_loc(col)
    for col in categorical_columns
]




## === cell 3
X = train_df.drop(["id", "target"], axis=1)
y = train_df["target"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

max_threads = os.cpu_count() or 1

model = CatBoostClassifier(
    cat_features=cat_feature_indices,
    n_estimators=1200,
    learning_rate=0.10315154739037707,
    depth=6,
    l2_leaf_reg=1,
    task_type="CPU",
    thread_count=max_threads,
    verbose=False,
    loss_function="Logloss",
    eval_metric="AUC",  # monitor AUC on validation
    random_seed=42,
)

train_pool = Pool(data=X_train, label=y_train, cat_features=cat_feature_indices)
val_pool = Pool(data=X_val, label=y_val, cat_features=cat_feature_indices)




## === cell 4
model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=False)

val_pred = model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_pred)
print(f"Validation AUC: {val_auc:.5f}")

del train_pool, val_pool, train_df, X, y, X_train, X_val, y_train, y_val
import gc

gc.collect()




## === cell 5
print("Model training completed.")




## === cell 6
test_features = test_df.drop(["id"], axis=1)
test_proba = model.predict_proba(test_features)[:, 1]

submission = pd.DataFrame({"id": test_df["id"], "target": test_proba})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
