# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.91417

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)



## === cell 1
from pathlib import Path

CANDIDATE_COMP_DIRS = [
    Path("/kaggle/input/tabular-playground-series-may-2022"),
    Path("/kaggle/data/tabular-playground-series-may-2022"),
    Path("/kaggle/working/tabular-playground-series-may-2022"),
]

FALLBACK_BASES = [
    Path("/kaggle/input"),
    Path("/kaggle/data"),
    Path("/kaggle/working"),
]


def _find_in_dir(base: Path, filename: str) -> str | None:
    p = base / filename
    return str(p) if p.exists() else None


def _find_file_grouped() -> tuple[str, str, str]:
    for comp_dir in CANDIDATE_COMP_DIRS:
        tr = _find_in_dir(comp_dir, "train.csv")
        te = _find_in_dir(comp_dir, "test.csv")
        su = _find_in_dir(comp_dir, "sample_submission.csv")
        if tr and te and su:
            return tr, te, su

    def _find_any(filename: str) -> str:
        for base in FALLBACK_BASES:
            p = base / filename
            if p.exists():
                return str(p)
        root = Path("/kaggle")
        if root.exists():
            hits = list(root.rglob(filename))
            if hits:
                return str(hits[0])
        raise FileNotFoundError(
            f"Could not locate {filename} under known Kaggle paths."
        )

    return (
        _find_any("train.csv"),
        _find_any("test.csv"),
        _find_any("sample_submission.csv"),
    )


train_path, test_path, sub_path = _find_file_grouped()

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)
sample_submission = pd.read_csv(sub_path)

print("Resolved paths:")
print("train:", train_path)
print("test :", test_path)
print("sub  :", sub_path)
print("Shapes:", train_data.shape, test_data.shape, sample_submission.shape)



## === cell 2
test_ids = test_data["id"].copy()

train_y = train_data["target"].astype(int)

train_X_df = train_data.drop(columns=["id", "target"])
test_X_df = test_data.drop(columns=["id"])



## === cell 3
train_X_df = pd.get_dummies(train_X_df, columns=["f_27"], drop_first=False)
test_X_df = pd.get_dummies(test_X_df, columns=["f_27"], drop_first=False)

test_X_df = test_X_df.reindex(columns=train_X_df.columns, fill_value=0)



## === cell 4
from sklearn import preprocessing

train_X_df = train_X_df.astype(np.float32)
test_X_df = test_X_df.astype(np.float32)

scaler = preprocessing.StandardScaler().fit(train_X_df)
train_X = scaler.transform(train_X_df).astype(np.float32, copy=False)
test_X = scaler.transform(test_X_df).astype(np.float32, copy=False)



## === cell 5
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X_tr, X_va, y_tr, y_va = train_test_split(
    train_X, train_y, test_size=0.1, random_state=42, stratify=train_y
)



## === cell 6
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    eval_metric="auc",
    tree_method="hist",
    missing=np.nan,
    min_child_weight=1,
    gamma=0.0,
    use_label_encoder=False,
)

model.fit(X_tr, y_tr, verbose=False)
va_pred = model.predict_proba(X_va)[:, 1]
print("Holdout AUC (sanity check):", roc_auc_score(y_va, va_pred))



## === cell 7
model.fit(train_X, train_y, verbose=False)



## === cell 8
predictions = model.predict_proba(test_X)[:, 1]
predictions = np.clip(predictions, 0.0, 1.0)

sub = sample_submission[["id"]].copy()

if len(sub) != len(test_ids):
    raise ValueError(
        f"Row count mismatch: sample_submission={len(sub)} vs test={len(test_ids)}"
    )

test_id_to_pred = pd.Series(predictions, index=test_ids.values)
sub["target"] = sub["id"].map(test_id_to_pred)

missing = sub["target"].isna().sum()
if missing != 0:
    print(
        f"WARNING: {missing} ids in sample_submission not found in test predictions; filling with 0.5"
    )
    sub["target"] = sub["target"].fillna(0.5)

sub["target"] = sub["target"].astype(float)

assert list(sub.columns) == ["id", "target"]
assert sub["target"].between(0.0, 1.0).all()
assert sub.isna().sum().sum() == 0

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print("Any missing targets:", sub["target"].isna().sum())
