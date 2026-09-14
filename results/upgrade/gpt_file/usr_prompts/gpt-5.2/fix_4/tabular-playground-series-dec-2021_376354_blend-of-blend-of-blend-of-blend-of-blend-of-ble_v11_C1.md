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
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

np.random.seed(42)




## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/tabular-playground-series-dec-2021",
    "/kaggle/data/tabular-playground-series-dec-2021",
    "/kaggle/input",
    "/kaggle/data",
]


def first_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


base_dir = None
for cand in BASE_CANDIDATES:
    train_path = os.path.join(cand, "train.csv")
    test_path = os.path.join(cand, "test.csv")
    sample_path = os.path.join(cand, "sample_submission.csv")
    if (
        os.path.exists(train_path)
        and os.path.exists(test_path)
        and os.path.exists(sample_path)
    ):
        base_dir = cand
        break

if base_dir is None:
    raise FileNotFoundError(
        "Could not find train.csv/test.csv/sample_submission.csv in expected /kaggle paths."
    )

TRAIN_PATH = os.path.join(base_dir, "train.csv")
TEST_PATH = os.path.join(base_dir, "test.csv")
SAMPLE_PATH = os.path.join(base_dir, "sample_submission.csv")

TRAIN_PATH, TEST_PATH, SAMPLE_PATH




## === cell 2
_train_head = pd.read_csv(TRAIN_PATH, nrows=1)
feature_cols_all = [c for c in _train_head.columns if c not in ("Cover_Type",)]
feature_cols = [c for c in feature_cols_all if c != "Id"]

dtype_train = {c: np.int32 for c in feature_cols_all}
dtype_train["Id"] = np.int32
dtype_train["Cover_Type"] = np.int8

dtype_test = {c: np.int32 for c in (["Id"] + feature_cols)}
dtype_test["Id"] = np.int32

train = pd.read_csv(TRAIN_PATH, dtype=dtype_train)
test = pd.read_csv(TEST_PATH, dtype=dtype_test)
submission = pd.read_csv(SAMPLE_PATH, usecols=["Id"])

assert "Cover_Type" in train.columns
assert "Id" in train.columns and "Id" in test.columns and "Id" in submission.columns

X = train.drop(columns=["Cover_Type"])
y_raw = train["Cover_Type"].astype(int)

X_test = test.copy()

X_feat = X[feature_cols]
X_test_feat = X_test[feature_cols]

y = (y_raw - 1).astype(np.int64)  # 0..6

try:
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_feat, y, test_size=0.02, random_state=42, stratify=y
    )
except ValueError:
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_feat, y, test_size=0.02, random_state=42, shuffle=True
    )

X_feat.shape, X_test_feat.shape, y_raw.value_counts().sort_index()




## === cell 3
scaler = StandardScaler(with_mean=False)

X_tr_s = scaler.fit_transform(X_tr)
X_va_s = scaler.transform(X_va)
X_test_s = scaler.transform(X_test_feat)


def fit_predict_models(X_train_s, y_train, X_valid_s, X_test_s):
    models = []

    models.append(
        (
            "lr_lbfgs",
            LogisticRegression(
                max_iter=200,
                n_jobs=-1,
                multi_class="auto",
                solver="lbfgs",
                C=2.0,
                random_state=42,
            ),
        )
    )

    models.append(
        (
            "linearsvc",
            LinearSVC(C=1.5, random_state=42),
        )
    )

    models.append(
        (
            "knn",
            KNeighborsClassifier(
                n_neighbors=35,
                weights="distance",
                metric="minkowski",
                p=2,
                n_jobs=-1,
            ),
        )
    )

    models.append(
        (
            "gnb",
            GaussianNB(),
        )
    )

    valid_preds = {}
    test_preds = {}

    for name, model in models:
        model.fit(X_train_s, y_train)
        pv = model.predict(X_valid_s).astype(np.int64)
        pt = model.predict(X_test_s).astype(np.int64)
        valid_preds[name] = pv
        test_preds[name] = pt

    return valid_preds, test_preds


valid_preds, test_preds = fit_predict_models(X_tr_s, y_tr, X_va_s, X_test_s)

for k, pv in valid_preds.items():
    acc = accuracy_score(y_va, pv)
    print(f"{k}: val acc={acc:.5f}")




## === cell 4
def row_mode_int_0based(arr2d: np.ndarray, n_classes: int) -> np.ndarray:
    """
    arr2d: (n_samples, n_models) integer predictions in [0, n_classes-1].
    returns: (n_samples,) integer mode; ties broken by smallest class label (deterministic).
    """
    arr2d = np.asarray(arr2d, dtype=np.int64)
    n, m = arr2d.shape
    counts = np.bincount(
        (arr2d + (np.arange(n, dtype=np.int64)[:, None] * n_classes)).ravel(),
        minlength=n * n_classes,
    ).reshape(n, n_classes)
    return counts.argmax(axis=1).astype(np.int64)


pred_cols = sorted(test_preds.keys())
pred_matrix = np.vstack(
    [test_preds[c] for c in pred_cols]
).T  # (n_samples, n_models), 0-based

ensemble_pred_0 = row_mode_int_0based(pred_matrix, n_classes=7)
ensemble_pred = (ensemble_pred_0 + 1).astype(int)  # decode back to 1..7 for submission

sub = submission.copy()
sub["Cover_Type"] = ensemble_pred

sub = sub[["Id", "Cover_Type"]]
assert len(sub) == len(test)
assert sub["Cover_Type"].between(1, 7).all()

sub.to_csv("submission.csv", index=False)
sub.head()
