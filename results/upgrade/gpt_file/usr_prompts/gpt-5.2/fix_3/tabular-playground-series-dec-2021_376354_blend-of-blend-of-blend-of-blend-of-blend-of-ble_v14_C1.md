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

# 5. Target score

0.9565828571428572

# 6. Current score

0.68275

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.68275) has done: 'I fix the failure in the stratified split by guarding against rare classes (which can happen if `Cover_Type` is not cleanly parsed as integers 1–7, or if the dataset variant has odd labels), falling back to a non-stratified split only for the sanity-check. I also make cell execution robust by ensuring `clf` is always defined even if the sanity-check split is skipped/altered, so the full-train fit and submission generation always run. Finally, I keep the model/pipeline core logic identical and ensure the script writes a valid `submission_final.csv` with the required columns and row count.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier

RANDOM_STATE = 42

DATA_DIR = "../input/tabular-playground-series-dec-2021"
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

if not os.path.exists(TRAIN_PATH):
    DATA_DIR = "../input"
    TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
    TEST_PATH = os.path.join(DATA_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_PATH), f"train.csv not found at {TRAIN_PATH}"
assert os.path.exists(TEST_PATH), f"test.csv not found at {TEST_PATH}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"sample_submission.csv not found at {SAMPLE_SUB_PATH}"



## === cell 1
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

assert "Cover_Type" in train.columns, "Target column Cover_Type missing from train.csv"
assert "Id" in train.columns and "Id" in test.columns, "Id column missing"
assert list(sample_sub.columns) == ["Id", "Cover_Type"], "Unexpected submission columns"

X = train.drop(columns=["Cover_Type"])
y = train["Cover_Type"]

y = pd.to_numeric(y, errors="coerce").astype("Int64")
if y.isna().any():
    raise ValueError(
        f"Found NaN/invalid labels in Cover_Type after conversion: {int(y.isna().sum())} rows"
    )
y = y.astype(int)

unique_labels = np.sort(y.unique())
print(
    "Unique Cover_Type labels:",
    unique_labels[:20],
    "..." if unique_labels.size > 20 else "",
)
assert (
    unique_labels.min() >= 1 and unique_labels.max() <= 7
), "Unexpected label range; expected 1..7"

feature_cols = [c for c in X.columns if c != "Id"]
X_train_full = X[feature_cols]
X_test_full = test[feature_cols]



## === cell 2
clf = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        (
            "model",
            SGDClassifier(
                loss="log_loss",
                alpha=1e-5,
                penalty="l2",
                max_iter=25,
                tol=1e-3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ]
)

class_counts = y.value_counts()
min_count = int(class_counts.min())
can_stratify = min_count >= 2

X_tr, X_va, y_tr, y_va = train_test_split(
    X_train_full,
    y,
    test_size=0.05,
    random_state=RANDOM_STATE,
    stratify=y if can_stratify else None,
)

clf.fit(X_tr, y_tr)
va_pred = clf.predict(X_va)
va_acc = accuracy_score(y_va, va_pred)
print(
    f"Validation accuracy (sanity check, stratify={can_stratify}, min_class_count={min_count}): {va_acc:.6f}"
)



## === cell 3
clf.fit(X_train_full, y)

test_pred = clf.predict(X_test_full).astype(int)

submission = sample_sub.copy()
submission["Id"] = test["Id"].values
submission["Cover_Type"] = test_pred

assert submission.shape[0] == test.shape[0], "Submission row count mismatch"
assert list(submission.columns) == [
    "Id",
    "Cover_Type",
], "Submission must have columns: Id, Cover_Type"
assert submission["Id"].isna().sum() == 0, "Found NaN Ids in submission"
assert (
    submission["Cover_Type"].between(1, 7).all()
), "Cover_Type out of expected range 1..7"

submission.to_csv("submission_final.csv", index=False)
print(submission.head())
print("Wrote submission_final.csv")
