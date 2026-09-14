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
import pandas as pd
import numpy as np

import sklearn.model_selection as skl_ms
from catboost import CatBoostClassifier

RANDOM_STATE = 42
os.environ.setdefault("PYTHONHASHSEED", str(RANDOM_STATE))



## === cell 1
DATA_DIR = "../input/tabular-playground-series-dec-2021"
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_cols = pd.read_csv(train_path, nrows=1).columns.tolist()
test_cols = pd.read_csv(test_path, nrows=1).columns.tolist()

train_dtypes = {}
for c in train_cols:
    if c == "Id":
        train_dtypes[c] = np.int32
    elif c == "Cover_Type":
        train_dtypes[c] = np.int8
    else:
        train_dtypes[c] = np.int32

test_dtypes = {}
for c in test_cols:
    if c == "Id":
        test_dtypes[c] = np.int32
    else:
        test_dtypes[c] = np.int32

train_df = pd.read_csv(train_path, dtype=train_dtypes)
test_df = pd.read_csv(test_path, dtype=test_dtypes)

train_df["Cover_Type"] = train_df["Cover_Type"].astype(np.int8)

print("Train shape:", train_df.shape, " Test shape:", test_df.shape)
print("Cover_Type unique:", np.sort(train_df["Cover_Type"].unique()))
print("Min class count:", train_df["Cover_Type"].value_counts().min())



## === cell 2
test_size = 0.01  # keep identical size as original intent

X = train_df.drop(columns=["Cover_Type"])
y = train_df["Cover_Type"]  # keep as pandas Series of {1..7} for stratify

try:
    X_train, X_valid, y_train, y_valid = skl_ms.train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=y,
    )
except ValueError as e:
    print(
        "WARNING: Stratified split failed, falling back to non-stratified split. Error:",
        repr(e),
    )
    X_train, X_valid, y_train, y_valid = skl_ms.train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=None,
    )

print("Split shapes:", X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
print("Valid class counts:")
print(y_valid.value_counts().sort_index())



## === cell 3
model = CatBoostClassifier(
    iterations=5000,
    task_type="CPU",
    random_seed=RANDOM_STATE,
    verbose=False,
    thread_count=-1,
    loss_function="MultiClass",
    eval_metric="Accuracy",
    use_best_model=True,
)

model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False)



## === cell 4
accuracy = model.score(X_valid, y_valid)
print(f"Accuracy of catboost on validation data: {accuracy:.6f}")



## === cell 5
subm_df = pd.read_csv(sub_path)

preds = model.predict(test_df)
preds = np.asarray(preds).reshape(-1).astype(int)

subm_df = subm_df.iloc[: len(test_df)].copy()
subm_df["Id"] = test_df["Id"].to_numpy()
subm_df["Cover_Type"] = preds.astype(int)
subm_df = subm_df[["Id", "Cover_Type"]]

out_path = "submission_cb.csv"
subm_df.to_csv(out_path, index=False)
print(f"Wrote submission to: {out_path} with shape {subm_df.shape}")
print("Submission Cover_Type value counts (head):")
print(subm_df["Cover_Type"].value_counts().head(10))
