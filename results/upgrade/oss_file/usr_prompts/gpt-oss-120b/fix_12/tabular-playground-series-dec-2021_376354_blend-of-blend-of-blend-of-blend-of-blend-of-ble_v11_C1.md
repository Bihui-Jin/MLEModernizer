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

os.environ["OMP_NUM_THREADS"] = "8"  # keep full CPU parallelism for scikit‑learn

import glob
import numpy as np
import pandas as pd
import gc
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score


def find_file(pattern):
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        raise FileNotFoundError(f"No file matches pattern: {pattern}")
    return matches[0]


train_path = find_file("/kaggle/input/**/train.csv")
test_path = find_file("/kaggle/input/**/test.csv")
sample_sub_path = find_file("/kaggle/input/**/sample_submission.csv")

all_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
feature_cols = [c for c in all_cols if c not in ("Id", "Cover_Type")]

dtype_map = {col: np.float32 for col in feature_cols}
dtype_map.update({"Id": np.int32, "Cover_Type": np.int8})

train_df = pd.read_csv(
    train_path, usecols=feature_cols + ["Id", "Cover_Type"], dtype=dtype_map, engine="c"
)

test_dtype_map = {col: np.float32 for col in feature_cols}
test_dtype_map["Id"] = np.int32

test_df = pd.read_csv(
    test_path, usecols=feature_cols + ["Id"], dtype=test_dtype_map, engine="c"
)

test_ids = test_df[["Id"]].astype(np.int32)

sample_submission = pd.read_csv(sample_sub_path)

gc.collect()



## === cell 1
X = train_df[feature_cols].to_numpy(
    copy=False
)  # shape (n_samples, n_features), float32
y = train_df["Cover_Type"].to_numpy(copy=False)  # shape (n_samples,), int8
X_test = test_df[feature_cols].to_numpy(
    copy=False
)  # shape (n_test, n_features), float32

del train_df, test_df
gc.collect()



## === cell 2
rng = np.random.RandomState(42)
perm = rng.permutation(X.shape[0])
split_idx = int(0.8 * X.shape[0])
train_idx, val_idx = perm[:split_idx], perm[split_idx:]

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

X_full = X
y_full = y

del perm
gc.collect()

model = HistGradientBoostingClassifier(
    max_iter=400,
    learning_rate=0.1,
    max_depth=None,
    random_state=42,
    early_stopping=False,
    verbose=0,
)

model.fit(X_train, y_train)

val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation accuracy: {val_acc:.5f}")

final_model = HistGradientBoostingClassifier(
    max_iter=400,
    learning_rate=0.1,
    max_depth=None,
    random_state=42,
    early_stopping=False,
    verbose=0,
)

final_model.fit(X_full, y_full)

test_pred = final_model.predict(X_test)

submission = pd.DataFrame({"Id": test_ids["Id"], "Cover_Type": test_pred})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(submission.head())

del X_train, X_val, y_train, y_val, X_test, test_ids, X_full, y_full, model, final_model
gc.collect()
