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
xgboost==2.0.3

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

np.random.seed(42)



## === cell 1
TRAIN_PATH = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
TEST_PATH = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

_cols = pd.read_csv(TRAIN_PATH, nrows=0).columns.tolist()

target_col = "Cover_Type"
id_col = "Id"
feature_cols = [c for c in _cols if c not in (target_col, id_col)]

bin_cols = [
    c
    for c in feature_cols
    if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
]

dtype_train = {id_col: "int64", target_col: "int8"}
dtype_test = {id_col: "int64"}
for c in feature_cols:
    if c in bin_cols:
        dtype_train[c] = "int8"
        dtype_test[c] = "int8"
    else:
        dtype_train[c] = "float32"
        dtype_test[c] = "float32"

usecols_train = [id_col] + feature_cols + [target_col]
usecols_test = [id_col] + feature_cols

train_df = pd.read_csv(TRAIN_PATH, usecols=usecols_train, dtype=dtype_train)
test_df = pd.read_csv(TEST_PATH, usecols=usecols_test, dtype=dtype_test)



## === cell 2
y = train_df["Cover_Type"].astype(np.int32) - 1

train_df.drop(["Cover_Type", "Id"], axis=1, inplace=True)
X = train_df  # view
X_test = test_df.drop(["Id"], axis=1)

print(X.shape, y.shape, X_test.shape)
print(
    "y unique (0-based):",
    np.unique(y),
    "counts(min/max):",
    np.bincount(y, minlength=7).min(),
    np.bincount(y, minlength=7).max(),
)

y_min, y_max = int(y.min()), int(y.max())
assert (
    y_min >= 0 and y_max <= 6
), f"Unexpected y range after 0-basing: [{y_min}, {y_max}] (expected [0, 6])"



## === cell 3
from sklearn.model_selection import train_test_split

class_counts = np.bincount(y, minlength=7)
rare_classes = np.where(class_counts < 2)[0]

if rare_classes.size > 0:
    rare_mask = np.isin(y, rare_classes)
    common_mask = ~rare_mask

    X_rare = X.loc[rare_mask]
    y_rare = y.loc[rare_mask]
    X_common = X.loc[common_mask]
    y_common = y.loc[common_mask]

    x_train_c, x_val, y_train_c, y_val = train_test_split(
        X_common,
        y_common,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y_common,
    )

    x_train = pd.concat([x_train_c, X_rare], axis=0)
    y_train = pd.concat([y_train_c, y_rare], axis=0)

    perm = np.random.RandomState(42).permutation(len(x_train))
    x_train = x_train.iloc[perm]
    y_train = y_train.iloc[perm]

    print("Rare classes forced into training set:", rare_classes.tolist())
else:
    x_train, x_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=y,
    )

print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
print("y_train unique:", np.unique(y_train), "y_val unique:", np.unique(y_val))
print("y_train bincount:", np.bincount(y_train, minlength=7))
print("y_val bincount:", np.bincount(y_val, minlength=7))

assert (
    np.unique(y_train).size == 7
), f"Training split is missing classes: {np.unique(y_train)}"
assert np.unique(y_val).size >= 2, "Validation split unexpectedly has <2 classes."

x_train_np = np.ascontiguousarray(x_train.to_numpy(dtype=np.float32, copy=False))
x_val_np = np.ascontiguousarray(x_val.to_numpy(dtype=np.float32, copy=False))
y_train_np = y_train.to_numpy(dtype=np.int32, copy=False)
y_val_np = y_val.to_numpy(dtype=np.int32, copy=False)

X_test_np = np.ascontiguousarray(X_test.to_numpy(dtype=np.float32, copy=False))



## === cell 4
from xgboost import XGBClassifier, callback

model_xgbc = XGBClassifier(
    n_estimators=20000,
    n_jobs=4,
    learning_rate=0.1,
    objective="multi:softprob",
    num_class=7,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
)

model_xgbc.fit(
    x_train_np,
    y_train_np,
    eval_set=[(x_val_np, y_val_np)],
    callbacks=[callback.EarlyStopping(rounds=5, save_best=True)],
    verbose=50,
)



## === cell 5
y_predict_xgbc = model_xgbc.predict(X_test_np).astype(np.int32) + 1

print(
    y_predict_xgbc[:10],
    np.unique(y_predict_xgbc),
    "min/max:",
    int(y_predict_xgbc.min()),
    int(y_predict_xgbc.max()),
)

result = pd.DataFrame(
    {
        "Id": test_df["Id"].astype(np.int64),
        "Cover_Type": y_predict_xgbc.astype(np.int64),
    }
)
print(result.head())

assert result.shape[0] == test_df.shape[0], "Submission row count mismatch vs test set."
assert result.columns.tolist() == [
    "Id",
    "Cover_Type",
], "Submission columns must be: Id, Cover_Type"
assert result["Cover_Type"].between(1, 7).all(), "Cover_Type must be in [1, 7]"

result.to_csv("/kaggle/working/submission.csv", index=False)
print("Done, wrote /kaggle/working/submission.csv")
print(result.shape)
print(result.head())
