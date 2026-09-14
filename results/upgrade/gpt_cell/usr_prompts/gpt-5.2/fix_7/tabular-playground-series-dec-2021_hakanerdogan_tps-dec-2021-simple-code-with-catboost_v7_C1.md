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

No external packages required in the script and installed.

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
import gc
import numpy as np
import pandas as pd

DATA_DIR = "../input/tabular-playground-series-dec-2021"
TRAIN_PATH = f"{DATA_DIR}/train.csv"
TEST_PATH = f"{DATA_DIR}/test.csv"
SUB_PATH = f"{DATA_DIR}/sample_submission.csv"

NUMERIC_FLOAT_COLS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
WILDERNESS_COLS = [f"Wilderness_Area{i}" for i in range(1, 5)]
SOIL_COLS = [f"Soil_Type{i}" for i in range(1, 41)]

TRAIN_COLS = ["Id"] + NUMERIC_FLOAT_COLS + WILDERNESS_COLS + SOIL_COLS + ["Cover_Type"]
TEST_COLS = ["Id"] + NUMERIC_FLOAT_COLS + WILDERNESS_COLS + SOIL_COLS

dtype_train = {"Id": np.int32, "Cover_Type": np.int8}
dtype_test = {"Id": np.int32}
for c in NUMERIC_FLOAT_COLS:
    dtype_train[c] = np.float32
    dtype_test[c] = np.float32
for c in WILDERNESS_COLS + SOIL_COLS:
    dtype_train[c] = np.uint8
    dtype_test[c] = np.uint8

read_csv_kwargs_train = dict(usecols=TRAIN_COLS, dtype=dtype_train)
read_csv_kwargs_test = dict(usecols=TEST_COLS, dtype=dtype_test)

try:
    train = pd.read_csv(TRAIN_PATH, engine="pyarrow", **read_csv_kwargs_train)
    test = pd.read_csv(TEST_PATH, engine="pyarrow", **read_csv_kwargs_test)
except Exception:
    train = pd.read_csv(TRAIN_PATH, **read_csv_kwargs_train)
    test = pd.read_csv(TEST_PATH, **read_csv_kwargs_test)



## === cell 1
FEATURE_COLS = NUMERIC_FLOAT_COLS + WILDERNESS_COLS + SOIL_COLS

y_train = train["Cover_Type"].to_numpy(dtype=np.int32, copy=False) - 1  # classes 0..6

X_train = train[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
X_test = test[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)

del train, test
gc.collect()



## === cell 2
params = {
    "learning_rate": 0.37644647769699235,
    "depth": 10,
    "one_hot_max_size": 4,
    "l2_leaf_reg": 0.05846053355686806,
}



## === cell 3
from catboost import CatBoostClassifier

thread_count = os.cpu_count() or 4

clf_CatBoostClassifier = CatBoostClassifier(
    **params,
    verbose=0,
    task_type="CPU",
    thread_count=thread_count,
    allow_writing_files=False,
    random_seed=0,
)

clf_CatBoostClassifier.fit(X_train, y_train)

pred = clf_CatBoostClassifier.predict(X_test).astype(np.int32) + 1  # shift back

sample_submission = pd.read_csv(SUB_PATH, usecols=["Id"], dtype={"Id": np.int32})
sample_submission["Cover_Type"] = pred.astype(np.int32, copy=False)
sample_submission.to_csv("submission_CatBoostClassifier.csv", index=False)
sample_submission.head()
