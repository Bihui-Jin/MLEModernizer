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
import gc
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

os.environ["OMP_NUM_THREADS"] = "5"




## === cell 1
train_path = "../input/tabular-playground-series-dec-2021/train.csv"
test_path = "../input/tabular-playground-series-dec-2021/test.csv"
sample_sub_path = "../input/tabular-playground-series-dec-2021/sample_submission.csv"

header = pd.read_csv(train_path, nrows=0)
all_cols = header.columns.tolist()
target_col = "Cover_Type"
id_col = "Id"

dtype_map = {col: np.float32 for col in all_cols}
dtype_map[target_col] = np.int32
dtype_map[id_col] = np.int32

train_df = pd.read_csv(
    train_path,
    dtype=dtype_map,
    low_memory=False,
)

test_dtype_map = {
    col: np.float32 if col != id_col else np.int32
    for col in all_cols
    if col != target_col
}
test_df = pd.read_csv(
    test_path,
    dtype=test_dtype_map,
    low_memory=False,
)
test_df[id_col] = test_df[id_col].astype(np.int32)

sample_submission = pd.read_csv(sample_sub_path)




## === cell 2
feature_cols = [c for c in train_df.columns if c not in (target_col, id_col)]

X = train_df[feature_cols]  # pandas DataFrame, no extra copy
y = train_df[target_col]  # pandas Series (int32)

del train_df
gc.collect()

X_train_df, X_val_df, y_train_series, y_val_series = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

X_train = X_train_df.values.astype(np.float32, copy=False)
X_val = X_val_df.values.astype(np.float32, copy=False)
y_train = y_train_series.values.astype(np.int32, copy=False)
y_val = y_val_series.values.astype(np.int32, copy=False)

del X, y, X_train_df, X_val_df, y_train_series, y_val_series
gc.collect()

rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    max_features="sqrt",
    n_jobs=5,
    random_state=42,
    min_samples_leaf=1,
    max_samples=0.5,
)

with joblib.parallel_backend("threading", n_jobs=5):
    rf.fit(X_train, y_train)

val_pred = rf.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation Accuracy: {val_acc:.6f}")

del X_train, X_val, y_train, y_val
gc.collect()




## === cell 3
X_test = test_df[feature_cols].values.astype(np.float32, copy=False)

with joblib.parallel_backend("threading", n_jobs=5):
    test_pred = rf.predict(X_test)

submission = pd.DataFrame({id_col: test_df[id_col], target_col: test_pred})
submission.to_csv("submission_final.csv", index=False)
print("Submission saved to submission_final.csv")
