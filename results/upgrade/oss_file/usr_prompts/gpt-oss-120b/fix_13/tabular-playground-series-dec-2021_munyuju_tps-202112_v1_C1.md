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
lightgbm==4.6.0
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier

file_count = sum(1 for _, _, filenames in os.walk("/kaggle/input") for _ in filenames)
print(f"Total files found: {file_count}")



## === cell 1
train_path = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
test_path = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"
sample_sub_path = (
    "/kaggle/input/tabular-playground-series-dec-2021/sample_submission.csv"
)

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_sub_path)

test_ids = test_df["Id"].values



## === cell 2
X_raw = train_df.drop(["Cover_Type", "Id"], axis=1).values.astype(np.float32)
y = train_df["Cover_Type"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

X_test_raw = test_df.drop(["Id"], axis=1).values.astype(np.float32)
X_test_scaled = scaler.transform(X_test_raw).astype(np.float32)

del X_raw, X_test_raw, train_df, test_df



## === cell 3
X_tr, X_val, y_tr, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, shuffle=True
)

model = LGBMClassifier(
    n_estimators=3000,
    learning_rate=0.05,
    random_state=63,
    n_jobs=-1,
    objective="multiclass",
    class_weight="balanced",
    verbose=-1,
)

model.fit(
    X_tr,
    y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="multi_error",
    early_stopping_rounds=200,
    verbose=False,
)

val_pred = model.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation accuracy: {val_acc:.5f}")



## === cell 4
if hasattr(model, "best_iteration_") and model.best_iteration_ is not None:
    best_iter = model.best_iteration_
else:
    best_iter = model.n_estimators

full_estimators = int(best_iter * 1.10)

model_full = LGBMClassifier(
    n_estimators=full_estimators,
    learning_rate=0.05,
    random_state=63,
    n_jobs=-1,
    objective="multiclass",
    class_weight="balanced",
    verbose=-1,
)

model_full.fit(X_scaled, y)



## === cell 5
test_pred = model_full.predict(X_test_scaled)
submission = pd.DataFrame(
    {
        "Id": test_ids,
        "Cover_Type": test_pred,
    }
)



## === cell 6
submission.to_csv("submission.csv", index=False)
print("submission.csv written, shape:", submission.shape)
