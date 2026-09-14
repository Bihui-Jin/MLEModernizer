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
import numpy as np
import pandas as pd
import os

INPUT_DIR = "/kaggle/input/tabular-playground-series-dec-2021"

np.random.seed(42)



## === cell 1
s_data = pd.read_csv(
    f"{INPUT_DIR}/sample_submission.csv",
    usecols=["Id", "Cover_Type"],
)
s_data.head()



## === cell 2
train_path = f"{INPUT_DIR}/train.csv"

train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
target = "Cover_Type"
features = [c for c in train_cols if c not in ("Id", target)]

bin_cols = [
    c for c in features if c.startswith("Wilderness_Area") or c.startswith("Soil_Type")
]

dtype_train = {"Id": np.int32, target: np.int8}
for c in features:
    dtype_train[c] = np.int8 if c in bin_cols else np.float32

train_data = None



## === cell 3
test_path = f"{INPUT_DIR}/test.csv"

test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()
test_features = [c for c in test_cols if c != "Id"]

dtype_test = {"Id": np.int32}
for c in test_features:
    dtype_test[c] = np.int8 if c in bin_cols else np.float32

test_data = None



## === cell 4
pass



## === cell 5
pass



## === cell 6
train_rows = 0
train_na = 0
for chunk in pd.read_csv(
    train_path,
    dtype=dtype_train,
    engine="c",
    low_memory=False,
    memory_map=True,
    chunksize=250_000,
):
    train_rows += len(chunk)
    train_na += int(chunk.isna().sum().sum())

test_rows = 0
test_na = 0
for chunk in pd.read_csv(
    test_path,
    dtype=dtype_test,
    engine="c",
    low_memory=False,
    memory_map=True,
    chunksize=250_000,
):
    test_rows += len(chunk)
    test_na += int(chunk.isna().sum().sum())

print(
    "Shape of Train DF -", (train_rows, len(train_cols) - 1)
)  # exclude Id as index equivalent
print("Shape of Test DF -", (test_rows, len(test_cols)))
print("NA values in Train DF :", train_na)
print("NA values in Test DF :", test_na)



## === cell 7
target = "Cover_Type"
features = [col for col in train_cols if col not in ("Id", target)]
features[:10], len(features)



## === cell 8
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(copy=False)

n_train = train_rows
n_test = test_rows
n_features = len(features)

X_train = np.empty((n_train, n_features), dtype=np.float32)
y = np.empty((n_train,), dtype=np.int8)

row0 = 0
for chunk in pd.read_csv(
    train_path,
    dtype=dtype_train,
    engine="c",
    low_memory=False,
    memory_map=True,
    usecols=["Id"] + features + [target],
    chunksize=250_000,
):
    m = len(chunk)
    Xb = chunk[features].to_numpy(dtype=np.float32, copy=False)
    yb = chunk[target].to_numpy(copy=False)
    X_train[row0 : row0 + m] = Xb
    y[row0 : row0 + m] = yb
    scaler.partial_fit(Xb)
    row0 += m

X_train = scaler.transform(X_train)

X_test = np.empty((n_test, n_features), dtype=np.float32)
test_ids = np.empty((n_test,), dtype=np.int32)

row0 = 0
for chunk in pd.read_csv(
    test_path,
    dtype=dtype_test,
    engine="c",
    low_memory=False,
    memory_map=True,
    usecols=["Id"] + features,
    chunksize=250_000,
):
    m = len(chunk)
    test_ids[row0 : row0 + m] = chunk["Id"].to_numpy(copy=False)
    X_test[row0 : row0 + m] = chunk[features].to_numpy(dtype=np.float32, copy=False)
    row0 += m

X_test = scaler.transform(X_test)



## === cell 9
print(f"Shape of data X - {X_train.shape}, y - {y.shape} and X_test - {X_test.shape}")



## === cell 10
catb_params = {
    "objective": "MultiClass",
    "task_type": "CPU",
    "random_seed": 42,
    "verbose": 0,
    "thread_count": -1,
}



## === cell 11
from catboost import CatBoostClassifier

model = CatBoostClassifier(**catb_params)
model.fit(X_train, y)



## === cell 12
predict = model.predict(X_test)



## === cell 13
predict = np.asarray(predict).reshape(-1)
predict[:10], predict.shape



## === cell 14
predictions = pd.DataFrame({"Id": test_ids, "Cover_Type": predict.astype(int)})
predictions.to_csv("submission.csv", index=False)

print(predictions.head())
print("Wrote submission.csv with shape:", predictions.shape)
