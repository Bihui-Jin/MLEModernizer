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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)



## === cell 1
s_data = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/sample_submission.csv"
)
s_data.head()



## === cell 2
train_path = "/kaggle/input/tabular-playground-series-dec-2021/train.csv"
test_path = "/kaggle/input/tabular-playground-series-dec-2021/test.csv"

target = "Cover_Type"

cont_cols = [
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
cont_set = set(cont_cols)

read_csv_kwargs = dict(engine="c", low_memory=False)

train_cols = pd.read_csv(train_path, nrows=1, **read_csv_kwargs).columns.tolist()
test_cols = pd.read_csv(test_path, nrows=1, **read_csv_kwargs).columns.tolist()

dtype_train = {}
for c in train_cols:
    if c == "Id":
        dtype_train[c] = np.int32
    elif c == target:
        dtype_train[c] = np.uint8
    elif c in cont_set:
        dtype_train[c] = np.float32
    else:
        dtype_train[c] = np.uint8

dtype_test = {}
for c in test_cols:
    if c == "Id":
        dtype_test[c] = np.int32
    elif c in cont_set:
        dtype_test[c] = np.float32
    else:
        dtype_test[c] = np.uint8

train_data = pd.read_csv(train_path, dtype=dtype_train, **read_csv_kwargs)
train_data.set_index("Id", inplace=True)
train_data.head()



## === cell 3
test_data = pd.read_csv(test_path, dtype=dtype_test, **read_csv_kwargs)
test_data.head()



## === cell 4
pass



## === cell 5
pass



## === cell 6
print("Shape of Train DF -", train_data.shape)
print("Shape of Test DF -", test_data.shape)
print("NA values in Train DF :", train_data.isna().sum().sum())
print("NA values in Test DF :", test_data.isna().sum().sum())



## === cell 7
target = "Cover_Type"
features = [col for col in train_data.columns if col != target]
features



## === cell 8
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(copy=False)

X_all = train_data[features].to_numpy(dtype=np.float32, copy=False)
scaler.fit(X_all)  # uses X_all values; no copy inside scaler that changes semantics
scaler.transform(X_all)  # in-place because copy=False and X_all is float32 ndarray

X_test_all = test_data[features].to_numpy(dtype=np.float32, copy=False)
scaler.transform(X_test_all)  # in-place

y = train_data[target].to_numpy(copy=False)



## === cell 9
print(f"Shape of data X - {X_all.shape}, y - {y.shape} and X_test - {X_test_all.shape}")



## === cell 10
catb_params = {
    "objective": "MultiClass",
    "task_type": "GPU",
    "allow_writing_files": False,
    "thread_count": os.cpu_count() or -1,
}



## === cell 11
from catboost import CatBoostClassifier

model = CatBoostClassifier(**catb_params)
try:
    model.fit(X_all, y, early_stopping_rounds=200, verbose=0)
except Exception as e:
    msg = str(e)
    if (
        "CUDA error" in msg
        or "driver version is insufficient" in msg
        or "CUDA driver version is insufficient" in msg
    ):
        catb_params["task_type"] = "CPU"
        model = CatBoostClassifier(**catb_params)
        model.fit(X_all, y, early_stopping_rounds=200, verbose=0)
    else:
        raise



## === cell 12
predict = model.predict(X_test_all)



## === cell 13
predict



## === cell 14
pred = np.asarray(predict).reshape(-1)

predictions = pd.DataFrame(
    {"Id": test_data["Id"].to_numpy(copy=False), "Cover_Type": pred}
)
predictions.to_csv("submission.csv", index=False)

predictions.head()
