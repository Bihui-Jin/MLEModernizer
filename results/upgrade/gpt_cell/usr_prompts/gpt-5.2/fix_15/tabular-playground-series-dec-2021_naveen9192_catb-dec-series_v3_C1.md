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
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)



## === cell 1
s_data = pd.read_csv(
    "/kaggle/input/tabular-playground-series-dec-2021/sample_submission.csv"
)



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

usecols_train = train_cols
usecols_test = test_cols



## === cell 3
pass



## === cell 4
pass



## === cell 5
target = "Cover_Type"
features = [col for col in train_cols if col != target]

y = pd.read_csv(
    train_path, usecols=[target], dtype={target: np.uint8}, **read_csv_kwargs
)[target].to_numpy(copy=False)

test_ids = pd.read_csv(
    test_path, usecols=["Id"], dtype={"Id": np.int32}, **read_csv_kwargs
)["Id"].to_numpy(copy=False)



## === cell 6
try:
    from sklearnex import patch_sklearn  # type: ignore

    patch_sklearn()
except Exception:
    pass

from sklearn.preprocessing import StandardScaler  # noqa: F401

CHUNK_ROWS = 750_000

n_features = len(features)

dtype_train_features = {c: dtype_train[c] for c in features}
dtype_test_features = {c: dtype_test[c] for c in features}

n_total = 0
sum_ = np.zeros(n_features, dtype=np.float64)
sumsq = np.zeros(n_features, dtype=np.float64)

for chunk in pd.read_csv(
    train_path,
    usecols=features,
    dtype=dtype_train_features,
    chunksize=CHUNK_ROWS,
    **read_csv_kwargs,
):
    Xc = chunk.to_numpy(dtype=np.float64, copy=False)
    n_total += Xc.shape[0]
    sum_ += Xc.sum(axis=0)
    sumsq += (Xc * Xc).sum(axis=0)

mean_ = sum_ / n_total
var_ = sumsq / n_total - mean_ * mean_
scale_ = np.sqrt(var_, dtype=np.float64)
scale_[scale_ == 0.0] = 1.0

mean32 = mean_.astype(np.float32, copy=False)
scale32 = scale_.astype(np.float32, copy=False)


def _iter_standardized_train():
    for chunk in pd.read_csv(
        train_path,
        usecols=features,
        dtype=dtype_train_features,
        chunksize=CHUNK_ROWS,
        **read_csv_kwargs,
    ):
        Xc = chunk.to_numpy(dtype=np.float32, copy=False)
        Xc = (Xc - mean32) / scale32
        yield Xc


def _iter_standardized_test():
    for chunk in pd.read_csv(
        test_path,
        usecols=features,
        dtype=dtype_test_features,
        chunksize=CHUNK_ROWS,
        **read_csv_kwargs,
    ):
        Xc = chunk.to_numpy(dtype=np.float32, copy=False)
        Xc = (Xc - mean32) / scale32
        yield Xc




## === cell 7
print(f"Train rows: {n_total}, n_features: {n_features}, test rows: {len(test_ids)}")



## === cell 8
catb_params = {
    "objective": "MultiClass",
    "task_type": "GPU",
    "allow_writing_files": False,
    "thread_count": os.cpu_count() or -1,
    "verbose": 0,
}



## === cell 9
from catboost import CatBoostClassifier, Pool

X_train_path = "/kaggle/working/X_train_std_f32.mmap"
X_test_path = "/kaggle/working/X_test_std_f32.mmap"

X_train = np.memmap(
    X_train_path, mode="w+", dtype=np.float32, shape=(n_total, n_features)
)
row = 0
for Xc in _iter_standardized_train():
    n = Xc.shape[0]
    X_train[row : row + n] = Xc
    row += n
X_train.flush()
del X_train
X_train = np.memmap(
    X_train_path, mode="r", dtype=np.float32, shape=(n_total, n_features)
)

n_test = len(test_ids)
X_test = np.memmap(X_test_path, mode="w+", dtype=np.float32, shape=(n_test, n_features))
row = 0
for Xc in _iter_standardized_test():
    n = Xc.shape[0]
    X_test[row : row + n] = Xc
    row += n
X_test.flush()
del X_test
X_test = np.memmap(X_test_path, mode="r", dtype=np.float32, shape=(n_test, n_features))

train_pool = Pool(data=X_train, label=y)
test_pool = Pool(data=X_test)

model = CatBoostClassifier(**catb_params)

try:
    model.fit(train_pool)
except Exception as e:
    msg = str(e)
    if (
        "CUDA error" in msg
        or "driver version is insufficient" in msg
        or "CUDA driver version is insufficient" in msg
    ):
        catb_params["task_type"] = "CPU"
        model = CatBoostClassifier(**catb_params)
        model.fit(train_pool)
    else:
        raise



## === cell 10
predict = model.predict(test_pool)



## === cell 11
pass



## === cell 12
pred = np.asarray(predict).reshape(-1)

predictions = pd.DataFrame({"Id": test_ids, "Cover_Type": pred})
predictions.to_csv("submission.csv", index=False)
print(predictions.head())
