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
import random
import numpy as np
import pandas as pd

try:
    import datatable as dt
except ModuleNotFoundError:
    dt = None

import sklearn.model_selection as skl_ms
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostClassifier, Pool

SEED = 42
random.seed(SEED)
np.random.seed(SEED)




## === cell 1
def reduce_memory_usage(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df




## === cell 2
train_path = "../input/tabular-playground-series-dec-2021/train.csv"
test_path = "../input/tabular-playground-series-dec-2021/test.csv"


def _build_fast_dtypes(csv_path, has_target):
    cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    dtypes = {}
    for c in cols:
        if c == "Id":
            dtypes[c] = np.int32
        elif has_target and c == "Cover_Type":
            dtypes[c] = np.int16
        else:
            dtypes[c] = np.float32
    return dtypes


if dt is not None:
    train_cols = pd.read_csv(train_path, nrows=0).columns.tolist()
    test_cols = pd.read_csv(test_path, nrows=0).columns.tolist()

    train_types = {}
    for c in train_cols:
        if c == "Id":
            train_types[c] = dt.int32
        elif c == "Cover_Type":
            train_types[c] = dt.int16
        else:
            train_types[c] = dt.float32

    test_types = {}
    for c in test_cols:
        if c == "Id":
            test_types[c] = dt.int32
        else:
            test_types[c] = dt.float32

    train_df = dt.fread(train_path, columns=train_types).to_pandas()
    test_df = dt.fread(test_path, columns=test_types).to_pandas()
else:
    train_df = pd.read_csv(
        train_path, dtype=_build_fast_dtypes(train_path, has_target=True)
    )
    test_df = pd.read_csv(
        test_path, dtype=_build_fast_dtypes(test_path, has_target=False)
    )

mask_not_5 = train_df["Cover_Type"].to_numpy(copy=False) != 5
nr_5 = int((~mask_not_5).sum())
print(f"Nr of cover_type = 5: {nr_5}")
if nr_5:
    train_df = train_df.loc[mask_not_5].reset_index(drop=True)

encoder = LabelEncoder()
train_df["Cover_Type"] = encoder.fit_transform(train_df["Cover_Type"])



## === cell 3
test_size = 0.01  # Use around 0.2 if just testing model, this is for subm
train, test = skl_ms.train_test_split(
    train_df, test_size=test_size, random_state=SEED, shuffle=True
)

y_train = train.pop("Cover_Type")
y_test = test.pop("Cover_Type")

if "Id" in train.columns:
    train.drop(columns=["Id"], inplace=True)
if "Id" in test.columns:
    test.drop(columns=["Id"], inplace=True)

X_train = train
X_test = test

train_pool = Pool(X_train, y_train)
eval_pool = Pool(X_test, y_test)



## === cell 4
n_threads = os.cpu_count() or -1

model = CatBoostClassifier(
    iterations=5000,
    task_type="CPU",
    thread_count=n_threads,
    random_seed=SEED,
    od_type="Iter",
    od_wait=200,  # standard patience; stops only after prolonged no-improvement
    use_best_model=True,
)
model.fit(train_pool, eval_set=eval_pool, verbose=False)



## === cell 5
accuracy = model.score(eval_pool)
print(f"Accuracy of catboost on test data: {accuracy}")



## === cell 6
subm_df = pd.read_csv(
    "../input/tabular-playground-series-dec-2021/sample_submission.csv"
)

if "Id" in test_df.columns:
    test_features = test_df.drop(columns=["Id"])
else:
    test_features = test_df

preds = model.predict(test_features)
preds = np.asarray(preds).ravel().astype(np.int64, copy=False)

subm_df["Cover_Type"] = encoder.inverse_transform(preds)
subm_df.to_csv("Submission CB.csv", index=False)
